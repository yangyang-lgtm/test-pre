#include "privateuse/Analysis/AccessAnalysis.h"
#include "privateuse/Conversion/Passes.h"
#include "privateuse/Dialect/PUMem/IR/PUMemDialect.h"
#include "privateuse/Utils/Constants.h"
#include "privateuse/Utils/Utils.h"

#include "llvm/ADT/SmallVector.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Visitors.h"
#include "triton/Dialect/Triton/IR/Dialect.h"

#define GEN_PASS_DEF_LOWERUNSTRUCTUREDACCESS
#include "privateuse/Conversion/Passes.h.inc"

using namespace mlir;
using namespace mlir::triton;

namespace {

bool isConvertibleFunction(triton::FuncOp func)
{
  if (func.isExternal()) return false;
  for (Type type : func.getFunctionType().getInputs())
    if (isa<TensorType>(type)) return false;
  for (Type type : func.getFunctionType().getResults())
    if (isa<TensorType, triton::PointerType>(type)) return false;
  return true;
}

RankedTensorType getAccessPointerType(Operation* op)
{
  Value ptr;
  if (auto load = dyn_cast<LoadOp>(op)) {
    if (load.getIsVolatile()) return {};
    ptr = load.getPtr();
  } else if (auto store = dyn_cast<StoreOp>(op)) {
    if (store.getIgnoreCta()) return {};
    ptr = store.getPtr();
  } else {
    return {};
  }
  auto type = dyn_cast<RankedTensorType>(ptr.getType());
  if (!type || type.getRank() == 0) return {};
  if (!isa<triton::PointerType>(type.getElementType())) return {};
  return type;
}

Value getAccessMask(Operation* op)
{
  if (auto load = dyn_cast<LoadOp>(op)) return load.getMask();
  return cast<StoreOp>(op).getMask();
}

MemRefType sramBufferType(RankedTensorType tensorType, OpBuilder& builder)
{
  return MemRefType::get(
    tensorType.getShape(), tensorType.getElementType(), MemRefLayoutAttrInterface(),
    builder.getI64IntegerAttr(kSramSpace)
  );
}

MemRefType tensorBufferType(Value tensor, OpBuilder& builder)
{
  auto type = cast<RankedTensorType>(tensor.getType());
  return MemRefType::get(
    type.getShape(), type.getElementType(), MemRefLayoutAttrInterface(), builder.getI64IntegerAttr(kSramSpace)
  );
}

Type mergeOffsetType(Type lhs, Type rhs, Builder& builder)
{
  if (lhs.isInteger(64) || rhs.isInteger(64)) return builder.getI64Type();
  return builder.getI32Type();
}

Value castOffset(Value value, Type target, OpBuilder& builder, Location loc)
{
  auto type = cast<RankedTensorType>(value.getType());
  if (type.getElementType() == target) return value;
  auto resultType = RankedTensorType::get(type.getShape(), target);
  // Triton offsets are signed; widening must sign-extend (extui corrupts negative offsets).
  return arith::ExtSIOp::create(builder, loc, resultType, value);
}

Value castScalarOffset(Value value, Type target, OpBuilder& builder, Location loc)
{
  if (value.getType() == target) return value;
  return arith::ExtSIOp::create(builder, loc, target, value);
}

struct PointerChain
{
  Value base;
  Value offsets;
};

class UnstructuredLowering
{
public:
  UnstructuredLowering(triton::FuncOp func, ModuleOp module) : func(func), analyzer(module) {}

  LogicalResult run()
  {
    SmallVector<Operation*> accesses;
    func.walk([&](Operation* op) {
      if (getAccessPointerType(op)) accesses.push_back(op);
    });

    bool lowered = false;
    for (Operation* op : accesses)
      if (failed(lowerAccess(op, lowered))) return failure();
    if (lowered) func->setAttr(kLoweredAccessAttr, UnitAttr::get(func->getContext()));
    return success();
  }

private:
  bool hasEffectiveMask(Operation* op)
  {
    if (!getAccessMask(op)) return false;
    ScopedDiagnosticHandler handler(func->getContext(), [](Diagnostic&) {});
    FailureOr<AccessInfo> info = analyzer.analyze(op);
    return failed(info) || !info->mask.present || !info->mask.allTrue;
  }

  BlockArgument findBase(Value ptr)
  {
    if (auto arg = dyn_cast<BlockArgument>(ptr)) return isa<triton::PointerType>(arg.getType()) ? arg : BlockArgument();

    if (auto bitcast = ptr.getDefiningOp<BitcastOp>()) {
      if (isPtrTypeLike(bitcast.getSrc().getType())) return findBase(bitcast.getSrc());
      return {};
    }
    if (auto addptr = ptr.getDefiningOp<AddPtrOp>()) return findBase(addptr.getPtr());
    if (auto splat = ptr.getDefiningOp<SplatOp>()) return findBase(splat.getSrc());
    if (auto broadcast = ptr.getDefiningOp<BroadcastOp>()) return findBase(broadcast.getSrc());
    if (auto expand = ptr.getDefiningOp<ExpandDimsOp>()) return findBase(expand.getSrc());
    return {};
  }

  LogicalResult lowerAccess(Operation* op, bool& lowered)
  {
    bool masked = hasEffectiveMask(op);
    Value ptr;
    if (auto load = dyn_cast<LoadOp>(op))
      ptr = load.getPtr();
    else
      ptr = cast<StoreOp>(op).getPtr();

    auto pointerType = cast<RankedTensorType>(ptr.getType());
    auto pointee     = cast<triton::PointerType>(pointerType.getElementType()).getPointeeType();
    if (storageElementType(pointee) != pointee) return success();

    BlockArgument base = findBase(ptr);
    if (!base || base.getOwner() != &func.getBody().front()) return success();
    auto baseType = dyn_cast<triton::PointerType>(base.getType());
    if (!baseType || storageElementType(baseType.getPointeeType()) != storageElementType(pointee)) return success();

    OpBuilder builder(op);
    FailureOr<PointerChain> chain = buildChain(ptr, base, builder, op->getLoc());
    if (failed(chain)) return success();

    if (auto load = dyn_cast<LoadOp>(op)) {
      if (failed(lowerLoad(load, *chain, masked, pointee))) return failure();
    } else {
      if (failed(lowerStore(cast<StoreOp>(op), *chain, masked, pointee))) return failure();
    }
    lowered = true;
    return success();
  }

  FailureOr<Value> buildScalarOffsets(Value ptr, BlockArgument base, OpBuilder& builder, Location loc)
  {
    if (ptr == base) return Value{};
    if (auto bitcast = ptr.getDefiningOp<BitcastOp>()) {
      if (!isPtrTypeLike(bitcast.getSrc().getType())) return failure();
      return buildScalarOffsets(bitcast.getSrc(), base, builder, loc);
    }
    auto addptr = ptr.getDefiningOp<AddPtrOp>();
    if (!addptr) return failure();
    FailureOr<Value> inner = buildScalarOffsets(addptr.getPtr(), base, builder, loc);
    if (failed(inner)) return failure();
    Value offsets = addptr.getOffset();
    if (!*inner) return offsets;
    Type type = mergeOffsetType((*inner).getType(), offsets.getType(), builder);
    Value lhs = castScalarOffset(*inner, type, builder, loc);
    Value rhs = castScalarOffset(offsets, type, builder, loc);
    return Value(arith::AddIOp::create(builder, loc, lhs, rhs));
  }

  FailureOr<PointerChain> buildChain(Value ptr, BlockArgument base, OpBuilder& builder, Location loc)
  {
    if (auto addptr = ptr.getDefiningOp<AddPtrOp>()) {
      FailureOr<PointerChain> chain = buildChain(addptr.getPtr(), base, builder, loc);
      if (failed(chain)) return failure();
      Value offsets = addptr.getOffset();
      if (!chain->offsets) return PointerChain{base, offsets};

      Type chainType  = cast<RankedTensorType>(chain->offsets.getType()).getElementType();
      Type offsetType = cast<RankedTensorType>(offsets.getType()).getElementType();
      Type type       = mergeOffsetType(chainType, offsetType, builder);
      Value lhs       = castOffset(chain->offsets, type, builder, loc);
      Value rhs       = castOffset(offsets, type, builder, loc);
      return PointerChain{base, arith::AddIOp::create(builder, loc, lhs, rhs)};
    }
    if (ptr == base) return PointerChain{base, Value{}};
    if (auto splat = ptr.getDefiningOp<SplatOp>()) {
      FailureOr<Value> offsets = buildScalarOffsets(splat.getSrc(), base, builder, loc);
      if (failed(offsets)) return failure();
      if (!*offsets) return PointerChain{base, Value{}};
      auto type = RankedTensorType::get(cast<RankedTensorType>(splat.getType()).getShape(), (*offsets).getType());
      return PointerChain{base, tensor::SplatOp::create(builder, loc, *offsets, type)};
    }
    if (auto broadcast = ptr.getDefiningOp<BroadcastOp>()) {
      FailureOr<PointerChain> chain = buildChain(broadcast.getSrc(), base, builder, loc);
      if (failed(chain)) return failure();
      if (!chain->offsets) return PointerChain{base, Value{}};

      auto shape       = cast<RankedTensorType>(broadcast.getType()).getShape();
      auto elementType = cast<RankedTensorType>(chain->offsets.getType()).getElementType();
      auto type        = RankedTensorType::get(shape, elementType);
      return PointerChain{base, BroadcastOp::create(builder, loc, type, chain->offsets)};
    }
    if (auto bitcast = ptr.getDefiningOp<BitcastOp>()) {
      if (!isPtrTypeLike(bitcast.getSrc().getType())) return failure();
      return buildChain(bitcast.getSrc(), base, builder, loc);
    }
    if (auto expand = ptr.getDefiningOp<ExpandDimsOp>()) {
      FailureOr<PointerChain> chain = buildChain(expand.getSrc(), base, builder, loc);
      if (failed(chain)) return failure();
      if (!chain->offsets) return PointerChain{base, Value{}};

      auto shape       = cast<RankedTensorType>(expand.getType()).getShape();
      auto elementType = cast<RankedTensorType>(chain->offsets.getType()).getElementType();
      auto type        = RankedTensorType::get(shape, elementType);
      return PointerChain{base, ExpandDimsOp::create(builder, loc, type, chain->offsets, expand.getAxisAttr())};
    }
    return failure();
  }

  Value materializeBase(const PointerChain& chain, Type pointee, OpBuilder& builder, Location loc)
  {
    auto tokenType = MemRefType::get({ShapedType::kDynamic}, pointee);
    return UnrealizedConversionCastOp::create(builder, loc, tokenType, chain.base).getResult(0);
  }

  Value materializeIndices(const PointerChain& chain, RankedTensorType accessType, OpBuilder& builder, Location loc)
  {
    Value offsets = chain.offsets;
    if (!offsets) {
      auto type  = RankedTensorType::get(accessType.getShape(), builder.getI32Type());
      Value zero = arith::ConstantOp::create(builder, loc, builder.getI32IntegerAttr(0));
      offsets    = tensor::SplatOp::create(builder, loc, zero, type);
    }
    return bufferization::ToBufferOp::create(
      builder, loc, tensorBufferType(offsets, builder), offsets, /*read_only=*/true
    );
  }

  Value materializeMask(Value mask, RankedTensorType accessType, OpBuilder& builder, Location loc)
  {
    if (!isa<RankedTensorType>(mask.getType())) {
      auto type = RankedTensorType::get(accessType.getShape(), builder.getI1Type());
      mask      = tensor::SplatOp::create(builder, loc, mask, type);
    }
    return bufferization::ToBufferOp::create(builder, loc, tensorBufferType(mask, builder), mask, /*read_only=*/true);
  }

  LogicalResult lowerLoad(LoadOp load, const PointerChain& chain, bool masked, Type pointee)
  {
    OpBuilder builder(load);
    Location loc    = load.getLoc();
    auto tensorType = cast<RankedTensorType>(load.getType());

    Value mask, other;
    if (masked) {
      mask  = materializeMask(load.getMask(), tensorType, builder, loc);
      other = triton::maskedFillValue(load, tensorType.getElementType(), builder, loc);
      if (!other) return failure();
    }

    Value baseToken = materializeBase(chain, pointee, builder, loc);
    Value indices   = materializeIndices(chain, tensorType, builder, loc);
    Value buf       = memref::AllocOp::create(
      builder, loc, sramBufferType(tensorType, builder), builder.getI64IntegerAttr(kDefaultAlignment)
    );
    pumem::GatherOp::create(builder, loc, baseToken, indices, buf, mask, other);

    Value tensor =
      bufferization::ToTensorOp::create(builder, loc, tensorType, buf, /*restrict=*/true, /*writable=*/true);
    load.getResult().replaceAllUsesWith(tensor);
    load->erase();
    return success();
  }

  LogicalResult lowerStore(StoreOp store, const PointerChain& chain, bool masked, Type pointee)
  {
    OpBuilder builder(store);
    Location loc    = store.getLoc();
    auto tensorType = cast<RankedTensorType>(store.getValue().getType());

    Value mask;
    if (masked) mask = materializeMask(store.getMask(), tensorType, builder, loc);

    Value buf = bufferization::ToBufferOp::create(
      builder, loc, sramBufferType(tensorType, builder), store.getValue(), /*read_only=*/true
    );
    Value baseToken = materializeBase(chain, pointee, builder, loc);
    Value indices   = materializeIndices(chain, tensorType, builder, loc);
    pumem::ScatterOp::create(builder, loc, buf, indices, baseToken, mask);
    store->erase();
    return success();
  }

  triton::FuncOp func;
  AccessAnalyzer analyzer;
};

struct LowerUnstructuredAccessPass : public ::impl::LowerUnstructuredAccessBase<LowerUnstructuredAccessPass>
{
  using LowerUnstructuredAccessBase::LowerUnstructuredAccessBase;

  void getDependentDialects(DialectRegistry& registry) const override
  {
    // clang-format off
    registry.insert<
      arith::ArithDialect,
      bufferization::BufferizationDialect,
      memref::MemRefDialect,
      pumem::PUMemDialect,
      tensor::TensorDialect
    >();
    // clang-format on
  }

  void runOnOperation() override
  {
    ModuleOp module = getOperation();
    if (module.walk([](triton::CallOp) { return WalkResult::interrupt(); }).wasInterrupted()) return;

    SmallVector<triton::FuncOp> functions;
    module.walk([&](triton::FuncOp func) { functions.push_back(func); });

    for (triton::FuncOp func : functions) {
      if (!isConvertibleFunction(func)) continue;
      if (failed(UnstructuredLowering(func, module).run())) {
        signalPassFailure();
        return;
      }
    }
  }
};

} // namespace

std::unique_ptr<OperationPass<ModuleOp>> triton::createLowerUnstructuredAccessPass()
{
  return std::make_unique<LowerUnstructuredAccessPass>();
}
