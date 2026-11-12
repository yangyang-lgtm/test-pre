#include "privateuse/Conversion/Passes.h"
#include "privateuse/Dialect/PUMem/IR/PUMemDialect.h"
#include "privateuse/Utils/Constants.h"

#define GEN_PASS_DEF_LOWERUNSTRUCTUREDACCESS
#include "privateuse/Conversion/Passes.h.inc"

#include "llvm/ADT/SmallVector.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Visitors.h"
#include "triton/Dialect/Triton/IR/Dialect.h"

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

MemRefType offsetBufferType(Value offsets, OpBuilder& builder)
{
  auto type = cast<RankedTensorType>(offsets.getType());
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

struct PointerChain
{
  Value base;
  Value offsets;
};

class UnstructuredLowering
{
public:
  explicit UnstructuredLowering(triton::FuncOp func) : func(func) {}

  void run()
  {
    SmallVector<Operation*> accesses;
    func.walk([&](Operation* op) {
      if (getAccessPointerType(op) && !getAccessMask(op)) accesses.push_back(op);
    });

    bool lowered = false;
    for (Operation* op : accesses) lowered |= lowerAccess(op);
    if (lowered) func->setAttr(kLoweredAccessAttr, UnitAttr::get(func->getContext()));
  }

private:
  BlockArgument findBase(Value ptr)
  {
    if (auto addptr = ptr.getDefiningOp<AddPtrOp>()) return findBase(addptr.getPtr());
    if (auto splat = ptr.getDefiningOp<SplatOp>()) return dyn_cast<BlockArgument>(splat.getSrc());
    if (auto broadcast = ptr.getDefiningOp<BroadcastOp>()) return findBase(broadcast.getSrc());
    if (auto expand = ptr.getDefiningOp<ExpandDimsOp>()) return findBase(expand.getSrc());
    return {};
  }

  bool lowerAccess(Operation* op)
  {
    Value ptr;
    if (auto load = dyn_cast<LoadOp>(op))
      ptr = load.getPtr();
    else
      ptr = cast<StoreOp>(op).getPtr();

    auto pointerType = cast<RankedTensorType>(ptr.getType());
    auto pointee     = cast<triton::PointerType>(pointerType.getElementType()).getPointeeType();

    BlockArgument base = findBase(ptr);
    if (!base || base.getOwner() != &func.getBody().front()) return false;
    auto baseType = dyn_cast<triton::PointerType>(base.getType());
    if (!baseType || baseType.getPointeeType() != pointee) return false;

    OpBuilder builder(op);
    Location loc       = op->getLoc();
    PointerChain chain = buildChain(ptr, base, builder, loc);

    if (auto load = dyn_cast<LoadOp>(op))
      lowerLoad(load, chain);
    else
      lowerStore(cast<StoreOp>(op), chain);
    return true;
  }

  PointerChain buildChain(Value ptr, BlockArgument base, OpBuilder& builder, Location loc)
  {
    if (auto addptr = ptr.getDefiningOp<AddPtrOp>()) {
      PointerChain chain = buildChain(addptr.getPtr(), base, builder, loc);
      Value offsets      = addptr.getOffset();
      if (!chain.offsets) return {base, offsets};

      Type chainType  = cast<RankedTensorType>(chain.offsets.getType()).getElementType();
      Type offsetType = cast<RankedTensorType>(offsets.getType()).getElementType();
      Type type       = mergeOffsetType(chainType, offsetType, builder);
      Value lhs       = castOffset(chain.offsets, type, builder, loc);
      Value rhs       = castOffset(offsets, type, builder, loc);
      return {base, arith::AddIOp::create(builder, loc, lhs, rhs)};
    }
    if (isa<SplatOp>(ptr.getDefiningOp())) return {base, Value{}};
    if (auto broadcast = ptr.getDefiningOp<BroadcastOp>()) {
      PointerChain chain = buildChain(broadcast.getSrc(), base, builder, loc);
      if (!chain.offsets) return {base, Value{}};

      auto shape       = cast<RankedTensorType>(broadcast.getType()).getShape();
      auto elementType = cast<RankedTensorType>(chain.offsets.getType()).getElementType();
      auto type        = RankedTensorType::get(shape, elementType);
      return {base, BroadcastOp::create(builder, loc, type, chain.offsets)};
    }
    auto expand        = cast<ExpandDimsOp>(ptr.getDefiningOp());
    PointerChain chain = buildChain(expand.getSrc(), base, builder, loc);
    if (!chain.offsets) return {base, Value{}};

    auto shape       = cast<RankedTensorType>(expand.getType()).getShape();
    auto elementType = cast<RankedTensorType>(chain.offsets.getType()).getElementType();
    auto type        = RankedTensorType::get(shape, elementType);
    return {base, ExpandDimsOp::create(builder, loc, type, chain.offsets, expand.getAxisAttr())};
  }

  Value materializeBase(const PointerChain& chain, OpBuilder& builder, Location loc)
  {
    auto pointee   = cast<triton::PointerType>(chain.base.getType()).getPointeeType();
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
      builder, loc, offsetBufferType(offsets, builder), offsets, /*read_only=*/true
    );
  }

  void lowerLoad(LoadOp load, const PointerChain& chain)
  {
    OpBuilder builder(load);
    Location loc    = load.getLoc();
    auto tensorType = cast<RankedTensorType>(load.getType());
    Value base      = materializeBase(chain, builder, loc);
    Value indices   = materializeIndices(chain, tensorType, builder, loc);
    Value buf       = memref::AllocOp::create(
      builder, loc, sramBufferType(tensorType, builder), builder.getI64IntegerAttr(kDefaultAlignment)
    );
    pumem::GatherOp::create(builder, loc, base, indices, buf);
    Value tensor =
      bufferization::ToTensorOp::create(builder, loc, tensorType, buf, /*restrict=*/true, /*writable=*/true);
    load.getResult().replaceAllUsesWith(tensor);
    load->erase();
  }

  void lowerStore(StoreOp store, const PointerChain& chain)
  {
    OpBuilder builder(store);
    Location loc    = store.getLoc();
    auto tensorType = cast<RankedTensorType>(store.getValue().getType());
    Value buf       = bufferization::ToBufferOp::create(
      builder, loc, sramBufferType(tensorType, builder), store.getValue(), /*read_only=*/true
    );
    Value base    = materializeBase(chain, builder, loc);
    Value indices = materializeIndices(chain, tensorType, builder, loc);
    pumem::ScatterOp::create(builder, loc, buf, indices, base);
    store->erase();
  }

  triton::FuncOp func;
};

struct LowerUnstructuredAccessPass : public ::impl::LowerUnstructuredAccessBase<LowerUnstructuredAccessPass>
{
  using LowerUnstructuredAccessBase::LowerUnstructuredAccessBase;

  void getDependentDialects(DialectRegistry& registry) const override
  {
    registry.insert<
      arith::ArithDialect, bufferization::BufferizationDialect, memref::MemRefDialect, pumem::PUMemDialect,
      tensor::TensorDialect>();
  }

  void runOnOperation() override
  {
    ModuleOp module = getOperation();
    if (module.walk([](triton::CallOp) { return WalkResult::interrupt(); }).wasInterrupted()) return;

    SmallVector<triton::FuncOp> functions;
    module.walk([&](triton::FuncOp func) { functions.push_back(func); });

    for (triton::FuncOp func : functions) {
      if (!isConvertibleFunction(func)) continue;
      UnstructuredLowering(func).run();
    }
  }
};

} // namespace

std::unique_ptr<OperationPass<ModuleOp>> triton::createLowerUnstructuredAccessPass()
{
  return std::make_unique<LowerUnstructuredAccessPass>();
}
