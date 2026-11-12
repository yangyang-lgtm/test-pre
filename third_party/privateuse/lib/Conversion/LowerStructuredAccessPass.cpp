#include "privateuse/Analysis/AccessAnalysis.h"
#include "privateuse/Conversion/Passes.h"
#include "privateuse/Utils/Constants.h"
#define GEN_PASS_DEF_LOWERSTRUCTUREDACCESS
#include "privateuse/Conversion/Passes.h.inc"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
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

using namespace mlir;
using namespace mlir::triton;

namespace {

llvm::DenseSet<Operation*> liveAddressOps(triton::FuncOp func)
{
  SmallVector<Value> worklist;
  func.walk([&](Operation* op) {
    if (auto load = dyn_cast<LoadOp>(op)) {
      worklist.push_back(load.getPtr());
      if (Value mask = load.getMask()) worklist.push_back(mask);
      return;
    }
    if (auto store = dyn_cast<StoreOp>(op)) {
      worklist.push_back(store.getPtr());
      if (Value mask = store.getMask()) worklist.push_back(mask);
    }
  });

  llvm::DenseSet<Operation*> ops;
  while (!worklist.empty()) {
    Operation* def = worklist.pop_back_val().getDefiningOp();
    if (!def || !func->isProperAncestor(def) || !ops.insert(def).second) continue;
    for (Value operand : def->getOperands()) worklist.push_back(operand);
  }
  return ops;
}

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

bool isLowerable(const AccessInfo& info, RankedTensorType pointerType, Operation* op, triton::FuncOp func)
{
  if (info.deferred != DeferredReason::None) return false;
  if (info.mask.present && !info.mask.allTrue) return false;
  if (isa<StoreOp>(op) && !info.view.isInjective()) return false;
  auto base = dyn_cast<BlockArgument>(info.view.base);
  if (!base || base.getOwner() != &func.getBody().front()) return false;
  auto baseType = dyn_cast<triton::PointerType>(base.getType());
  if (!baseType) return false;
  auto pointee = cast<triton::PointerType>(pointerType.getElementType()).getPointeeType();
  return baseType.getPointeeType() == pointee;
}

MemRefType sramBufferType(RankedTensorType tensorType, OpBuilder& builder)
{
  return MemRefType::get(
    tensorType.getShape(), tensorType.getElementType(), MemRefLayoutAttrInterface(),
    builder.getI64IntegerAttr(kSramSpace)
  );
}

class AccessLowering
{
public:
  AccessLowering(triton::FuncOp func, ModuleOp module) : func(func), analyzer(module) {}

  LogicalResult run()
  {
    if (!collectInfos()) return success();
    for (const auto& [op, info] : infos)
      if (failed(lowerAccess(op, info))) return failure();
    func->setAttr(kLoweredAccessAttr, UnitAttr::get(func->getContext()));
    lowerSplats();
    return success();
  }

private:
  bool collectInfos()
  {
    ScopedDiagnosticHandler handler(func->getContext(), [](Diagnostic&) {});

    size_t count = infos.size();
    func.walk([&](Operation* op) {
      auto pointerType = getAccessPointerType(op);
      if (!pointerType) return;

      FailureOr<AccessInfo> info = analyzer.analyze(op);
      if (failed(info) || !isLowerable(*info, pointerType, op, func)) return;
      infos.try_emplace(op, std::move(*info));
    });
    return infos.size() != count;
  }

  FailureOr<Value> materializeView(OpBuilder& builder, Location loc, const AccessInfo& info)
  {
    auto pointee   = cast<triton::PointerType>(info.view.base.getType()).getPointeeType();
    auto tokenType = MemRefType::get({ShapedType::kDynamic}, pointee);
    Value base     = UnrealizedConversionCastOp::create(builder, loc, tokenType, info.view.base).getResult(0);

    ExprPtr total = info.view.offset;
    SmallVector<int64_t> sizes;
    SmallVector<int64_t> strides;
    SmallVector<OpFoldResult> sizeAttrs;
    SmallVector<OpFoldResult> strideAttrs;
    for (const AccessDim& dim : info.view.dims) {
      total = ScalarExpr::add(std::move(total), dim.start);
      sizes.push_back(dim.size);
      strides.push_back(dim.stride);
      sizeAttrs.push_back(builder.getIndexAttr(dim.size));
      strideAttrs.push_back(builder.getIndexAttr(dim.stride));
    }

    FailureOr<Value> offset = analyzer.materializeIndex(total, builder, loc);
    if (failed(offset)) return failure();

    auto layout   = StridedLayoutAttr::get(builder.getContext(), ShapedType::kDynamic, strides);
    auto viewType = MemRefType::get(sizes, pointee, layout);
    return memref::ReinterpretCastOp::create(builder, loc, viewType, base, *offset, sizeAttrs, strideAttrs).getResult();
  }

  LogicalResult lowerLoad(LoadOp load, const AccessInfo& info)
  {
    OpBuilder builder(load);
    Location loc          = load.getLoc();
    FailureOr<Value> view = materializeView(builder, loc, info);
    if (failed(view)) return failure();

    auto tensorType = cast<RankedTensorType>(load.getType());
    Value buf       = memref::AllocOp::create(
      builder, loc, sramBufferType(tensorType, builder), builder.getI64IntegerAttr(kDefaultAlignment)
    );
    memref::CopyOp::create(builder, loc, *view, buf);
    Value tensor =
      bufferization::ToTensorOp::create(builder, loc, tensorType, buf, /*restrict=*/true, /*writable=*/true);
    load.getResult().replaceAllUsesWith(tensor);
    load->erase();
    return success();
  }

  LogicalResult lowerStore(StoreOp store, const AccessInfo& info)
  {
    OpBuilder builder(store);
    Location loc          = store.getLoc();
    FailureOr<Value> view = materializeView(builder, loc, info);
    if (failed(view)) return failure();

    auto tensorType = cast<RankedTensorType>(store.getValue().getType());
    Value buf       = bufferization::ToBufferOp::create(
      builder, loc, sramBufferType(tensorType, builder), store.getValue(), /*read_only=*/true
    );
    memref::CopyOp::create(builder, loc, buf, *view);
    store->erase();
    return success();
  }

  LogicalResult lowerAccess(Operation* op, const AccessInfo& info)
  {
    if (auto load = dyn_cast<LoadOp>(op)) return lowerLoad(load, info);
    return lowerStore(cast<StoreOp>(op), info);
  }

  void lowerSplats()
  {
    llvm::DenseSet<Operation*> liveAddress = liveAddressOps(func);
    SmallVector<SplatOp> splats;
    func.walk([&](SplatOp splat) { splats.push_back(splat); });

    for (SplatOp splat : splats) {
      if (liveAddress.contains(splat.getOperation())) continue;
      auto tensorType = cast<RankedTensorType>(splat.getType());
      if (isa<triton::PointerType>(tensorType.getElementType())) continue;
      OpBuilder builder(splat);
      auto result = tensor::SplatOp::create(builder, splat.getLoc(), splat.getSrc(), tensorType);
      splat.replaceAllUsesWith(result.getResult());
      splat->erase();
    }
  }

  triton::FuncOp func;
  AccessAnalyzer analyzer;
  llvm::DenseMap<Operation*, AccessInfo> infos;
};

struct LowerStructuredAccessPass : public ::impl::LowerStructuredAccessBase<LowerStructuredAccessPass>
{
  using LowerStructuredAccessBase::LowerStructuredAccessBase;

  void getDependentDialects(DialectRegistry& registry) const override
  {
    registry
      .insert<arith::ArithDialect, bufferization::BufferizationDialect, memref::MemRefDialect, tensor::TensorDialect>();
  }

  void runOnOperation() override
  {
    ModuleOp module = getOperation();
    if (module.walk([](triton::CallOp) { return WalkResult::interrupt(); }).wasInterrupted()) return;

    SmallVector<triton::FuncOp> functions;
    module.walk([&](triton::FuncOp func) { functions.push_back(func); });

    for (triton::FuncOp func : functions) {
      if (!isConvertibleFunction(func)) continue;
      if (failed(AccessLowering(func, module).run())) {
        signalPassFailure();
        return;
      }
    }
  }
};

} // namespace

std::unique_ptr<OperationPass<ModuleOp>> triton::createLowerStructuredAccessPass()
{
  return std::make_unique<LowerStructuredAccessPass>();
}
