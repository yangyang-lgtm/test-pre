#include "privateuse/Analysis/AccessAnalysis.h"
#include "privateuse/Conversion/Passes.h"
#include "privateuse/Utils/Constants.h"
#include "privateuse/Utils/Utils.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Visitors.h"
#include "triton/Dialect/Triton/IR/Dialect.h"

#include <algorithm>
#include <limits>

#define GEN_PASS_DEF_LOWERSTRUCTUREDACCESS
#include "privateuse/Conversion/Passes.h.inc"

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

Type getAccessPointerType(Operation* op)
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
  if (auto tensor = dyn_cast<RankedTensorType>(ptr.getType())) {
    if (tensor.getRank() == 0) return {};
    if (!isa<triton::PointerType>(tensor.getElementType())) return {};
    return tensor;
  }
  if (isa<triton::PointerType>(ptr.getType())) return ptr.getType();
  return {};
}

int64_t floorDiv(int64_t lhs, int64_t rhs)
{
  int64_t quotient = lhs / rhs;
  if (lhs % rhs != 0 && (lhs < 0) != (rhs < 0)) --quotient;
  return quotient;
}

int64_t ceilDiv(int64_t lhs, int64_t rhs)
{
  int64_t quotient = lhs / rhs;
  if (lhs % rhs != 0 && (lhs < 0) == (rhs < 0)) ++quotient;
  return quotient;
}

bool isMaskWindowInjective(const AccessInfo& info)
{
  if (!info.mask.present || info.mask.allTrue) return false;

  AccessView window = info.view;
  for (auto [d, dim] : llvm::enumerate(window.dims)) {
    for (const LinearBound& bound : info.mask.dimBounds[d]) {
      if (bound.lower) return false;
      if (!bound.offset->isConst() || bound.offset->value == std::numeric_limits<int64_t>::min()) return false;
      int64_t limit = -bound.offset->value;
      if (!bound.strict && limit == std::numeric_limits<int64_t>::max()) return false;
      int64_t count = bound.strict ? ceilDiv(limit, bound.stride) : floorDiv(limit, bound.stride) + 1;
      dim.size      = std::min(dim.size, std::max<int64_t>(count, 0));
    }
  }
  return window.isInjective();
}

bool isLowerable(const AccessInfo& info, Type pointerType, Operation* op, triton::FuncOp func)
{
  if (info.deferred != DeferredReason::None) return false;
  if (info.mask.scalarDynamic || info.mask.allFalse) return false;
  if (isa<StoreOp>(op) && !info.view.isInjective() && !isMaskWindowInjective(info)) return false;
  auto base = dyn_cast<BlockArgument>(info.view.base);
  if (!base || base.getOwner() != &func.getBody().front()) return false;
  auto baseType = dyn_cast<triton::PointerType>(base.getType());
  if (!baseType) return false;
  auto accessType = cast<triton::PointerType>(
    isa<RankedTensorType>(pointerType) ? cast<RankedTensorType>(pointerType).getElementType() : pointerType
  );
  auto pointee = accessType.getPointeeType();
  if (storageElementType(pointee) != pointee) return false;
  return storageElementType(baseType.getPointeeType()) == storageElementType(pointee);
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

  Value materializeBase(const AccessInfo& info, Type pointee, OpBuilder& builder, Location loc)
  {
    auto tokenType = MemRefType::get({ShapedType::kDynamic}, pointee);
    return UnrealizedConversionCastOp::create(builder, loc, tokenType, info.view.base).getResult(0);
  }

  FailureOr<Value>
  materializeView(OpBuilder& builder, Location loc, const AccessInfo& info, Type pointee, ArrayRef<Value> dimSizes = {})
  {
    Value base = materializeBase(info, pointee, builder, loc);

    ExprPtr total = info.view.offset;
    SmallVector<int64_t> sizes;
    SmallVector<int64_t> strides;
    SmallVector<OpFoldResult> sizeAttrs;
    SmallVector<OpFoldResult> strideAttrs;
    for (auto [d, dim] : llvm::enumerate(info.view.dims)) {
      total         = ScalarExpr::add(std::move(total), dim.start);
      Value dynamic = d < dimSizes.size() ? dimSizes[d] : Value();
      sizes.push_back(dynamic ? ShapedType::kDynamic : dim.size);
      strides.push_back(dim.stride);
      sizeAttrs.push_back(dynamic ? OpFoldResult(dynamic) : OpFoldResult(builder.getIndexAttr(dim.size)));
      strideAttrs.push_back(builder.getIndexAttr(dim.stride));
    }

    FailureOr<Value> offset = analyzer.materializeIndex(total, builder, loc);
    if (failed(offset)) return failure();

    auto layout   = StridedLayoutAttr::get(builder.getContext(), ShapedType::kDynamic, strides);
    auto viewType = MemRefType::get(sizes, pointee, layout);
    return memref::ReinterpretCastOp::create(builder, loc, viewType, base, *offset, sizeAttrs, strideAttrs).getResult();
  }

  LogicalResult lowerScalarLoad(LoadOp load, const AccessInfo& info)
  {
    OpBuilder builder(load);
    Location loc            = load.getLoc();
    Value base              = materializeBase(info, load.getType(), builder, loc);
    FailureOr<Value> offset = analyzer.materializeIndex(info.view.offset, builder, loc);
    if (failed(offset)) return failure();

    Value value = memref::LoadOp::create(builder, loc, base, *offset);
    load.replaceAllUsesWith(value);
    load->erase();
    return success();
  }

  LogicalResult lowerScalarStore(StoreOp store, const AccessInfo& info)
  {
    OpBuilder builder(store);
    Location loc            = store.getLoc();
    Value base              = materializeBase(info, store.getValue().getType(), builder, loc);
    FailureOr<Value> offset = analyzer.materializeIndex(info.view.offset, builder, loc);
    if (failed(offset)) return failure();

    memref::StoreOp::create(builder, loc, store.getValue(), base, *offset);
    store->erase();
    return success();
  }

  LogicalResult lowerLoad(LoadOp load, const AccessInfo& info)
  {
    auto tensorType = dyn_cast<RankedTensorType>(load.getType());
    if (!tensorType) return lowerScalarLoad(load, info);

    OpBuilder builder(load);
    Location loc = load.getLoc();
    bool masked  = info.mask.present && !info.mask.allTrue;
    SmallVector<Value> dimSizes;
    if (masked) {
      FailureOr<SmallVector<Value>> sizes = analyzer.materializeMaskSizes(info, builder, loc);
      if (failed(sizes)) return failure();
      dimSizes = std::move(*sizes);
    }

    Type elementType      = tensorType.getElementType();
    FailureOr<Value> view = materializeView(builder, loc, info, elementType, dimSizes);
    if (failed(view)) return failure();

    Value buf = memref::AllocOp::create(
      builder, loc, sramBufferType(tensorType, builder), builder.getI64IntegerAttr(kDefaultAlignment)
    );
    Value dst = buf;
    if (masked) {
      Value fillValue = triton::maskedFillValue(load, elementType, builder, loc);
      if (!fillValue) return failure();
      linalg::FillOp::create(builder, loc, ValueRange{fillValue}, ValueRange{buf});
      dst = createBufferWindow(builder, loc, buf, tensorType.getShape(), dimSizes);
    }
    memref::CopyOp::create(builder, loc, *view, dst);
    Value tensor =
      bufferization::ToTensorOp::create(builder, loc, tensorType, buf, /*restrict=*/true, /*writable=*/true);
    load.getResult().replaceAllUsesWith(tensor);
    load->erase();
    return success();
  }

  LogicalResult lowerStore(StoreOp store, const AccessInfo& info)
  {
    auto tensorType = dyn_cast<RankedTensorType>(store.getValue().getType());
    if (!tensorType) return lowerScalarStore(store, info);

    OpBuilder builder(store);
    Location loc = store.getLoc();
    bool masked  = info.mask.present && !info.mask.allTrue;
    SmallVector<Value> dimSizes;
    if (masked) {
      FailureOr<SmallVector<Value>> sizes = analyzer.materializeMaskSizes(info, builder, loc);
      if (failed(sizes)) return failure();
      dimSizes = std::move(*sizes);
    }

    FailureOr<Value> view = materializeView(builder, loc, info, tensorType.getElementType(), dimSizes);
    if (failed(view)) return failure();

    Value buf = bufferization::ToBufferOp::create(
      builder, loc, sramBufferType(tensorType, builder), store.getValue(), /*read_only=*/true
    );
    Value src = masked ? createBufferWindow(builder, loc, buf, tensorType.getShape(), dimSizes) : buf;
    memref::CopyOp::create(builder, loc, src, *view);
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
    // clang-format off
    registry.insert<
      arith::ArithDialect,
      bufferization::BufferizationDialect,
      linalg::LinalgDialect,
      memref::MemRefDialect,
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
