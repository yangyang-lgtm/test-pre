#include "privateuse/Conversion/Passes.h"
#include "privateuse/Utils/Constants.h"

#define GEN_PASS_DEF_PTRTOMEMREF
#include "privateuse/Conversion/Passes.h.inc"

#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/Transforms/FuncConversions.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Transforms/Patterns.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Transforms/DialectConversion.h"
#include "triton/Dialect/Triton/IR/Dialect.h"

using namespace mlir;
using namespace mlir::triton;

namespace {

bool isPointerLike(Type type)
{
  if (isa<triton::PointerType>(type)) return true;
  auto tensor = dyn_cast<RankedTensorType>(type);
  return tensor && isa<triton::PointerType>(tensor.getElementType());
}

bool isPointerChainOp(Operation* op)
{
  return isa<SplatOp, AddPtrOp, BroadcastOp, ExpandDimsOp>(op);
}

bool hasRemainingPointerUses(triton::FuncOp func)
{
  WalkResult result = func.walk([](Operation* op) {
    if (isPointerChainOp(op)) return WalkResult::advance();
    if (isa<
          scf::ForOp, scf::IfOp, scf::IndexSwitchOp, scf::WhileOp, scf::YieldOp, scf::ConditionOp,
          UnrealizedConversionCastOp>(op))
      return WalkResult::advance();
    if (llvm::any_of(op->getOperandTypes(), isPointerLike)) return WalkResult::interrupt();
    return WalkResult::advance();
  });
  return result.wasInterrupted();
}

class PtrTokenConverter : public TypeConverter
{
public:
  PtrTokenConverter()
  {
    addConversion([](Type type) { return type; });
    addConversion([](triton::PointerType pointer) -> Type {
      return MemRefType::get({ShapedType::kDynamic}, pointer.getPointeeType());
    });
    addConversion([](RankedTensorType tensor) -> std::optional<Type> {
      auto pointer = dyn_cast<triton::PointerType>(tensor.getElementType());
      if (!pointer) return std::nullopt;
      return MemRefType::get({ShapedType::kDynamic}, pointer.getPointeeType());
    });

    auto createUnrealizedCast = [](OpBuilder& builder, Type resultType, ValueRange inputs, Location loc) -> Value {
      return UnrealizedConversionCastOp::create(builder, loc, resultType, inputs).getResult(0);
    };
    addSourceMaterialization(createUnrealizedCast);
    addTargetMaterialization(createUnrealizedCast);
  }
};

struct SplatTokenLowering : OpConversionPattern<SplatOp>
{
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(SplatOp splat, SplatOp::Adaptor adaptor, ConversionPatternRewriter& rewriter) const override
  {
    rewriter.replaceOp(splat, adaptor.getSrc());
    return success();
  }
};

struct BroadcastTokenLowering : OpConversionPattern<BroadcastOp>
{
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
    BroadcastOp broadcast, BroadcastOp::Adaptor adaptor, ConversionPatternRewriter& rewriter
  ) const override
  {
    rewriter.replaceOp(broadcast, adaptor.getSrc());
    return success();
  }
};

struct ExpandDimsTokenLowering : OpConversionPattern<ExpandDimsOp>
{
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(ExpandDimsOp expand, ExpandDimsOp::Adaptor adaptor, ConversionPatternRewriter& rewriter)
    const override
  {
    rewriter.replaceOp(expand, adaptor.getSrc());
    return success();
  }
};

struct AddPtrTokenLowering : OpConversionPattern<AddPtrOp>
{
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(AddPtrOp addptr, AddPtrOp::Adaptor adaptor, ConversionPatternRewriter& rewriter) const override
  {
    rewriter.replaceOp(addptr, adaptor.getPtr());
    return success();
  }
};

struct PtrToMemrefPass : public ::impl::PtrToMemrefBase<PtrToMemrefPass>
{
  using PtrToMemrefBase::PtrToMemrefBase;

  void getDependentDialects(DialectRegistry& registry) const override
  {
    registry.insert<arith::ArithDialect, memref::MemRefDialect, scf::SCFDialect, tensor::TensorDialect>();
  }

  void runOnOperation() override
  {
    ModuleOp module = getOperation();

    llvm::DenseSet<Operation*> marked;
    llvm::DenseSet<Operation*> keepChains;
    module.walk([&](triton::FuncOp func) {
      if (!func->hasAttr(kLoweredAccessAttr)) return;
      marked.insert(func.getOperation());
      if (hasRemainingPointerUses(func)) keepChains.insert(func.getOperation());
    });
    if (marked.empty()) return;

    PtrTokenConverter converter;
    RewritePatternSet patterns(&getContext());
    populateFunctionOpInterfaceTypeConversionPattern<triton::FuncOp>(patterns, converter);
    patterns.add<SplatTokenLowering, BroadcastTokenLowering, ExpandDimsTokenLowering, AddPtrTokenLowering>(
      converter, &getContext()
    );

    ConversionTarget target(getContext());
    target.addLegalDialect<arith::ArithDialect, memref::MemRefDialect, scf::SCFDialect, tensor::TensorDialect>();
    scf::populateSCFStructuralTypeConversionsAndLegality(converter, patterns, target);
    target
      .addDynamicallyLegalOp<scf::ForOp, scf::IfOp, scf::YieldOp, scf::WhileOp, scf::ConditionOp, scf::IndexSwitchOp>(
        [&](Operation* op) -> std::optional<bool> {
          auto func = op->getParentOfType<triton::FuncOp>();
          if (!func || !marked.contains(func.getOperation())) return true;
          if (keepChains.contains(func.getOperation())) return true;
          return std::nullopt;
        }
      );
    target.addDynamicallyLegalOp<triton::FuncOp>([&](triton::FuncOp op) {
      return !marked.contains(op.getOperation()) ||
             converter.isSignatureLegal(cast<FunctionType>(op.getFunctionType()));
    });
    target.addDynamicallyLegalOp<SplatOp, AddPtrOp, BroadcastOp, ExpandDimsOp>([&](Operation* op) {
      auto func = op->getParentOfType<triton::FuncOp>();
      if (!func || !marked.contains(func.getOperation())) return true;
      if (!llvm::any_of(op->getOperandTypes(), isPointerLike) && !llvm::any_of(op->getResultTypes(), isPointerLike))
        return true;
      return keepChains.contains(func.getOperation());
    });

    if (failed(applyPartialConversion(module, target, std::move(patterns)))) {
      signalPassFailure();
      return;
    }

    for (Operation* op : marked) {
      auto func = cast<triton::FuncOp>(op);
      func->removeAttr(kLoweredAccessAttr);
      for (auto [index, arg] : llvm::enumerate(func.getBody().front().getArguments()))
        if (isa<MemRefType>(arg.getType())) func.setArgAttrs(index, DictionaryAttr::get(&getContext()));
    }

    SmallVector<UnrealizedConversionCastOp> casts;
    module.walk([&](UnrealizedConversionCastOp cast) { casts.push_back(cast); });
    reconcileUnrealizedCasts(casts);
  }
};

} // namespace

std::unique_ptr<OperationPass<ModuleOp>> triton::createPtrToMemrefPass()
{
  return std::make_unique<PtrToMemrefPass>();
}
