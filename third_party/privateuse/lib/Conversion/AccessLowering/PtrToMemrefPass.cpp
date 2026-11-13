#include "privateuse/Conversion/AccessLowering/Passes.h"
#include "privateuse/Utils/Utils.h"

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

#define GEN_PASS_DEF_PTRTOMEMREF
#include "privateuse/Conversion/AccessLowering/Passes.h.inc"

using namespace mlir;
using namespace mlir::triton;

namespace {

bool isPointerLike(Type type)
{
  if (isa<triton::PointerType>(type)) return true;
  auto tensor = dyn_cast<RankedTensorType>(type);
  return tensor && isa<triton::PointerType>(tensor.getElementType());
}

class PtrTokenConverter : public TypeConverter
{
public:
  PtrTokenConverter()
  {
    addConversion([](Type type) { return type; });
    addConversion([](triton::PointerType pointer) -> Type {
      return MemRefType::get({ShapedType::kDynamic}, storageElementType(pointer.getPointeeType()));
    });
    addConversion([](RankedTensorType tensor) -> std::optional<Type> {
      auto pointer = dyn_cast<triton::PointerType>(tensor.getElementType());
      if (!pointer) return std::nullopt;
      return MemRefType::get({ShapedType::kDynamic}, storageElementType(pointer.getPointeeType()));
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

    PtrTokenConverter converter;
    RewritePatternSet patterns(&getContext());
    populateFunctionOpInterfaceTypeConversionPattern<triton::FuncOp>(patterns, converter);
    patterns.add<SplatTokenLowering, BroadcastTokenLowering, ExpandDimsTokenLowering, AddPtrTokenLowering>(
      converter, &getContext()
    );

    ConversionTarget target(getContext());
    target.addLegalDialect<arith::ArithDialect, memref::MemRefDialect, scf::SCFDialect, tensor::TensorDialect>();
    scf::populateSCFStructuralTypeConversionsAndLegality(converter, patterns, target);
    target.addDynamicallyLegalOp<scf::ForOp, scf::IfOp, scf::YieldOp>([&](Operation* op) -> std::optional<bool> {
      if (!op->getParentOfType<triton::FuncOp>()) return true;
      return std::nullopt;
    });
    target.addDynamicallyLegalOp<triton::FuncOp>([&](triton::FuncOp op) {
      return converter.isSignatureLegal(cast<FunctionType>(op.getFunctionType()));
    });
    target.addDynamicallyLegalOp<SplatOp, AddPtrOp, BroadcastOp, ExpandDimsOp>([&](Operation* op) {
      return !llvm::any_of(op->getOperandTypes(), isPointerLike) && !llvm::any_of(op->getResultTypes(), isPointerLike);
    });

    if (failed(applyPartialConversion(module, target, std::move(patterns)))) {
      signalPassFailure();
      return;
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
