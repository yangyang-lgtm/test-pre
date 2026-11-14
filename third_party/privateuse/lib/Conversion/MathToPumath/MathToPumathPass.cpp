#include "privateuse/Conversion/MathToPumath/MathToPumath.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Func/Transforms/FuncConversions.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/TypeSwitch.h"
#include "privateuse/Conversion/Common/I1ToI8.h"
#include "privateuse/Dialect/PuMath/IR/PuMathDialect.h"

#define DEBUG_TYPE "math-to-pumath"

using namespace mlir;

#define GEN_PASS_DEF_MATHTOPUMATH
#include "privateuse/Conversion/MathToPumath/Passes.h.inc"

namespace {

struct PumathTypeConverter : public TypeConverter
{
  explicit PumathTypeConverter(MLIRContext* context) { triton::addI1ToI8Conversions(*this); }
};

template <typename SrcOp, typename DstOp> struct OneToOneConversion : public OpConversionPattern<SrcOp>
{
  using OpConversionPattern<SrcOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(SrcOp op, typename SrcOp::Adaptor adaptor, ConversionPatternRewriter& rewriter) const override
  {
    SmallVector<Type> resultTypes;
    if (failed(this->getTypeConverter()->convertTypes(op->getResultTypes(), resultTypes))) return failure();

    OperationState state(op.getLoc(), DstOp::getOperationName());
    state.addOperands(adaptor.getOperands());
    state.addTypes(resultTypes);
    for (NamedAttribute attr : op->getAttrs()) state.addAttribute(attr.getName(), convertAttr(attr.getValue()));
    rewriter.replaceOp(op, rewriter.create(state)->getResults());
    return success();
  }

private:
  static Attribute convertAttr(Attribute attr)
  {
    MLIRContext* ctx = attr.getContext();
    return TypeSwitch<Attribute, Attribute>(attr)
      .Case([&](arith::FastMathFlagsAttr a) {
        return puarith::FastMathFlagsAttr::get(
          ctx, static_cast<puarith::FastMathFlags>(a.getValue())
        );
      })
      .Default([](Attribute a) { return a; });
  }
};

struct MathToPumathPass : public ::impl::MathToPumathBase<MathToPumathPass>
{
  using MathToPumathBase::MathToPumathBase;

  void getDependentDialects(DialectRegistry& registry) const override
  {
    registry.insert<math::MathDialect, pumath::PuMathDialect, memref::MemRefDialect, vector::VectorDialect>();
  }

  void runOnOperation() override
  {
    ModuleOp moduleOp = getOperation();

    PumathTypeConverter typeConverter(&getContext());
    RewritePatternSet patterns(&getContext());

    // clang-format off
    patterns.add<
      OneToOneConversion<math::AbsFOp, pumath::AbsFOp>,
      OneToOneConversion<math::AbsIOp, pumath::AbsIOp>,
      OneToOneConversion<math::AcosOp, pumath::AcosOp>,
      OneToOneConversion<math::AcoshOp, pumath::AcoshOp>,
      OneToOneConversion<math::AsinOp, pumath::AsinOp>,
      OneToOneConversion<math::AsinhOp, pumath::AsinhOp>,
      OneToOneConversion<math::AtanOp, pumath::AtanOp>,
      OneToOneConversion<math::Atan2Op, pumath::Atan2Op>,
      OneToOneConversion<math::AtanhOp, pumath::AtanhOp>,
      OneToOneConversion<math::CbrtOp, pumath::CbrtOp>,
      OneToOneConversion<math::CeilOp, pumath::CeilOp>,
      OneToOneConversion<math::ClampFOp, pumath::ClampFOp>,
      OneToOneConversion<math::CopySignOp, pumath::CopySignOp>,
      OneToOneConversion<math::CosOp, pumath::CosOp>,
      OneToOneConversion<math::CoshOp, pumath::CoshOp>,
      OneToOneConversion<math::CountLeadingZerosOp, pumath::CountLeadingZerosOp>,
      OneToOneConversion<math::CountTrailingZerosOp, pumath::CountTrailingZerosOp>,
      OneToOneConversion<math::CtPopOp, pumath::CtPopOp>,
      OneToOneConversion<math::ErfOp, pumath::ErfOp>,
      OneToOneConversion<math::ErfcOp, pumath::ErfcOp>,
      OneToOneConversion<math::ExpOp, pumath::ExpOp>,
      OneToOneConversion<math::Exp2Op, pumath::Exp2Op>,
      OneToOneConversion<math::ExpM1Op, pumath::ExpM1Op>,
      OneToOneConversion<math::FPowIOp, pumath::FPowIOp>,
      OneToOneConversion<math::FloorOp, pumath::FloorOp>,
      OneToOneConversion<math::FmaOp, pumath::FmaOp>,
      OneToOneConversion<math::IPowIOp, pumath::IPowIOp>,
      OneToOneConversion<math::IsFiniteOp, pumath::IsFiniteOp>,
      OneToOneConversion<math::IsInfOp, pumath::IsInfOp>,
      OneToOneConversion<math::IsNaNOp, pumath::IsNaNOp>,
      OneToOneConversion<math::IsNormalOp, pumath::IsNormalOp>,
      OneToOneConversion<math::LogOp, pumath::LogOp>,
      OneToOneConversion<math::Log10Op, pumath::Log10Op>,
      OneToOneConversion<math::Log1pOp, pumath::Log1pOp>,
      OneToOneConversion<math::Log2Op, pumath::Log2Op>,
      OneToOneConversion<math::PowFOp, pumath::PowFOp>,
      OneToOneConversion<math::RoundOp, pumath::RoundOp>,
      OneToOneConversion<math::RoundEvenOp, pumath::RoundEvenOp>,
      OneToOneConversion<math::RsqrtOp, pumath::RsqrtOp>,
      OneToOneConversion<math::SinOp, pumath::SinOp>,
      OneToOneConversion<math::SincosOp, pumath::SincosOp>,
      OneToOneConversion<math::SinhOp, pumath::SinhOp>,
      OneToOneConversion<math::SqrtOp, pumath::SqrtOp>,
      OneToOneConversion<math::TanOp, pumath::TanOp>,
      OneToOneConversion<math::TanhOp, pumath::TanhOp>,
      OneToOneConversion<math::TruncOp, pumath::TruncOp>
    >(typeConverter, &getContext());
    // clang-format on

    populateFunctionOpInterfaceTypeConversionPattern<func::FuncOp>(patterns, typeConverter);
    populateCallOpTypeConversionPattern(patterns, typeConverter);
    populateReturnOpTypeConversionPattern(patterns, typeConverter);
    triton::populateI1ToI8Patterns(patterns, typeConverter);

    ConversionTarget target(getContext());
    target.addIllegalDialect<math::MathDialect>();
    target.addLegalDialect<pumath::PuMathDialect>();
    target.addDynamicallyLegalOp<func::FuncOp>([&](func::FuncOp op) {
      return typeConverter.isSignatureLegal(op.getFunctionType()) && typeConverter.isLegal(&op.getBody());
    });
    target.addDynamicallyLegalOp<func::CallOp, func::ReturnOp>([&](Operation* op) { return typeConverter.isLegal(op); }
    );
    triton::markI1ToI8OpsDynamicallyLegal(target, typeConverter);

    if (failed(applyPartialConversion(moduleOp, target, std::move(patterns)))) signalPassFailure();
  }
};

}

std::unique_ptr<OperationPass<ModuleOp>> triton::createMathToPumathPass()
{
  return std::make_unique<MathToPumathPass>();
}
