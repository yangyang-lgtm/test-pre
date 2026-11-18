#include "privateuse/Conversion/ArithToPuarith/ArithToPuarith.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Func/Transforms/FuncConversions.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/TypeSwitch.h"
#include "privateuse/Conversion/Common/I1ToI8.h"
#include "privateuse/Dialect/PuArith/IR/PuArithDialect.h"

#define GEN_PASS_DECL_ARITHTOPUARITH
#include "privateuse/Conversion/ArithToPuarith/Passes.h.inc"

#define GEN_PASS_DEF_ARITHTOPUARITH
#include "privateuse/Conversion/ArithToPuarith/Passes.h.inc"

#define DEBUG_TYPE "arith-to-puarith"

using namespace mlir;

namespace {

struct PuarithTypeConverter : public TypeConverter
{
  explicit PuarithTypeConverter(MLIRContext* context) { triton::addI1ToI8Conversions(*this); }
};

static Attribute convertAttr(Attribute attr)
{
  MLIRContext* ctx = attr.getContext();
  return TypeSwitch<Attribute, Attribute>(attr)
    .Case([&](arith::FastMathFlagsAttr a) {
      return puarith::FastMathFlagsAttr::get(ctx, static_cast<puarith::FastMathFlags>(a.getValue()));
    })
    .Case([&](arith::IntegerOverflowFlagsAttr a) {
      return puarith::IntegerOverflowFlagsAttr::get(ctx, static_cast<puarith::IntegerOverflowFlags>(a.getValue()));
    })
    .Default([](Attribute a) { return a; });
}

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
};

struct ConstantConversion : public OpConversionPattern<arith::ConstantOp>
{
  using OpConversionPattern<arith::ConstantOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(
    arith::ConstantOp op, arith::ConstantOp::Adaptor adaptor, ConversionPatternRewriter& rewriter
  ) const override
  {
    Type resultType = this->getTypeConverter()->convertType(op.getType());
    if (!resultType) return failure();

    TypedAttr value = cast<TypedAttr>(op.getValue());
    auto constType  = dyn_cast<IntegerType>(op.getType());
    if (constType && constType.getWidth() == 1 && constType.isSignless()) {
      auto intValue = dyn_cast<IntegerAttr>(value);
      if (!intValue) return op.emitOpError("constant type conversion requires an integer attribute");
      value = IntegerAttr::get(resultType, intValue.getValue().zextOrTrunc(8));
    }
    rewriter.replaceOpWithNewOp<puarith::ConstantOp>(op, resultType, value);
    return success();
  }
};

static bool isI1Like(Type type)
{
  if (auto intTy = dyn_cast<IntegerType>(type)) return intTy.getWidth() == 1 && intTy.isSignless();
  if (auto shapedTy = dyn_cast<ShapedType>(type)) {
    auto intTy = dyn_cast<IntegerType>(shapedTy.getElementType());
    return intTy && intTy.getWidth() == 1 && intTy.isSignless();
  }
  return false;
}

struct TruncIToAndI1Conversion : public OpConversionPattern<arith::TruncIOp>
{
  using OpConversionPattern<arith::TruncIOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(
    arith::TruncIOp op, arith::TruncIOp::Adaptor adaptor, ConversionPatternRewriter& rewriter
  ) const override
  {
    if (!isI1Like(op.getType())) return failure();

    Type inTy  = op.getIn().getType();
    Type resTy = this->getTypeConverter()->convertType(op.getType());

    Type elemTy       = getElementTypeOrSelf(inTy);
    Attribute oneElem = IntegerAttr::get(elemTy, APInt(cast<IntegerType>(elemTy).getWidth(), 1));
    TypedAttr one;
    if (auto shapedTy = dyn_cast<ShapedType>(inTy))
      one = DenseElementsAttr::get(shapedTy, cast<TypedAttr>(oneElem));
    else
      one = cast<TypedAttr>(oneElem);
    auto oneCst = puarith::ConstantOp::create(rewriter, op.getLoc(), one);

    auto lowBit = puarith::AndIOp::create(rewriter, op.getLoc(), adaptor.getIn(), oneCst);
    rewriter.replaceOpWithNewOp<puarith::TruncIOp>(op, resTy, lowBit);
    return success();
  }
};

template <typename ExtOp> struct ExtFromI1Erase : public OpConversionPattern<ExtOp>
{
  using OpConversionPattern<ExtOp>::OpConversionPattern;
  using Adaptor = typename ExtOp::Adaptor;

  LogicalResult matchAndRewrite(ExtOp op, Adaptor adaptor, ConversionPatternRewriter& rewriter) const override
  {
    if (!isI1Like(op.getIn().getType())) return failure();

    Type newSrcTy = this->getTypeConverter()->convertType(op.getIn().getType());
    Type newResTy = this->getTypeConverter()->convertType(op.getType());
    if (newSrcTy != newResTy) return failure();

    rewriter.replaceOp(op, adaptor.getIn());
    return success();
  }
};

template <typename SrcOp, typename DstOp, typename CastOp>
struct VectorCmpCastConversion : public OpConversionPattern<SrcOp>
{
  using OpConversionPattern<SrcOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(SrcOp op, typename SrcOp::Adaptor adaptor, ConversionPatternRewriter& rewriter) const override
  {
    SmallVector<Type> resultTypes;
    if (failed(this->getTypeConverter()->convertTypes(op->getResultTypes(), resultTypes))) return failure();
    auto cmpTy = dyn_cast<VectorType>(resultTypes[0]);
    if (!cmpTy) return failure();
    Type computeTy    = this->getTypeConverter()->convertType(op->getOperand(0).getType());
    auto computeVecTy = dyn_cast<VectorType>(computeTy);
    if (!computeVecTy || computeVecTy.getNumElements() != cmpTy.getNumElements() ||
        computeVecTy.getElementType() == cmpTy.getElementType())
      return failure();

    OperationState state(op.getLoc(), DstOp::getOperationName());
    state.addOperands(adaptor.getOperands());
    state.addTypes(computeTy);
    for (NamedAttribute attr : op->getAttrs()) state.addAttribute(attr.getName(), convertAttr(attr.getValue()));
    Value cmp  = rewriter.create(state)->getResult(0);
    Value cast = CastOp::create(rewriter, op.getLoc(), cmpTy, cmp);
    rewriter.replaceOp(op, cast);
    return success();
  }

private:
  static Attribute convertAttr(Attribute attr)
  {
    MLIRContext* ctx = attr.getContext();
    return TypeSwitch<Attribute, Attribute>(attr)
      .Case([&](arith::FastMathFlagsAttr a) {
        return puarith::FastMathFlagsAttr::get(ctx, static_cast<puarith::FastMathFlags>(a.getValue()));
      })
      .Default([](Attribute a) { return a; });
  }
};

struct VectorSelectCastConversion : public OpConversionPattern<arith::SelectOp>
{
  using OpConversionPattern<arith::SelectOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(
    arith::SelectOp op, arith::SelectOp::Adaptor adaptor, ConversionPatternRewriter& rewriter
  ) const override
  {
    SmallVector<Type> resultTypes;
    if (failed(this->getTypeConverter()->convertTypes(op->getResultTypes(), resultTypes))) return failure();
    auto resTy = dyn_cast<VectorType>(resultTypes[0]);
    if (!resTy) return failure();

    Type condTy    = this->getTypeConverter()->convertType(adaptor.getCondition().getType());
    auto condVecTy = dyn_cast<VectorType>(condTy);
    if (!condVecTy || condVecTy.getElementType() == resTy.getElementType()) return failure();

    Type castTy = VectorType::get(condVecTy.getShape(), resTy.getElementType());
    Value cond;
    if (isa<FloatType>(resTy.getElementType()))
      cond = puarith::SIToFPOp::create(rewriter, op.getLoc(), castTy, adaptor.getCondition());
    else
      cond = puarith::ExtSIOp::create(rewriter, op.getLoc(), castTy, adaptor.getCondition());
    rewriter.replaceOpWithNewOp<puarith::SelectOp>(op, resTy, cond, adaptor.getTrueValue(), adaptor.getFalseValue());
    return success();
  }
};

struct ArithToPuarithPass : public ::impl::ArithToPuarithBase<ArithToPuarithPass>
{
  using ArithToPuarithBase::ArithToPuarithBase;

  void getDependentDialects(DialectRegistry& registry) const override
  {
    registry.insert<arith::ArithDialect, puarith::PuArithDialect>();
  }

  void runOnOperation() override
  {
    ModuleOp moduleOp = getOperation();

    PuarithTypeConverter typeConverter(&getContext());
    RewritePatternSet patterns(&getContext());

    patterns.add<ConstantConversion, TruncIToAndI1Conversion>(typeConverter, &getContext(), PatternBenefit(2));
    patterns.add<ExtFromI1Erase<arith::ExtSIOp>, ExtFromI1Erase<arith::ExtUIOp>>(
      typeConverter, &getContext(), PatternBenefit(2)
    );

    patterns.add<
      VectorCmpCastConversion<arith::CmpIOp, puarith::CmpIOp, puarith::TruncIOp>,
      VectorCmpCastConversion<arith::CmpFOp, puarith::CmpFOp, puarith::FPToUIOp>>(
      typeConverter, &getContext(), PatternBenefit(2)
    );

    patterns.add<VectorSelectCastConversion>(typeConverter, &getContext(), PatternBenefit(2));

    // clang-format off
    patterns.add<
      OneToOneConversion<arith::AddIOp, puarith::AddIOp>,
      OneToOneConversion<arith::AddUIExtendedOp, puarith::AddUIExtendedOp>,
      OneToOneConversion<arith::SubIOp, puarith::SubIOp>,
      OneToOneConversion<arith::SubUIExtendedOp, puarith::SubUIExtendedOp>,
      OneToOneConversion<arith::MulIOp, puarith::MulIOp>,
      OneToOneConversion<arith::MulSIExtendedOp, puarith::MulSIExtendedOp>,
      OneToOneConversion<arith::MulUIExtendedOp, puarith::MulUIExtendedOp>,
      OneToOneConversion<arith::DivUIOp, puarith::DivUIOp>,
      OneToOneConversion<arith::DivSIOp, puarith::DivSIOp>,
      OneToOneConversion<arith::CeilDivUIOp, puarith::CeilDivUIOp>,
      OneToOneConversion<arith::CeilDivSIOp, puarith::CeilDivSIOp>,
      OneToOneConversion<arith::FloorDivSIOp, puarith::FloorDivSIOp>,
      OneToOneConversion<arith::RemUIOp, puarith::RemUIOp>,
      OneToOneConversion<arith::RemSIOp, puarith::RemSIOp>,
      OneToOneConversion<arith::AndIOp, puarith::AndIOp>,
      OneToOneConversion<arith::OrIOp, puarith::OrIOp>,
      OneToOneConversion<arith::XOrIOp, puarith::XOrIOp>,
      OneToOneConversion<arith::ShLIOp, puarith::ShLIOp>,
      OneToOneConversion<arith::ShRUIOp, puarith::ShRUIOp>,
      OneToOneConversion<arith::ShRSIOp, puarith::ShRSIOp>,
      OneToOneConversion<arith::NegFOp, puarith::NegFOp>,
      OneToOneConversion<arith::FlushDenormalsOp, puarith::FlushDenormalsOp>,
      OneToOneConversion<arith::AddFOp, puarith::AddFOp>,
      OneToOneConversion<arith::SubFOp, puarith::SubFOp>,
      OneToOneConversion<arith::MulFOp, puarith::MulFOp>,
      OneToOneConversion<arith::DivFOp, puarith::DivFOp>,
      OneToOneConversion<arith::RemFOp, puarith::RemFOp>,
      OneToOneConversion<arith::MaximumFOp, puarith::MaximumFOp>,
      OneToOneConversion<arith::MaxNumFOp, puarith::MaxNumFOp>,
      OneToOneConversion<arith::MaxSIOp, puarith::MaxSIOp>,
      OneToOneConversion<arith::MaxUIOp, puarith::MaxUIOp>,
      OneToOneConversion<arith::MinimumFOp, puarith::MinimumFOp>,
      OneToOneConversion<arith::MinNumFOp, puarith::MinNumFOp>,
      OneToOneConversion<arith::MinSIOp, puarith::MinSIOp>,
      OneToOneConversion<arith::MinUIOp, puarith::MinUIOp>,
      OneToOneConversion<arith::ExtUIOp, puarith::ExtUIOp>,
      OneToOneConversion<arith::ExtSIOp, puarith::ExtSIOp>,
      OneToOneConversion<arith::ExtFOp, puarith::ExtFOp>,
      OneToOneConversion<arith::TruncIOp, puarith::TruncIOp>,
      OneToOneConversion<arith::TruncFOp, puarith::TruncFOp>,
      OneToOneConversion<arith::ConvertFOp, puarith::ConvertFOp>,
      OneToOneConversion<arith::UIToFPOp, puarith::UIToFPOp>,
      OneToOneConversion<arith::SIToFPOp, puarith::SIToFPOp>,
      OneToOneConversion<arith::FPToUIOp, puarith::FPToUIOp>,
      OneToOneConversion<arith::FPToSIOp, puarith::FPToSIOp>,
      OneToOneConversion<arith::IndexCastOp, puarith::IndexCastOp>,
      OneToOneConversion<arith::IndexCastUIOp, puarith::IndexCastUIOp>,
      OneToOneConversion<arith::BitcastOp, puarith::BitcastOp>,
      OneToOneConversion<arith::CmpIOp, puarith::CmpIOp>,
      OneToOneConversion<arith::CmpFOp, puarith::CmpFOp>,
      OneToOneConversion<arith::SelectOp, puarith::SelectOp>
    >(typeConverter, &getContext());
    // clang-format on

    populateFunctionOpInterfaceTypeConversionPattern<func::FuncOp>(patterns, typeConverter);
    populateCallOpTypeConversionPattern(patterns, typeConverter);
    populateReturnOpTypeConversionPattern(patterns, typeConverter);
    triton::populateI1ToI8Patterns(patterns, typeConverter);

    ConversionTarget target(getContext());
    target.addIllegalDialect<arith::ArithDialect>();
    target.addLegalDialect<puarith::PuArithDialect>();
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

std::unique_ptr<OperationPass<ModuleOp>> triton::createArithToPuarithPass()
{
  return std::make_unique<ArithToPuarithPass>();
}
