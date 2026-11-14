#include "privateuse/Conversion/ToPuc/PucPatterns.h"

#include "llvm/ADT/TypeSwitch.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Transforms/DialectConversion.h"
#include "privateuse/Dialect/PuArith/IR/PuArithDialect.h"
#include "privateuse/Dialect/PuMath/IR/PuMathDialect.h"
#include "privateuse/Dialect/Puc/IR/PucDialect.h"

using namespace mlir;

namespace {

std::string typeSuffixName(Type type, bool isUnsigned)
{
  return TypeSwitch<Type, std::string>(getElementTypeOrSelf(type))
    .Case<Float16Type>([](auto) { return "f16"; })
    .Case<BFloat16Type>([](auto) { return "bf16"; })
    .Case<Float32Type>([](auto) { return "f32"; })
    .Case<Float64Type>([](auto) { return "f64"; })
    .Case<IndexType>([](auto) { return "i64"; })
    .Case<IntegerType>([&](IntegerType t) { return (isUnsigned ? "u" : "i") + std::to_string(t.getWidth()); })
    .Default([](Type) { return std::string(""); });
}

bool sourceUnsignedOf(StringRef mnemonic)
{
  if (mnemonic.starts_with("ui")) return true;
  return mnemonic.ends_with("ui") && !mnemonic.starts_with("fpto");
}

bool destinationUnsignedOf(StringRef mnemonic)
{
  return mnemonic.starts_with("fpto") && mnemonic.ends_with("ui");
}

std::string calleeNameOf(Operation* op)
{
  return "scalar_" + op->getName().stripDialect().str();
}

LogicalResult
replaceWithCallee(Operation* op, ValueRange operands, ConversionPatternRewriter& rewriter, StringRef callee)
{
  rewriter.replaceOpWithNewOp<puc::CallOp>(
    op, op->getResultTypes(), FlatSymbolRefAttr::get(op->getContext(), callee), ArrayAttr{}, operands
  );
  return success();
}

LogicalResult
replaceWithCall(Operation* op, ValueRange operands, ConversionPatternRewriter& rewriter, StringRef suffix = {})
{
  std::string callee = calleeNameOf(op) + suffix.str();
  return replaceWithCallee(op, operands, rewriter, callee);
}

struct ConstantConversion : OpConversionPattern<puarith::ConstantOp>
{
  using OpConversionPattern<puarith::ConstantOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(
    puarith::ConstantOp op, puarith::ConstantOp::Adaptor adaptor, ConversionPatternRewriter& rewriter
  ) const override
  {
    auto constant = puc::ConstantOp::materialize(rewriter, adaptor.getValue(), op.getType(), op.getLoc());
    if (!constant) return failure();
    rewriter.replaceOp(op, constant->getResults());
    return success();
  }
};

template <typename OpT> struct IndexCastOpConversion : public OpConversionPattern<OpT>
{
  using OpConversionPattern<OpT>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(OpT op, typename OpT::Adaptor adaptor, ConversionPatternRewriter& rewriter) const override
  {
    return replaceWithCallee(op, adaptor.getOperands(), rewriter, "/*index cast*/");
  }
};

template <typename OpT> struct CastOpConversion : public OpConversionPattern<OpT>
{
  using OpConversionPattern<OpT>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(OpT op, typename OpT::Adaptor adaptor, ConversionPatternRewriter& rewriter) const override
  {
    StringRef mnemonic      = op->getName().stripDialect();
    std::string source      = typeSuffixName(op->getOperand(0).getType(), sourceUnsignedOf(mnemonic));
    std::string destination = typeSuffixName(op->getResult(0).getType(), destinationUnsignedOf(mnemonic));
    if (source.empty() || destination.empty()) return failure();
    return replaceWithCall(op, adaptor.getOperands(), rewriter, "_" + source + "_" + destination);
  }
};

struct CmpIOpConversion : OpConversionPattern<puarith::CmpIOp>
{
  using OpConversionPattern<puarith::CmpIOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(
    puarith::CmpIOp op, puarith::CmpIOp::Adaptor adaptor, ConversionPatternRewriter& rewriter
  ) const override
  {
    std::string suffix = "_" + puarith::stringifyCmpIPredicate(op.getPredicate()).str();
    return replaceWithCall(op, adaptor.getOperands(), rewriter, suffix);
  }
};

struct CmpFOpConversion : OpConversionPattern<puarith::CmpFOp>
{
  using OpConversionPattern<puarith::CmpFOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(
    puarith::CmpFOp op, puarith::CmpFOp::Adaptor adaptor, ConversionPatternRewriter& rewriter
  ) const override
  {
    std::string suffix = "_" + puarith::stringifyCmpFPredicate(op.getPredicate()).str();
    return replaceWithCall(op, adaptor.getOperands(), rewriter, suffix);
  }
};

struct PlainCallConversion : public ConversionPattern
{
  PlainCallConversion(TypeConverter& typeConverter, MLIRContext* context)
    : ConversionPattern(typeConverter, MatchAnyOpTypeTag(), /*benefit=*/0, context)
  {
  }

  LogicalResult
  matchAndRewrite(Operation* op, ArrayRef<Value> operands, ConversionPatternRewriter& rewriter) const override
  {
    StringRef dialect = op->getName().getDialectNamespace();
    if (dialect != "puarith" && dialect != "pumath") return failure();
    return replaceWithCall(op, operands, rewriter);
  }
};

} // namespace

void mlir::triton::populateScalarToPucPatterns(RewritePatternSet& patterns, TypeConverter& typeConverter)
{
  // clang-format off
  patterns.add<
    ConstantConversion,
    IndexCastOpConversion<puarith::IndexCastOp>,
    IndexCastOpConversion<puarith::IndexCastUIOp>,
    CastOpConversion<puarith::ExtSIOp>,
    CastOpConversion<puarith::ExtUIOp>,
    CastOpConversion<puarith::TruncIOp>,
    CastOpConversion<puarith::ExtFOp>,
    CastOpConversion<puarith::TruncFOp>,
    CastOpConversion<puarith::ConvertFOp>,
    CastOpConversion<puarith::FPToSIOp>,
    CastOpConversion<puarith::FPToUIOp>,
    CastOpConversion<puarith::SIToFPOp>,
    CastOpConversion<puarith::UIToFPOp>,
    CastOpConversion<puarith::BitcastOp>,
    CmpIOpConversion,
    CmpFOpConversion,
    PlainCallConversion
  >(typeConverter, patterns.getContext());
  // clang-format on
}
