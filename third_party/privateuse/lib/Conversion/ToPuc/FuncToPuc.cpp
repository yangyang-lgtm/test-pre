#include "privateuse/Conversion/ToPuc/PucPatterns.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Transforms/DialectConversion.h"
#include "privateuse/Dialect/Puc/IR/PucDialect.h"

using namespace mlir;

namespace {

struct FuncOpConversion : OpConversionPattern<func::FuncOp>
{
  using OpConversionPattern<func::FuncOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(func::FuncOp op, func::FuncOp::Adaptor adaptor, ConversionPatternRewriter& rewriter) const override
  {
    SmallVector<NamedAttribute> attrs;
    for (NamedAttribute attr : op->getAttrs()) {
      if (attr.getName() == op.getSymNameAttrName() || attr.getName() == op.getFunctionTypeAttrName()) continue;
      attrs.push_back(attr);
    }

    auto newFunc = puc::FuncOp::create(rewriter, op.getLoc(), op.getName(), op.getFunctionType(), attrs, {});
    rewriter.inlineRegionBefore(op.getBody(), newFunc.getBody(), newFunc.getBody().end());
    rewriter.eraseOp(op);
    return success();
  }
};

struct ReturnOpConversion : OpConversionPattern<func::ReturnOp>
{
  using OpConversionPattern<func::ReturnOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(func::ReturnOp op, func::ReturnOp::Adaptor adaptor, ConversionPatternRewriter& rewriter)
    const override
  {
    rewriter.replaceOpWithNewOp<puc::ReturnOp>(op, adaptor.getOperands());
    return success();
  }
};

struct CallOpConversion : OpConversionPattern<func::CallOp>
{
  using OpConversionPattern<func::CallOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(func::CallOp op, func::CallOp::Adaptor adaptor, ConversionPatternRewriter& rewriter) const override
  {
    rewriter.replaceOpWithNewOp<puc::CallOp>(
      op, op->getResultTypes(), FlatSymbolRefAttr::get(rewriter.getContext(), op.getCallee()), ArrayAttr{},
      adaptor.getOperands()
    );
    return success();
  }
};

} // namespace

void mlir::triton::populateFuncToPucPatterns(RewritePatternSet& patterns, TypeConverter& typeConverter)
{
  // clang-format off
  patterns.add<
      FuncOpConversion,
      ReturnOpConversion,
      CallOpConversion
  >(typeConverter, patterns.getContext());
  // clang-format on
}
