#ifndef PRIVATEUSE_CONVERSION_ARITHTOLINALG_CONVERSIONPATTERNS_HPP
#define PRIVATEUSE_CONVERSION_ARITHTOLINALG_CONVERSIONPATTERNS_HPP

#include "privateuse/Conversion/ArithToLinalg/ConversionTools.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/ReshapeOpsUtils.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Transforms/DialectConversion.h"
#include "triton/Dialect/Triton/IR/Dialect.h"

namespace mlir {
namespace triton {

struct SplatConverter : public OpConversionPattern<triton::SplatOp>
{
  using OpConversionPattern<triton::SplatOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(triton::SplatOp op, OpAdaptor adaptor, ConversionPatternRewriter& rewriter) const override
  {
    auto opType = cast<RankedTensorType>(op.getType());
    auto loc    = op.getLoc();
    auto init   = tensor::EmptyOp::create(rewriter, loc, opType.getShape(), opType.getElementType());
    rewriter.replaceOpWithNewOp<linalg::FillOp>(op, ValueRange{adaptor.getSrc()}, ValueRange{init});
    return success();
  }
};

struct DenseConstantConverter : public OpConversionPattern<arith::ConstantOp>
{
  using OpConversionPattern<arith::ConstantOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(arith::ConstantOp op, OpAdaptor adaptor, ConversionPatternRewriter& rewriter) const override
  {
    auto attr = dyn_cast<DenseElementsAttr>(op.getValue());
    if (!attr || !attr.isSplat()) return rewriter.notifyMatchFailure(op, "not a dense splat constant");
    auto loc = op.getLoc();
    auto splatConst =
      arith::ConstantOp::materialize(rewriter, attr.getSplatValue<Attribute>(), attr.getElementType(), loc);
    auto resultType = cast<RankedTensorType>(op.getResult().getType());
    auto init       = tensor::EmptyOp::create(rewriter, loc, resultType.getShape(), attr.getElementType());
    rewriter.replaceOpWithNewOp<linalg::FillOp>(op, ValueRange{splatConst}, ValueRange{init});
    return success();
  }
};

struct BitcastConverter : public OpConversionPattern<triton::BitcastOp>
{
  using OpConversionPattern<triton::BitcastOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(triton::BitcastOp op, OpAdaptor adaptor, ConversionPatternRewriter& rewriter) const override
  {
    auto result = arith::BitcastOp::create(rewriter, op.getLoc(), op.getType(), adaptor.getSrc());
    rewriter.replaceOp(op, result.getResult());
    return success();
  }
};

struct MakeRangeConverter : public OpConversionPattern<triton::MakeRangeOp>
{
  using OpConversionPattern<triton::MakeRangeOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(triton::MakeRangeOp op, OpAdaptor adaptor, ConversionPatternRewriter& rewriter) const override
  {
    auto loc         = op.getLoc();
    auto type        = cast<RankedTensorType>(op.getResult().getType());
    auto shape       = type.getShape();
    auto elementType = type.getElementType();
    auto context     = rewriter.getContext();

    if (shape.size() != 1 || elementType.getIntOrFloatBitWidth() != 32)
      return rewriter.notifyMatchFailure(op, "make_range must be a 1-D i32 tensor");

    SmallVector<AffineMap> indexingMaps{
      AffineMap::get(1, 0, SmallVector<AffineExpr>{getAffineDimExpr(0, context)}, context)
    };

    auto init     = tensor::EmptyOp::create(rewriter, loc, shape, elementType);
    auto linalgOp = linalg::GenericOp::create(
      rewriter, loc, op->getResultTypes(), ValueRange{}, ValueRange{init}, indexingMaps, getNParallelLoopsAttrs(1),
      [&](OpBuilder& nestedBuilder, Location nestedLoc, ValueRange blockArgs) {
        Value index = linalg::IndexOp::create(nestedBuilder, loc, 0);
        Value res   = arith::IndexCastOp::create(nestedBuilder, loc, elementType, index);
        if (op.getStart()) {
          auto start = arith::ConstantIntOp::create(rewriter, loc, op.getStart(), elementType.getIntOrFloatBitWidth());
          res        = arith::AddIOp::create(nestedBuilder, loc, res, start);
        }
        linalg::YieldOp::create(nestedBuilder, loc, res);
      }
    );

    rewriter.replaceOp(op, linalgOp->getResults());
    return success();
  }
};

struct BroadcastConverter : public OpConversionPattern<triton::BroadcastOp>
{
  using OpConversionPattern<triton::BroadcastOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(triton::BroadcastOp op, OpAdaptor adaptor, ConversionPatternRewriter& rewriter) const override
  {
    auto loc          = op.getLoc();
    auto sourceType   = cast<RankedTensorType>(adaptor.getSrc().getType());
    auto resultType   = cast<RankedTensorType>(op.getType());
    auto elementType  = resultType.getElementType();
    size_t resultRank = resultType.getRank();

    SmallVector<AffineMap> indexingMaps;
    indexingMaps.push_back(getBroadcastAffineMap(op->getContext(), sourceType.getShape(), resultType.getShape()));
    indexingMaps.append(op->getNumResults(), rewriter.getMultiDimIdentityMap(resultRank));

    auto init     = tensor::EmptyOp::create(rewriter, loc, resultType.getShape(), elementType);
    auto linalgOp = linalg::GenericOp::create(
      rewriter, loc, op->getResultTypes(), ValueRange{adaptor.getSrc()}, ValueRange{init}, indexingMaps,
      getNParallelLoopsAttrs(resultRank),
      [&](OpBuilder& nestedBuilder, Location nestedLoc, ValueRange blockArgs) {
        linalg::YieldOp::create(nestedBuilder, loc, blockArgs[0]);
      }
    );

    rewriter.replaceOp(op, linalgOp->getResults());
    return success();
  }
};

struct ExpandDimsConverter : public OpConversionPattern<triton::ExpandDimsOp>
{
  using OpConversionPattern<triton::ExpandDimsOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(triton::ExpandDimsOp op, OpAdaptor adaptor, ConversionPatternRewriter& rewriter) const override
  {
    auto src     = adaptor.getSrc();
    auto srcType = cast<RankedTensorType>(src.getType());
    auto srcRank = srcType.getRank();
    auto resType = cast<RankedTensorType>(op->getResultTypes()[0]);

    SmallVector<ReassociationIndices> reassoc;
    int64_t c = 0;
    for (int64_t i = 0; i < srcRank; i++) {
      ReassociationIndices g;
      g.push_back(c++);
      if (op.getAxis() == i) {
        g.push_back(c++);
      } else if (op.getAxis() == i + 1 && i == srcRank - 1) {
        g.push_back(c++);
      }
      reassoc.push_back(g);
    }

    auto expandShapeOp = tensor::ExpandShapeOp::create(rewriter, op.getLoc(), resType, src, reassoc);
    rewriter.replaceOp(op, expandShapeOp.getResult());
    return success();
  }
};

struct ReshapeConverter : public OpConversionPattern<triton::ReshapeOp>
{
  using OpConversionPattern<triton::ReshapeOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(triton::ReshapeOp op, OpAdaptor adaptor, ConversionPatternRewriter& rewriter) const override
  {
    auto loc        = op.getLoc();
    auto input      = adaptor.getSrc();
    auto inputType  = cast<RankedTensorType>(input.getType());
    auto outputType = cast<RankedTensorType>(op.getResult().getType());
    if (!outputType.hasStaticShape()) return rewriter.notifyMatchFailure(op, "dynamic reshape shape is preserved");

    if (auto maybeReassociationMap = getReassociationIndicesForReshape(inputType, outputType)) {
      auto reassociationMap = *maybeReassociationMap;
      if (outputType.getRank() < inputType.getRank()) {
        rewriter.replaceOpWithNewOp<tensor::CollapseShapeOp>(op, outputType, input, reassociationMap);
      } else {
        rewriter.replaceOpWithNewOp<tensor::ExpandShapeOp>(op, outputType, input, reassociationMap);
      }
      return success();
    }

    auto shape = arith::ConstantOp::create(rewriter, loc, rewriter.getI64TensorAttr(outputType.getShape()));
    rewriter.replaceOpWithNewOp<tensor::ReshapeOp>(op, outputType, input, shape);
    return success();
  }
};

static void populateArithToLinalgConversionPatterns(RewritePatternSet& patterns)
{
  // clang-format off
  patterns.add<
    SplatConverter,
    DenseConstantConverter,
    BitcastConverter,
    MakeRangeConverter,
    BroadcastConverter,
    ExpandDimsConverter,
    ReshapeConverter
  >(patterns.getContext());
  // clang-format on

  linalg::populateElementwiseToLinalgConversionPatterns(patterns);
}

} // namespace triton
} // namespace mlir

#endif
