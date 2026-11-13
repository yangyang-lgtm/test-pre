#ifndef PRIVATEUSE_CONVERSION_ARITHTOLINALG_CONVERSIONPATTERNS_HPP
#define PRIVATEUSE_CONVERSION_ARITHTOLINALG_CONVERSIONPATTERNS_HPP

#include "privateuse/Conversion/ArithToLinalg/ConversionTools.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/ReshapeOpsUtils.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/IRMapping.h"
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

struct ReduceConverter : public OpConversionPattern<triton::ReduceOp>
{
  using OpConversionPattern<triton::ReduceOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(triton::ReduceOp op, OpAdaptor adaptor, ConversionPatternRewriter& rewriter) const override
  {
    auto sourceType = dyn_cast<RankedTensorType>(adaptor.getSrcs().front().getType());
    if (!sourceType) return rewriter.notifyMatchFailure(op, "expected ranked tensor operand");

    int64_t axis = op.getAxis();
    if (axis < 0 || axis >= sourceType.getRank()) return rewriter.notifyMatchFailure(op, "reduction axis out of range");

    auto loc          = op.getLoc();
    auto rank         = sourceType.getRank();
    bool isFullReduce = (rank == 1);

    SmallVector<Value> inputs(adaptor.getSrcs().begin(), adaptor.getSrcs().end());

    SmallVector<Value> initTensors;
    initTensors.reserve(inputs.size());
    for (auto [input, result] : llvm::zip_equal(inputs, op->getResults())) {
      auto elemType = cast<RankedTensorType>(input.getType()).getElementType();
      auto shape    = isFullReduce ? llvm::ArrayRef<int64_t>{} : cast<RankedTensorType>(result.getType()).getShape();
      initTensors.push_back(tensor::EmptyOp::create(rewriter, loc, shape, elemType));
    }

    auto* context = rewriter.getContext();
    SmallVector<AffineMap> maps;
    for (size_t i = 0, e = inputs.size(); i < e; ++i) maps.push_back(AffineMap::getMultiDimIdentityMap(rank, context));

    SmallVector<AffineExpr> keptDims;
    for (int64_t d = 0; d < rank; ++d)
      if (d != axis) keptDims.push_back(getAffineDimExpr(d, context));
    auto initMap = AffineMap::get(rank, 0, keptDims, context);
    for (size_t i = 0, e = initTensors.size(); i < e; ++i) maps.push_back(initMap);

    SmallVector<utils::IteratorType> iterators(rank, utils::IteratorType::parallel);
    iterators[axis] = utils::IteratorType::reduction;

    SmallVector<Type> resultTypes = llvm::map_to_vector(initTensors, [](Value value) { return value.getType(); });

    Block& combineBlock = op.getCombineOp().front();
    auto generic        = linalg::GenericOp::create(
      rewriter, loc, resultTypes, inputs, initTensors, maps, iterators,
      [&](OpBuilder& nestedBuilder, Location nestedLoc, ValueRange blockArgs) {
        IRMapping mapping;
        for (auto [origArg, blockArg] : llvm::zip_equal(combineBlock.getArguments(), blockArgs))
          mapping.map(origArg, blockArg);
        for (Operation& redOp : combineBlock.without_terminator()) nestedBuilder.clone(redOp, mapping);
        auto results = llvm::map_to_vector(combineBlock.getTerminator()->getOperands(), [&](Value value) {
          return mapping.lookup(value);
        });
        linalg::YieldOp::create(nestedBuilder, nestedLoc, results);
      }
    );

    SmallVector<Value> finalResults;
    finalResults.reserve(generic.getNumResults());
    for (auto [index, result] : llvm::enumerate(generic.getResults())) {
      if (isFullReduce) {
        auto elemType = cast<RankedTensorType>(inputs[index].getType()).getElementType();
        finalResults.push_back(tensor::ExtractOp::create(rewriter, loc, elemType, result, ValueRange{}));
      } else {
        finalResults.push_back(result);
      }
    }

    rewriter.replaceOp(op, finalResults);
    return success();
  }
};

struct DotConverter : public OpConversionPattern<triton::DotOp>
{
  using OpConversionPattern<triton::DotOp>::OpConversionPattern;

  static bool isZeroTensor(Value v, bool integers)
  {
    if (auto splatOp = v.getDefiningOp<triton::SplatOp>()) {
      if (auto constOp = splatOp.getSrc().getDefiningOp<arith::ConstantOp>()) {
        if (auto val = dyn_cast<FloatAttr>(constOp.getValue())) return val.getValueAsDouble() == 0.;
        if (auto val = dyn_cast<IntegerAttr>(constOp.getValue())) return val.getValue().isZero();
      }
      return false;
    }
    if (auto constOp = v.getDefiningOp<arith::ConstantOp>()) {
      if (auto dense = dyn_cast<DenseElementsAttr>(constOp.getValue()); dense && dense.isSplat())
        return integers ? dense.getSplatValue<APInt>().isZero() : dense.getSplatValue<APFloat>().isZero();
    }
    return false;
  }

  LogicalResult matchAndRewrite(triton::DotOp op, OpAdaptor adaptor, ConversionPatternRewriter& rewriter) const override
  {
    auto loc      = op.getLoc();
    auto dstType  = cast<RankedTensorType>(op.getType());
    auto elemType = dstType.getElementType();
    bool integers = elemType.isInteger();
    bool skipC    = isZeroTensor(op.getC(), integers);

    auto rank = dstType.getRank();
    if (rank > 3) return rewriter.notifyMatchFailure(op, "dot with rank > 3 is not supported");

    auto init = tensor::EmptyOp::create(rewriter, loc, dstType.getShape(), elemType);
    Value res;
    if (rank == 3)
      res = linalg::BatchMatmulOp::create(rewriter, loc, ValueRange{adaptor.getA(), adaptor.getB()}, ValueRange{init})
              .getResult(0);
    else
      res = linalg::MatmulOp::create(rewriter, loc, ValueRange{adaptor.getA(), adaptor.getB()}, ValueRange{init})
              .getResult(0);

    if (!skipC) {
      if (integers)
        res = arith::AddIOp::create(rewriter, loc, adaptor.getC(), res);
      else
        res = arith::AddFOp::create(rewriter, loc, adaptor.getC(), res);
    }

    rewriter.replaceOp(op, res);
    return success();
  }
};

struct TransposeConverter : public OpConversionPattern<triton::TransOp>
{
  using OpConversionPattern<triton::TransOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(triton::TransOp op, OpAdaptor adaptor, ConversionPatternRewriter& rewriter) const override
  {
    auto source     = adaptor.getSrc();
    auto sourceType = cast<RankedTensorType>(source.getType());
    auto resultType = cast<RankedTensorType>(op.getType());
    auto order      = op.getOrder();
    if (order.size() != sourceType.getRank())
      return rewriter.notifyMatchFailure(op, "transpose order size must match source rank");

    SmallVector<int64_t> perm(order.begin(), order.end());
    auto init      = tensor::EmptyOp::create(rewriter, op.getLoc(), resultType.getShape(), resultType.getElementType());
    auto transpose = linalg::TransposeOp::create(rewriter, op.getLoc(), source, init, perm).getResults()[0];
    rewriter.replaceOp(op, transpose);
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
    ReshapeConverter,
    ReduceConverter,
    DotConverter,
    TransposeConverter
  >(patterns.getContext());
  // clang-format on

  linalg::populateElementwiseToLinalgConversionPatterns(patterns);
}

} // namespace triton
} // namespace mlir

#endif
