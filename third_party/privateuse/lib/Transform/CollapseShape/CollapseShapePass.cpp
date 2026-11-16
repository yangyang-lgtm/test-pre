#include "privateuse/Conversion/ArithToLinalg/ConversionTools.h"
#include "privateuse/Transform/CollapseShape/Passes.h"

#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/ReshapeOpsUtils.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/Support/ErrorHandling.h"

#define GEN_PASS_DEF_COLLAPSESHAPE
#include "privateuse/Transform/CollapseShape/Passes.h.inc"

using namespace mlir;
using namespace mlir::triton;

namespace {

struct CollapseFill : public OpRewritePattern<linalg::FillOp>
{
  CollapseFill(MLIRContext* context) : OpRewritePattern<linalg::FillOp>(context) {}

  LogicalResult collapseMemRef(linalg::FillOp op, PatternRewriter& rewriter) const
  {
    auto loc        = op.getLoc();
    auto result     = op.getOutputs()[0];
    auto resultType = dyn_cast<MemRefType>(result.getType());
    if (!resultType) return failure();
    auto rank = resultType.getRank();
    if (rank <= 1 || dyn_cast<StridedLayoutAttr>(resultType.getLayout())) {
      return failure();
    }
    ReassociationExprs reassociationMap;
    for (unsigned i = 0; i < rank; ++i) {
      reassociationMap.push_back(rewriter.getAffineDimExpr(i));
    }
    auto elementType = resultType.getElementType();

    auto collapsedType = MemRefType::get(
      ArrayRef<int64_t>{resultType.getNumElements()}, elementType, AffineMap(), resultType.getMemorySpace()
    );
    auto output = memref::CollapseShapeOp::create(rewriter, loc, collapsedType, result, reassociationMap);
    op.getOutputsMutable()[0].set(output.getResult());
    return success();
  }

  LogicalResult collapseTensor(linalg::FillOp op, PatternRewriter& rewriter) const
  {
    auto loc        = op.getLoc();
    auto result     = op.getResult(0);
    auto resultType = dyn_cast<RankedTensorType>(result.getType());
    if (!resultType) return failure();
    auto rank = resultType.getRank();
    if (rank <= 1) {
      return failure();
    }
    auto elementType = resultType.getElementType();

    ReassociationExprs reassociationMap;
    for (unsigned i = 0; i < rank; ++i) {
      reassociationMap.push_back(rewriter.getAffineDimExpr(i));
    }

    auto init = tensor::CollapseShapeOp::create(
      rewriter, loc, RankedTensorType::get({resultType.getNumElements()}, elementType), op.getOutputs()[0],
      reassociationMap
    );
    auto fillOp = linalg::FillOp::create(rewriter, loc, op.getInputs(), ValueRange{init});

    auto expandOp =
      tensor::ExpandShapeOp::create(rewriter, loc, result.getType(), fillOp.getResult(0), reassociationMap);

    rewriter.replaceOp(op, expandOp.getResult());
    return success();
  }

  LogicalResult matchAndRewrite(linalg::FillOp op, PatternRewriter& rewriter) const override
  {
    if (!(op->getNumResults() <= 1)) llvm::report_fatal_error("code assumes single result!");
    if (op->getNumResults() == 1) {
      return collapseTensor(op, rewriter);
    } else if (op->getNumResults() == 0) {
      return collapseMemRef(op, rewriter);
    }
    return failure();
  }
};

struct CollapseTranspose : public OpRewritePattern<linalg::TransposeOp>
{
  CollapseTranspose(MLIRContext* context) : OpRewritePattern<linalg::TransposeOp>(context) {}
  LogicalResult matchAndRewrite(linalg::TransposeOp op, PatternRewriter& rewriter) const override
  {
    Value source    = op.getInput();
    auto sourceType = dyn_cast<RankedTensorType>(source.getType());
    if (!sourceType) {
      return rewriter.notifyMatchFailure(op, "expected ranked tensor type for source");
    }
    auto sourceRank  = sourceType.getRank();
    auto elementType = sourceType.getElementType();
    if (sourceRank <= 3) {
      return rewriter.notifyMatchFailure(op, "expected source rank > 3 for transpose collapse");
    }

    SmallVector<int64_t> perm(op.getPermutation());
    SmallVector<int64_t> transposedShape(sourceRank);
    SmallVector<ReassociationExprs> reassociationMap;
    SmallVector<int64_t> collapseShapeInput;
    int dim = 0;
    SmallVector<int64_t> permIdx(sourceRank);
    for (size_t i = 0; i < sourceRank; ++i) {
      permIdx[perm[i]] = i;
    }
    SmallVector<int64_t> mapDim;
    for (size_t i = 0; i < sourceRank; ++i) {
      auto id = permIdx[i];
      if (i > 0 && (id == 0 || !(perm[id] == perm[id - 1] + 1))) {
        dim++;
      }
      if (dim == collapseShapeInput.size()) {
        collapseShapeInput.push_back(1);
        reassociationMap.push_back({});
      }
      reassociationMap[dim].push_back(rewriter.getAffineDimExpr(i));
      mapDim.push_back(dim);
      collapseShapeInput[dim] *= sourceType.getDimSize(i);
    }
    if (collapseShapeInput.size() == sourceRank) {
      return rewriter.notifyMatchFailure(op, "cannot collapse broadcast shape");
    }

    SmallVector<int64_t> newPerm;
    for (size_t i = 0; i < sourceRank; ++i) {
      if (i > 0 && newPerm.back() == mapDim[perm[i]]) {
        continue;
      }
      newPerm.push_back(mapDim[perm[i]]);
    }
    perm = newPerm;
    transposedShape.clear();
    for (size_t i = 0; i < perm.size(); ++i) {
      transposedShape.push_back(collapseShapeInput[perm[i]]);
    }

    auto loc   = op.getLoc();
    sourceType = RankedTensorType::get(collapseShapeInput, elementType);
    source     = tensor::CollapseShapeOp::create(rewriter, loc, sourceType, source, reassociationMap);

    SmallVector<ReassociationExprs> reassociationMapRe(reassociationMap.size());
    int idx = 0;
    for (size_t i = 0; i < reassociationMap.size(); ++i) {
      for (size_t j = 0; j < reassociationMap[perm[i]].size(); ++j) {
        reassociationMapRe[i].push_back(rewriter.getAffineDimExpr(idx++));
      }
    }

    Value transposeInit = tensor::CollapseShapeOp::create(
      rewriter, loc, RankedTensorType::get(transposedShape, elementType), op.getInit(), reassociationMapRe
    );

    Value transpose = linalg::TransposeOp::create(rewriter, loc, source, transposeInit, perm).getResults()[0];

    rewriter.replaceOpWithNewOp<tensor::ExpandShapeOp>(op, op.getResultTypes()[0], transpose, reassociationMapRe);
    return success();
  }
};

struct CollapseBroadCast : public OpRewritePattern<linalg::GenericOp>
{
  CollapseBroadCast(MLIRContext* context) : OpRewritePattern<linalg::GenericOp>(context) {}
  LogicalResult matchAndRewrite(linalg::GenericOp op, PatternRewriter& rewriter) const override
  {
    if (!op->hasAttr("broadcastDims")) {
      return rewriter.notifyMatchFailure(op, "expected broadcastDims attribute");
    }
    if (!(op->getNumResults() == 1)) llvm::report_fatal_error("code assumes single result!");
    auto input      = op.getInputs()[0];
    auto sourceType = dyn_cast<RankedTensorType>(input.getType());
    if (!sourceType) {
      return rewriter.notifyMatchFailure(op, "expected ranked tensor type for source");
    }
    auto sourceRank = sourceType.getRank();
    if (sourceRank <= 1) {
      return rewriter.notifyMatchFailure(op, "expected source rank > 1 for broadcast collapse");
    }

    auto resultType = dyn_cast<RankedTensorType>(op.getResultTypes()[0]);
    if (!resultType) return rewriter.notifyMatchFailure(op, "expected ranked tensor type for result");
    auto elementType = resultType.getElementType();
    SmallVector<int64_t, 8> collapseShapeInput;
    SmallVector<int64_t, 8> collapseShapeOutput;
    SmallVector<ReassociationExprs, 8> reassociationMap;
    int dim = 0;
    for (size_t i = 0; i < sourceRank; i++) {
      if (i > 0 && !((sourceType.getDimSize(i) == 1 && sourceType.getDimSize(i - 1) == 1) ||
                     (sourceType.getDimSize(i) != 1 && sourceType.getDimSize(i - 1) != 1))) {
        dim++;
      }
      if (dim == collapseShapeInput.size()) {
        collapseShapeInput.push_back(1);
        collapseShapeOutput.push_back(1);
        reassociationMap.push_back({});
      }
      reassociationMap[dim].push_back(rewriter.getAffineDimExpr(i));
      collapseShapeInput[dim]  *= sourceType.getDimSize(i);
      collapseShapeOutput[dim] *= resultType.getDimSize(i);
    }
    if (collapseShapeInput.size() == sourceRank) {
      return rewriter.notifyMatchFailure(op, "cannot collapse broadcast shape");
    }

    auto loc          = op.getLoc();
    sourceType        = RankedTensorType::get(collapseShapeInput, elementType);
    input             = tensor::CollapseShapeOp::create(rewriter, loc, sourceType, input, reassociationMap);
    resultType        = RankedTensorType::get(collapseShapeOutput, elementType);
    size_t resultRank = resultType.getRank();

    SmallVector<AffineMap> indexingMaps;
    indexingMaps.reserve(op->getNumOperands() + op->getNumResults());
    indexingMaps.push_back(getBroadcastAffineMap(op->getContext(), sourceType.getShape(), resultType.getShape()));
    indexingMaps.append(op->getNumResults(), rewriter.getMultiDimIdentityMap(resultRank));

    if (!(op->getNumResults() == 1)) llvm::report_fatal_error("code assumes single result!");

    auto init = tensor::CollapseShapeOp::create(
      rewriter, loc, RankedTensorType::get(resultType.getShape(), elementType), op.getOutputs()[0], reassociationMap
    );

    auto linalgOp = linalg::GenericOp::create(
      rewriter, loc, init->getResultTypes(), ValueRange{input}, ValueRange{init}, indexingMaps,
      getNParallelLoopsAttrs(resultRank)
    );
    rewriter.cloneRegionBefore(op.getRegion(), linalgOp.getRegion(), linalgOp.getRegion().begin());
    linalgOp->setAttr("broadcastDims", rewriter.getDenseI64ArrayAttr(getBroadcastDims(sourceType, resultType)));
    rewriter.replaceOpWithNewOp<tensor::ExpandShapeOp>(
      op, op.getResultTypes()[0], linalgOp->getResult(0), reassociationMap
    );
    return success();
  }
};

struct CollapseReduce : public OpRewritePattern<linalg::GenericOp>
{
  CollapseReduce(MLIRContext* context) : OpRewritePattern<linalg::GenericOp>(context) {}

  LogicalResult matchAndRewrite(linalg::GenericOp op, PatternRewriter& rewriter) const override
  {
    if (op.getNumDpsInputs() != 1 || op.getNumDpsInits() != 1)
      return rewriter.notifyMatchFailure(op, "expected a single input and a single init");

    auto input     = op.getDpsInputs()[0];
    auto inputType = dyn_cast<RankedTensorType>(input.getType());
    if (!inputType) {
      return rewriter.notifyMatchFailure(op, "expected ranked tensor type for input");
    }

    int64_t inputRank = inputType.getRank();
    if (op.getNumLoops() != inputRank) return rewriter.notifyMatchFailure(op, "expected an identity input map");

    SmallVector<int64_t> dims;
    SmallVector<utils::IteratorType> iteratorTypes = op.getIteratorTypesArray();
    for (auto [index, type] : llvm::enumerate(iteratorTypes))
      if (type == utils::IteratorType::reduction) dims.push_back(index);
    if (dims.empty()) return rewriter.notifyMatchFailure(op, "expected a reduction iterator");

    auto maps = op.getIndexingMapsArray();
    if (!maps[0].isIdentity()) return rewriter.notifyMatchFailure(op, "expected an identity input map");

    SmallVector<AffineExpr> keptDims;
    for (int64_t d = 0; d < inputRank; ++d)
      if (!llvm::is_contained(dims, d)) keptDims.push_back(rewriter.getAffineDimExpr(d));
    if (maps[1] != AffineMap::get(inputRank, 0, keptDims, rewriter.getContext()))
      return rewriter.notifyMatchFailure(op, "expected a reduction output map");

    if (inputRank - dims.size() <= 1) {
      return rewriter.notifyMatchFailure(op, "expected input rank - reduction loops > 1 for reduce collapse");
    }
    SmallVector<int64_t> collapseShapeInput;
    SmallVector<int64_t> newDims;
    SmallVector<ReassociationExprs> reassociationMap;
    int dim = 0;
    for (size_t i = 0; i < inputRank; i++) {
      bool reduceAxis = llvm::is_contained(dims, i);
      if (i > 0 && (reduceAxis || llvm::is_contained(dims, i - 1))) {
        dim++;
      }
      if (reduceAxis) {
        newDims.push_back(dim);
      }
      if (dim == collapseShapeInput.size()) {
        collapseShapeInput.push_back(1);
        reassociationMap.push_back({});
      }
      reassociationMap[dim].push_back(rewriter.getAffineDimExpr(i));
      collapseShapeInput[dim] *= inputType.getDimSize(i);
    }
    if (collapseShapeInput.size() == inputRank) {
      return rewriter.notifyMatchFailure(op, "cannot collapse reduce shape");
    }
    SmallVector<int64_t> collapseShapeOutput;
    for (size_t i = 0; i < collapseShapeInput.size(); i++) {
      if (llvm::is_contained(newDims, i)) {
        continue;
      }
      collapseShapeOutput.push_back(collapseShapeInput[i]);
    }
    auto elementType  = inputType.getElementType();
    auto loc          = op.getLoc();
    auto newInputType = RankedTensorType::get(collapseShapeInput, elementType);
    input             = tensor::CollapseShapeOp::create(rewriter, loc, newInputType, input, reassociationMap);

    SmallVector<ReassociationExprs> reassociationMapOutput;
    int idx = 0;
    for (size_t i = 0; i < reassociationMap.size(); ++i) {
      if (llvm::is_contained(newDims, i)) {
        continue;
      } else {
        reassociationMapOutput.push_back({});
      }
      for (size_t j = 0; j < reassociationMap[i].size(); ++j) {
        reassociationMapOutput.back().push_back(rewriter.getAffineDimExpr(idx++));
      }
    }
    auto init = tensor::CollapseShapeOp::create(
      rewriter, loc, RankedTensorType::get(collapseShapeOutput, elementType), op.getDpsInits()[0],
      reassociationMapOutput
    );

    int64_t newRank = collapseShapeInput.size();
    SmallVector<AffineMap> newMaps;
    newMaps.push_back(AffineMap::getMultiDimIdentityMap(newRank, rewriter.getContext()));
    SmallVector<AffineExpr> newKeptDims;
    for (int64_t d = 0; d < newRank; ++d)
      if (!llvm::is_contained(newDims, d)) newKeptDims.push_back(rewriter.getAffineDimExpr(d));
    newMaps.push_back(AffineMap::get(newRank, 0, newKeptDims, rewriter.getContext()));

    SmallVector<utils::IteratorType> newIterators(newRank, utils::IteratorType::parallel);
    for (int64_t d : newDims) newIterators[d] = utils::IteratorType::reduction;

    Block* sourceBody = op.getBlock();
    auto newGeneric   = linalg::GenericOp::create(
      rewriter, loc, TypeRange{init.getType()}, ValueRange{input}, ValueRange{init}, newMaps, newIterators,
      [&](OpBuilder& b, Location bodyLoc, ValueRange args) {
        IRMapping mapping;
        for (auto [arg, mapped] : llvm::zip_equal(sourceBody->getArguments(), args)) mapping.map(arg, mapped);
        for (Operation& inner : sourceBody->without_terminator()) b.clone(inner, mapping);
        SmallVector<Value> yields = llvm::map_to_vector(sourceBody->getTerminator()->getOperands(), [&](Value value) {
          return mapping.lookup(value);
        });
        linalg::YieldOp::create(b, bodyLoc, yields);
      }
    );

    rewriter.replaceOpWithNewOp<tensor::ExpandShapeOp>(
      op, op.getResultTypes()[0], newGeneric->getResult(0), reassociationMapOutput
    );
    return success();
  }
};

struct CollapseElementwise : public OpRewritePattern<linalg::GenericOp>
{
  CollapseElementwise(MLIRContext* context) : OpRewritePattern<linalg::GenericOp>(context) {}

  LogicalResult matchAndRewrite(linalg::GenericOp op, PatternRewriter& rewriter) const override
  {
    if (op.getNumDpsInputs() == 0 || op.getNumDpsInits() == 0 || op.getNumResults() != op.getNumDpsInits())
      return rewriter.notifyMatchFailure(op, "expected a tensor-semantics generic");
    int64_t rank = op.getNumLoops();
    if (rank <= 1) return rewriter.notifyMatchFailure(op, "expected rank > 1 for elementwise collapse");
    for (utils::IteratorType iteratorType : op.getIteratorTypesArray())
      if (iteratorType != utils::IteratorType::parallel)
        return rewriter.notifyMatchFailure(op, "expected all-parallel loops");
    for (AffineMap map : op.getIndexingMapsArray())
      if (!map.isIdentity()) return rewriter.notifyMatchFailure(op, "expected identity maps");

    auto collapsedType = [&](Value value) -> RankedTensorType {
      auto type = dyn_cast<RankedTensorType>(value.getType());
      if (!type || type.getRank() != rank || !type.hasStaticShape()) return nullptr;
      return RankedTensorType::get({type.getNumElements()}, type.getElementType());
    };

    ReassociationExprs reassociationMap;
    for (int64_t i = 0; i < rank; ++i) reassociationMap.push_back(rewriter.getAffineDimExpr(i));

    auto loc = op.getLoc();
    SmallVector<Value> inputs;
    for (Value value : op.getDpsInputs()) {
      auto type = collapsedType(value);
      if (!type) return rewriter.notifyMatchFailure(op, "expected static ranked tensor inputs");
      inputs.push_back(tensor::CollapseShapeOp::create(rewriter, loc, type, value, reassociationMap));
    }
    SmallVector<Value> inits;
    for (Value value : op.getDpsInits()) {
      auto type = collapsedType(value);
      if (!type) return rewriter.notifyMatchFailure(op, "expected a static ranked tensor init");
      inits.push_back(tensor::CollapseShapeOp::create(rewriter, loc, type, value, reassociationMap));
    }

    SmallVector<Type> resultTypes;
    for (Value init : inits) resultTypes.push_back(init.getType());
    SmallVector<AffineMap> indexingMaps(op.getNumDpsInputs() + op.getNumDpsInits(),
                                        AffineMap::getMultiDimIdentityMap(1, rewriter.getContext()));
    SmallVector<utils::IteratorType> iteratorTypes(1, utils::IteratorType::parallel);

    Block* sourceBody = op.getBlock();
    auto newGeneric   = linalg::GenericOp::create(
      rewriter, loc, resultTypes, inputs, inits, indexingMaps, iteratorTypes,
      [&](OpBuilder& b, Location bodyLoc, ValueRange args) {
        IRMapping mapping;
        for (auto [arg, mapped] : llvm::zip_equal(sourceBody->getArguments(), args)) mapping.map(arg, mapped);
        for (Operation& inner : sourceBody->without_terminator()) b.clone(inner, mapping);
        SmallVector<Value> yields = llvm::map_to_vector(sourceBody->getTerminator()->getOperands(), [&](Value value) {
          return mapping.lookup(value);
        });
        linalg::YieldOp::create(b, bodyLoc, yields);
      }
    );

    SmallVector<Value> expanded;
    for (auto [result, original] : llvm::zip_equal(newGeneric.getResults(), op.getResults()))
      expanded.push_back(tensor::ExpandShapeOp::create(rewriter, loc, original.getType(), result, reassociationMap));
    rewriter.replaceOp(op, expanded);
    return success();
  }
};

class CollapseShapePass : public ::impl::CollapseShapeBase<CollapseShapePass>
{
public:
  void getDependentDialects(DialectRegistry& registry) const override
  {
    registry.insert<linalg::LinalgDialect, tensor::TensorDialect, memref::MemRefDialect>();
  }

  void runOnOperation() override
  {
    auto moduleOp = getOperation();
    RewritePatternSet patterns(&getContext());

    // clang-format off
    patterns.add<
      CollapseFill,
      CollapseBroadCast,
      CollapseTranspose,
      CollapseReduce,
      CollapseElementwise
    >(&getContext());
    // clang-format on

    if (failed(applyPatternsGreedily(moduleOp, std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

} // namespace

std::unique_ptr<OperationPass<ModuleOp>> triton::createCollapseShapePass()
{
  return std::make_unique<CollapseShapePass>();
}
