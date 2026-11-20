#include "privateuse/Transform/LinalgToRuntime/Passes.h"

#include "llvm/ADT/SmallVector.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "privateuse/Dialect/PUMem/IR/PUMemDialect.h"

#define GEN_PASS_DEF_LINALGTORUNTIME
#include "privateuse/Transform/LinalgToRuntime/Passes.h.inc"

using namespace mlir;
using namespace mlir::triton;

namespace {

class MatmulToRuntime : public OpRewritePattern<linalg::MatmulOp>
{
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::MatmulOp op, PatternRewriter& rewriter) const override
  {
    auto a = dyn_cast<MemRefType>(op.getInputs()[0].getType());
    auto b = dyn_cast<MemRefType>(op.getInputs()[1].getType());
    auto c = dyn_cast<MemRefType>(op.getOutputs()[0].getType());
    if (!a || !b || !c) return failure();
    if (!a.hasStaticShape() || !b.hasStaticShape() || !c.hasStaticShape()) return failure();
    if (a.getRank() != 2 || b.getRank() != 2 || c.getRank() != 2) return failure();

    int64_t M = a.getShape()[0];
    int64_t K = a.getShape()[1];
    int64_t N = b.getShape()[1];
    if (b.getShape()[0] != K) return failure();
    if (c.getShape() != ArrayRef<int64_t>{M, N}) return failure();
    if (a.getElementType() != b.getElementType() || a.getElementType() != c.getElementType()) return failure();

    auto templateArgs = rewriter.getArrayAttr(
      {TypeAttr::get(a.getElementType()), rewriter.getIndexAttr(M), rewriter.getIndexAttr(N), rewriter.getIndexAttr(K)}
    );
    pumem::RuntimeCallOp::create(
      rewriter, op.getLoc(), TypeRange{}, FlatSymbolRefAttr::get(op.getContext(), "runtime_matmul"), templateArgs,
      ValueRange{op.getInputs()[0], op.getInputs()[1], op.getOutputs()[0]}
    );
    rewriter.eraseOp(op);
    return success();
  }
};

class TransposeToRuntime : public OpRewritePattern<linalg::TransposeOp>
{
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::TransposeOp op, PatternRewriter& rewriter) const override
  {
    auto src = dyn_cast<MemRefType>(op.getInput().getType());
    auto dst = dyn_cast<MemRefType>(op.getInit().getType());
    if (!src || !dst) return failure();
    if (!src.hasStaticShape() || !dst.hasStaticShape()) return failure();
    if (src.getRank() != 2 || dst.getRank() != 2) return failure();
    if (src.getElementType() != dst.getElementType()) return failure();

    int64_t rows = src.getShape()[0];
    int64_t cols = src.getShape()[1];
    if (dst.getShape() != ArrayRef<int64_t>{cols, rows}) return failure();
    if (op.getPermutation() != ArrayRef<int64_t>{1, 0}) return failure();

    auto templateArgs = rewriter.getArrayAttr(
      {TypeAttr::get(src.getElementType()), rewriter.getIndexAttr(rows), rewriter.getIndexAttr(cols)}
    );
    pumem::RuntimeCallOp::create(
      rewriter, op.getLoc(), TypeRange{}, FlatSymbolRefAttr::get(op.getContext(), "runtime_transpose"), templateArgs,
      ValueRange{op.getInput(), op.getInit()}
    );
    rewriter.eraseOp(op);
    return success();
  }
};

struct LinalgToRuntimePass : public ::impl::LinalgToRuntimeBase<LinalgToRuntimePass>
{
  using LinalgToRuntimeBase::LinalgToRuntimeBase;

  void getDependentDialects(DialectRegistry& registry) const override { registry.insert<pumem::PUMemDialect>(); }

  void runOnOperation() override
  {
    RewritePatternSet patterns(&getContext());
    patterns.add<MatmulToRuntime, TransposeToRuntime>(&getContext());
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns)))) signalPassFailure();
  }
};

} // namespace

std::unique_ptr<OperationPass<ModuleOp>> triton::createLinalgToRuntimePass()
{
  return std::make_unique<LinalgToRuntimePass>();
}
