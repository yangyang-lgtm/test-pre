#include "privateuse/Transform/LinalgToRuntimeCall/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/DialectConversion.h"
#include "privateuse/Dialect/PUMem/IR/PUMemDialect.h"

#include "llvm/ADT/STLFunctionalExtras.h"

#define GEN_PASS_DECL_LINALGTORUNTIMECALL
#include "privateuse/Transform/LinalgToRuntimeCall/Passes.h.inc"

#define GEN_PASS_DEF_LINALGTORUNTIMECALL
#include "privateuse/Transform/LinalgToRuntimeCall/Passes.h.inc"

using namespace mlir;
using namespace mlir::triton;

namespace {

FailureOr<SmallVector<MemRefType>> checkMemrefOperands(
  Operation* op, function_ref<LogicalResult(const Twine&)> fail,
  function_ref<LogicalResult(MemRefType, ArrayRef<MemRefType>)> check
)
{
  SmallVector<MemRefType> types;
  for (Value operand : op->getOperands()) {
    auto type = dyn_cast<MemRefType>(operand.getType());
    if (!type) return fail("expected memref operands");
    if (failed(check(type, types))) return failure();
    types.push_back(type);
  }
  return types;
}

struct MatmulToRuntimeCallPattern : OpRewritePattern<linalg::MatmulOp>
{
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::MatmulOp op, PatternRewriter& rewriter) const override
  {
    auto fail = [&](const Twine& reason) -> LogicalResult {
      return op.emitError("cannot lower linalg.matmul to a runtime call: ") << reason;
    };
    auto check = [&](MemRefType type, ArrayRef<MemRefType> previous) -> LogicalResult {
      if (!type.hasStaticShape()) return fail("expected statically shaped operands");
      if (type.getRank() != 2) return fail("expected rank-2 operands");
      if (!previous.empty() && type.getElementType() != previous.front().getElementType())
        return fail("expected matching element types");
      return success();
    };

    if (op.getNumDpsInputs() != 2 || op.getNumDpsInits() != 1) return fail("expected two inputs and one init");

    auto types = checkMemrefOperands(op, fail, check);
    if (failed(types)) return failure();
    auto aTy = (*types)[0], bTy = (*types)[1], cTy = (*types)[2];

    int64_t M = aTy.getShape()[0];
    int64_t K = aTy.getShape()[1];
    int64_t N = bTy.getShape()[1];
    if (bTy.getShape()[0] != K) return fail("expected lhs shape [M, K] and rhs shape [K, N]");
    if (cTy.getShape() != ArrayRef<int64_t>{M, N}) return fail("expected init shape to match the result shape");

    auto name         = FlatSymbolRefAttr::get(op.getContext(), "runtime_matmul");
    auto templateArgs = rewriter.getArrayAttr(
      {TypeAttr::get(aTy.getElementType()), rewriter.getIndexAttr(M), rewriter.getIndexAttr(N), rewriter.getIndexAttr(K)
      }
    );
    rewriter.replaceOpWithNewOp<pumem::RuntimeCallOp>(op, TypeRange{}, name, templateArgs, op->getOperands());
    return success();
  }
};

struct TransposeToRuntimeCallPattern : OpRewritePattern<linalg::TransposeOp>
{
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::TransposeOp op, PatternRewriter& rewriter) const override
  {
    auto fail = [&](const Twine& reason) -> LogicalResult {
      return op.emitError("cannot lower linalg.transpose to a runtime call: ") << reason;
    };
    auto check = [&](MemRefType type, ArrayRef<MemRefType> previous) -> LogicalResult {
      if (!type.hasStaticShape()) return fail("expected statically shaped operands");
      if (type.getRank() != 2) return fail("expected rank-2 operands");
      if (!previous.empty() && type.getElementType() != previous.front().getElementType())
        return fail("expected matching element types");
      return success();
    };

    if (op.getNumDpsInputs() != 1 || op.getNumDpsInits() != 1) return fail("expected one input and one init");

    auto types = checkMemrefOperands(op, fail, check);
    if (failed(types)) return failure();
    auto srcTy = (*types)[0], dstTy = (*types)[1];

    auto permutation = op.getPermutation();
    if (permutation.size() != 2 || permutation[0] != 1 || permutation[1] != 0)
      return fail("expected permutation [1, 0]");

    int64_t rows = srcTy.getShape()[0];
    int64_t cols = srcTy.getShape()[1];
    if (dstTy.getShape() != ArrayRef<int64_t>{cols, rows})
      return fail("expected init shape to be the transposed input shape");

    auto name         = FlatSymbolRefAttr::get(op.getContext(), "runtime_transpose");
    auto templateArgs = rewriter.getArrayAttr(
      {TypeAttr::get(srcTy.getElementType()), rewriter.getIndexAttr(rows), rewriter.getIndexAttr(cols)}
    );
    rewriter.replaceOpWithNewOp<pumem::RuntimeCallOp>(op, TypeRange{}, name, templateArgs, op->getOperands());
    return success();
  }
};

struct LinalgToRuntimeCallPass : public ::impl::LinalgToRuntimeCallBase<LinalgToRuntimeCallPass>
{
  using LinalgToRuntimeCallBase::LinalgToRuntimeCallBase;

  void getDependentDialects(DialectRegistry& registry) const override
  {
    registry.insert<arith::ArithDialect, linalg::LinalgDialect, pumem::PUMemDialect>();
  }

  void runOnOperation() override
  {
    ConversionTarget target(getContext());
    target.addLegalDialect<arith::ArithDialect, pumem::PUMemDialect>();
    target.addIllegalOp<linalg::MatmulOp, linalg::TransposeOp>();

    RewritePatternSet patterns(&getContext());
    patterns.add<MatmulToRuntimeCallPattern, TransposeToRuntimeCallPattern>(&getContext());
    if (failed(applyPartialConversion(getOperation(), target, std::move(patterns)))) signalPassFailure();
  }
};

} // namespace

std::unique_ptr<OperationPass<ModuleOp>> triton::createLinalgToRuntimeCallPass()
{
  return std::make_unique<LinalgToRuntimeCallPass>();
}
