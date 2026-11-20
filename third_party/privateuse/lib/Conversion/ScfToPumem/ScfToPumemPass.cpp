#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "privateuse/Conversion/Common/I1ToI8.h"
#include "privateuse/Conversion/ScfToPumem/ScfToPumem.h"
#include "privateuse/Dialect/PUMem/IR/PUMemDialect.h"
#include "privateuse/Utils/Constants.h"

#define DEBUG_TYPE "scf-to-pumem"

using namespace mlir;
using namespace mlir::triton;

#define GEN_PASS_DEF_SCFTOPUMEM
#include "privateuse/Conversion/ScfToPumem/Passes.h.inc"

namespace {

class I1TypeConverter : public TypeConverter
{
public:
  I1TypeConverter() { addI1ToI8Conversions(*this); }
};

class ForOpConversion : public OpConversionPattern<scf::ForOp>
{
public:
  using OpConversionPattern<scf::ForOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(scf::ForOp op, scf::ForOp::Adaptor adaptor, ConversionPatternRewriter& rewriter) const override
  {
    auto toIndex = [&](Value v) -> Value {
      if (isa<IndexType>(v.getType())) return v;
      return arith::IndexCastOp::create(rewriter, op.getLoc(), rewriter.getIndexType(), v);
    };

    auto newFor = pumem::ForOp::create(
      rewriter, op.getLoc(), toIndex(adaptor.getLowerBound()), toIndex(adaptor.getUpperBound()),
      toIndex(adaptor.getStep()), adaptor.getInitArgs()
    );

    for (StringRef name : {kPipelineAttr, kPipelinedLoopAttr})
      if (Attribute attr = op->getAttr(name)) newFor->setAttr(name, attr);

    Block* oldBody = &op.getRegion().front();
    Block* newBody = newFor.getBody();

    if (!newBody->empty()) rewriter.eraseOp(&newBody->back());

    rewriter.setInsertionPointToStart(newBody);
    Value oldIndvar = oldBody->getArgument(0);
    if (isa<IndexType>(oldIndvar.getType())) {
      oldIndvar.replaceAllUsesWith(newBody->getArgument(0));
    } else {
      Value indvarReplacement =
        arith::IndexCastOp::create(rewriter, op.getLoc(), oldIndvar.getType(), newBody->getArgument(0));
      oldIndvar.replaceAllUsesWith(indvarReplacement);
    }

    newBody->getOperations().splice(newBody->end(), oldBody->getOperations());

    for (auto [oldArg, newArg] : llvm::zip(oldBody->getArguments().drop_front(), newBody->getArguments().drop_front()))
      oldArg.replaceAllUsesWith(newArg);

    rewriter.replaceOp(op, newFor.getResults());
    return success();
  }
};

class IfOpConversion : public OpConversionPattern<scf::IfOp>
{
public:
  using OpConversionPattern<scf::IfOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(scf::IfOp op, scf::IfOp::Adaptor adaptor, ConversionPatternRewriter& rewriter) const override
  {
    Value cond = adaptor.getCondition();
    auto intTy = dyn_cast<IntegerType>(cond.getType());
    if (!intTy) return op.emitOpError("condition must be an integer type, got ") << cond.getType();
    if (intTy.getWidth() < 8) {
      cond = arith::ExtSIOp::create(rewriter, op.getLoc(), rewriter.getI8Type(), cond);
    } else if (intTy.getWidth() > 8) {
      cond = arith::TruncIOp::create(rewriter, op.getLoc(), rewriter.getI8Type(), cond);
    }

    SmallVector<Type> resultTypes;
    if (failed(getTypeConverter()->convertTypes(op.getResultTypes(), resultTypes))) return failure();

    auto newIf = pumem::IfOp::create(rewriter, op.getLoc(), resultTypes, cond, false, false);
    rewriter.inlineRegionBefore(op.getThenRegion(), newIf.getThenRegion(), newIf.getThenRegion().end());
    if (!op.getElseRegion().empty())
      rewriter.inlineRegionBefore(op.getElseRegion(), newIf.getElseRegion(), newIf.getElseRegion().end());

    rewriter.replaceOp(op, newIf.getResults());
    return success();
  }
};

class YieldOpConversion : public OpConversionPattern<scf::YieldOp>
{
public:
  using OpConversionPattern<scf::YieldOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(scf::YieldOp op, scf::YieldOp::Adaptor adaptor, ConversionPatternRewriter& rewriter) const override
  {
    rewriter.replaceOpWithNewOp<pumem::YieldOp>(op, adaptor.getOperands());
    return success();
  }
};

class ScfToPumemPass : public ::impl::ScfToPumemBase<ScfToPumemPass>
{
public:
  void getDependentDialects(DialectRegistry& registry) const override
  {
    // clang-format off
    registry.insert<
      scf::SCFDialect,
      pumem::PUMemDialect,
      arith::ArithDialect,
      func::FuncDialect,
      memref::MemRefDialect,
      vector::VectorDialect
    >();
    // clang-format on
  }

  void runOnOperation() override
  {
    I1TypeConverter typeConverter;
    RewritePatternSet patterns(&getContext());

    // clang-format off
    patterns.add<
      ForOpConversion,
      IfOpConversion,
      YieldOpConversion
    >(typeConverter, &getContext());
    // clang-format on

    populateI1ToI8Patterns(patterns, typeConverter);

    patterns.add<I1OpConversion<pumem::TileGatherOp>, I1OpConversion<pumem::TileScatterOp>>(
      typeConverter, &getContext()
    );

    ConversionTarget target(getContext());
    target.addLegalDialect<pumem::PUMemDialect, arith::ArithDialect>();
    target.addIllegalDialect<scf::SCFDialect>();
    markI1ToI8OpsDynamicallyLegal(target, typeConverter);
    target.addDynamicallyLegalOp<pumem::TileGatherOp, pumem::TileScatterOp>([&](Operation* op) {
      return typeConverter.isLegal(op);
    });

    if (failed(applyPartialConversion(getOperation(), target, std::move(patterns)))) signalPassFailure();
  }
};

}

std::unique_ptr<OperationPass<ModuleOp>> triton::createScfToPumemPass()
{
  return std::make_unique<ScfToPumemPass>();
}
