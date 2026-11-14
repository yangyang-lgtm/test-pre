#include "privateuse/Conversion/ToPuc/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Transforms/DialectConversion.h"
#include "privateuse/Conversion/ToPuc/PucPatterns.h"
#include "privateuse/Dialect/PuArith/IR/PuArithDialect.h"
#include "privateuse/Dialect/PuMath/IR/PuMathDialect.h"
#include "privateuse/Dialect/PUMem/IR/PUMemDialect.h"
#include "privateuse/Dialect/Puc/IR/PucDialect.h"

#define GEN_PASS_DEF_TOPUC
#include "privateuse/Conversion/ToPuc/Passes.h.inc"

using namespace mlir;

namespace {

struct ToPucPass : public ::impl::ToPucBase<ToPucPass>
{
  using ToPucBase::ToPucBase;

  void runOnOperation() override
  {
    ModuleOp moduleOp = getOperation();

    TypeConverter typeConverter;
    typeConverter.addConversion([](Type type) -> std::optional<Type> { return type; });

    RewritePatternSet patterns(&getContext());
    triton::populateFuncToPucPatterns(patterns, typeConverter);
    triton::populatePumemToPucPatterns(patterns, typeConverter);
    triton::populateScalarToPucPatterns(patterns, typeConverter);

    ConversionTarget target(getContext());
    target.addLegalDialect<puc::PucDialect>();
    target.addIllegalOp<UnrealizedConversionCastOp>();

    // clang-format off
    target.addIllegalDialect<
      func::FuncDialect,
      pumem::PUMemDialect,
      puarith::PuArithDialect,
      pumath::PuMathDialect,
      arith::ArithDialect
    >();
    // clang-format on

    if (failed(applyPartialConversion(moduleOp, target, std::move(patterns)))) signalPassFailure();
  }
};

} // namespace

std::unique_ptr<OperationPass<ModuleOp>> mlir::triton::createToPucPass()
{
  return std::make_unique<ToPucPass>();
}
