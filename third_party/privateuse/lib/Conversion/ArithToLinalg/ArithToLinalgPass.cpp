#include "privateuse/Conversion/ArithToLinalg/ConversionPatterns.hpp"
#include "privateuse/Conversion/ArithToLinalg/Passes.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Operation.h"
#include "mlir/Transforms/DialectConversion.h"
#include "privateuse/Dialect/PUMem/IR/PUMemDialect.h"
#include "triton/Dialect/Triton/IR/Dialect.h"

#define GEN_PASS_DEF_ARITHTOLINALG
#include "privateuse/Conversion/ArithToLinalg/Passes.h.inc"

using namespace mlir;
using namespace mlir::triton;

namespace {

bool isElementwiseMappableOnRankedTensors(Operation* op)
{
  if (auto constOp = dyn_cast<arith::ConstantOp>(op)) {
    if (!isa<RankedTensorType>(constOp.getResult().getType())) return false;
    auto dense = dyn_cast<DenseElementsAttr>(constOp.getValue());
    return dense && dense.isSplat();
  }
  if (!OpTrait::hasElementwiseMappableTraits(op)) return false;

  auto types = op->getOperandTypes();
  if (llvm::all_of(types, [](Type t) { return !isa<RankedTensorType>(t); })) {
    return false;
  }
  return llvm::all_of(types, [](Type t) {
    return isa<RankedTensorType>(t) || t.isIntOrFloat() || t.isIndex() || isa<ComplexType>(t);
  });
}

struct ArithToLinalgPass : public ::impl::ArithToLinalgBase<ArithToLinalgPass>
{
  using ArithToLinalgBase::ArithToLinalgBase;

  void getDependentDialects(DialectRegistry& registry) const override
  {
    // clang-format off
    registry.insert<
      arith::ArithDialect,
      math::MathDialect,
      linalg::LinalgDialect,
      tensor::TensorDialect,
      affine::AffineDialect,
      triton::TritonDialect
    >();
    // clang-format on
  }

  void runOnOperation() override
  {
    ModuleOp module = getOperation();
    if (module.walk([](triton::CallOp) { return WalkResult::interrupt(); }).wasInterrupted()) return;

    RewritePatternSet patterns(&getContext());
    ConversionTarget target(getContext());

    // clang-format off
    target.addLegalDialect<
      arith::ArithDialect,
      math::MathDialect,
      linalg::LinalgDialect,
      tensor::TensorDialect,
      bufferization::BufferizationDialect,
      memref::MemRefDialect,
      scf::SCFDialect,
      func::FuncDialect,
      cf::ControlFlowDialect,
      pumem::PUMemDialect,
      affine::AffineDialect
    >();
    // clang-format on

    target.addIllegalDialect<triton::TritonDialect>();
    target.addLegalOp<triton::FuncOp, triton::ReturnOp, triton::GetProgramIdOp>();

    target.addDynamicallyLegalDialect<arith::ArithDialect, math::MathDialect>([](Operation* op) {
      return !isElementwiseMappableOnRankedTensors(op);
    });

    populateArithToLinalgConversionPatterns(patterns);

    if (failed(applyPartialConversion(module, target, std::move(patterns)))) {
      signalPassFailure();
      return;
    }
  }
};

} // namespace

std::unique_ptr<OperationPass<ModuleOp>> triton::createArithToLinalgPass()
{
  return std::make_unique<ArithToLinalgPass>();
}
