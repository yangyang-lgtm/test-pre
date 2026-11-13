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
  if (!OpTrait::hasElementwiseMappableTraits(op)) return false;
  bool hasRankedTensor = false;
  for (Type type : op->getOperandTypes()) {
    if (isa<RankedTensorType>(type)) {
      hasRankedTensor = true;
      continue;
    }
    if (!type.isIntOrFloat() && !type.isIndex() && !isa<ComplexType>(type)) return false;
  }
  return hasRankedTensor;
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
      affine::AffineDialect,
      triton::TritonDialect
    >();
    // clang-format on

    target.addLegalOp<ModuleOp, UnrealizedConversionCastOp>();

    target.addDynamicallyLegalDialect<arith::ArithDialect, math::MathDialect>([](Operation* op) {
      if (auto constOp = dyn_cast<arith::ConstantOp>(op)) {
        if (!isa<RankedTensorType>(constOp.getResult().getType())) return true;
        if (auto denseAttr = dyn_cast<DenseElementsAttr>(constOp.getValue()))
          if (denseAttr.isSplat() && isa<FloatType, IntegerType>(denseAttr.getElementType())) return false;
        return true;
      }
      return !isElementwiseMappableOnRankedTensors(op);
    });

    target.addDynamicallyLegalOp<triton::SplatOp, triton::BroadcastOp, triton::ExpandDimsOp, triton::MakeRangeOp>(
      [](Operation* op) { return llvm::any_of(op->getResultTypes(), isPointerLike); }
    );
    target.addDynamicallyLegalOp<triton::ReshapeOp>([](triton::ReshapeOp op) {
      auto type = cast<RankedTensorType>(op.getResult().getType());
      if (isPointerLike(type)) return true;
      return !type.hasStaticShape();
    });
    target.addDynamicallyLegalOp<triton::BitcastOp>([](triton::BitcastOp op) { return isPointerLike(op.getType()); });

    populateArithToLinalgConversionPatterns(patterns);

    if (failed(applyPartialConversion(module, target, std::move(patterns)))) {
      signalPassFailure();
      return;
    }

    warnUnconvertedValueOps(module);
  }

  void warnUnconvertedValueOps(ModuleOp module)
  {
    module.walk([](Operation* op) {
      if (op->getNumResults() == 0) return;
      if (!isa<triton::TritonDialect>(op->getDialect())) return;
      if (isa<triton::LoadOp, triton::StoreOp, triton::AddPtrOp>(op)) return;
      if (isa<
            triton::SplatOp, triton::BroadcastOp, triton::ExpandDimsOp, triton::ReshapeOp, triton::MakeRangeOp,
            triton::BitcastOp>(op))
        return;
      bool hasNonPointerTensorResult = llvm::any_of(op->getResultTypes(), [](Type type) {
        auto tensor = dyn_cast<RankedTensorType>(type);
        return tensor && !isa<triton::PointerType>(tensor.getElementType());
      });
      if (hasNonPointerTensorResult || isa<triton::ReduceOp>(op))
        op->emitWarning("privateuse-arith-to-linalg: unconverted tensor value op '") << op->getName() << "'";
    });
  }
};

} // namespace

std::unique_ptr<OperationPass<ModuleOp>> triton::createArithToLinalgPass()
{
  return std::make_unique<ArithToLinalgPass>();
}
