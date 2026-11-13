#include "privateuse/Conversion/Bufferize/Passes.h"

#include "mlir/Dialect/Bufferization/IR/BufferizableOpInterface.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Bufferization/Transforms/OneShotAnalysis.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinOps.h"
#include "privateuse/Utils/Constants.h"

#define GEN_PASS_DEF_BUFFERIZE
#include "privateuse/Conversion/Bufferize/Passes.h.inc"

using namespace mlir;
using namespace mlir::triton;

namespace {

struct BufferizePass : public ::impl::BufferizeBase<BufferizePass>
{
  using BufferizeBase::BufferizeBase;

  void getDependentDialects(DialectRegistry& registry) const override
  {
    // clang-format off
    registry.insert<
      bufferization::BufferizationDialect,
      linalg::LinalgDialect,
      tensor::TensorDialect
    >();
    // clang-format on
  }

  void runOnOperation() override
  {
    bufferization::OneShotBufferizationOptions options;
    options.allowUnknownOps            = true;
    options.allowReturnAllocsFromLoops = true;
    options.bufferAlignment            = static_cast<unsigned>(kDefaultAlignment);
    options.defaultMemorySpaceFn       = [](bufferization::TensorLikeType type) -> std::optional<Attribute> {
      return IntegerAttr::get(IntegerType::get(type.getContext(), 64), kSramSpace);
    };

    bufferization::BufferizationState state;
    if (failed(bufferization::runOneShotBufferize(getOperation(), options, state))) signalPassFailure();
  }
};

} // namespace

std::unique_ptr<OperationPass<ModuleOp>> triton::createBufferizePass()
{
  return std::make_unique<BufferizePass>();
}
