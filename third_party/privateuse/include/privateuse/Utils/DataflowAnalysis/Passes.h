#ifndef PRIVATEUSE_UTILS_DATAFLOWANALYSIS_PASSES_H
#define PRIVATEUSE_UTILS_DATAFLOWANALYSIS_PASSES_H

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

#include <memory>

namespace mlir {
namespace triton {

std::unique_ptr<OperationPass<ModuleOp>> createAnalyzeDataflowPass();

#define GEN_PASS_REGISTRATION
#include "privateuse/Utils/DataflowAnalysis/Passes.h.inc"

} // namespace triton
} // namespace mlir

#endif
