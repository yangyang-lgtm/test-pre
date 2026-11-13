#ifndef PRIVATEUSE_UTILS_ACCESSANALYSIS_PASSES_H
#define PRIVATEUSE_UTILS_ACCESSANALYSIS_PASSES_H

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

#include <memory>

namespace mlir {
namespace triton {

std::unique_ptr<OperationPass<ModuleOp>> createAnalyzeAccessPass();

#define GEN_PASS_REGISTRATION
#include "privateuse/Utils/AccessAnalysis/Passes.h.inc"

} // namespace triton
} // namespace mlir

#endif
