#ifndef PRIVATEUSE_CONVERSION_PASSES_H
#define PRIVATEUSE_CONVERSION_PASSES_H

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

#include <memory>

namespace mlir {
namespace triton {

std::unique_ptr<OperationPass<ModuleOp>> createAnalyzeAccessPass();
std::unique_ptr<OperationPass<ModuleOp>> createLowerStructuredAccessPass();
std::unique_ptr<OperationPass<ModuleOp>> createPtrToMemrefPass();

#define GEN_PASS_REGISTRATION
#include "privateuse/Conversion/Passes.h.inc"

} // namespace triton
} // namespace mlir

#endif
