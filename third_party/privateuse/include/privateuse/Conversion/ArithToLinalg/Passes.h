#ifndef PRIVATEUSE_CONVERSION_ARITHTOLINALG_PASSES_H
#define PRIVATEUSE_CONVERSION_ARITHTOLINALG_PASSES_H

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

#include <memory>

namespace mlir {
namespace triton {

std::unique_ptr<OperationPass<ModuleOp>> createArithToLinalgPass();

#define GEN_PASS_REGISTRATION
#include "privateuse/Conversion/ArithToLinalg/Passes.h.inc"

} // namespace triton
} // namespace mlir

#endif
