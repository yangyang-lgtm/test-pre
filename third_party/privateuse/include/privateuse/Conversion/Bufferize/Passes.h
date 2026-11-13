#ifndef PRIVATEUSE_CONVERSION_BUFFERIZE_PASSES_H
#define PRIVATEUSE_CONVERSION_BUFFERIZE_PASSES_H

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

#include <memory>

namespace mlir {
namespace triton {

std::unique_ptr<OperationPass<ModuleOp>> createBufferizePass();

#define GEN_PASS_REGISTRATION
#include "privateuse/Conversion/Bufferize/Passes.h.inc"

} // namespace triton
} // namespace mlir

#endif
