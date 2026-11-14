#ifndef PRIVATEUSE_TRANSFORM_LINALGTORUNTIMECALL_PASSES_H
#define PRIVATEUSE_TRANSFORM_LINALGTORUNTIMECALL_PASSES_H

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

#include <memory>

namespace mlir {
namespace triton {

std::unique_ptr<OperationPass<ModuleOp>> createLinalgToRuntimeCallPass();

#define GEN_PASS_REGISTRATION
#include "privateuse/Transform/LinalgToRuntimeCall/Passes.h.inc"

} // namespace triton
} // namespace mlir

#endif
