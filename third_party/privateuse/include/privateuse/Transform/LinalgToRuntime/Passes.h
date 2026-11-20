#ifndef PRIVATEUSE_TRANSFORM_LINALGTORUNTIME_PASSES_H
#define PRIVATEUSE_TRANSFORM_LINALGTORUNTIME_PASSES_H

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

#include <memory>

namespace mlir {
namespace triton {

std::unique_ptr<OperationPass<ModuleOp>> createLinalgToRuntimePass();

#define GEN_PASS_REGISTRATION
#include "privateuse/Transform/LinalgToRuntime/Passes.h.inc"

} // namespace triton
} // namespace mlir

#endif
