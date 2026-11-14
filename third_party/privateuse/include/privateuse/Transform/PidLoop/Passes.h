#ifndef PRIVATEUSE_TRANSFORM_PIDLOOP_PASSES_H
#define PRIVATEUSE_TRANSFORM_PIDLOOP_PASSES_H

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

#include <cstdint>
#include <memory>

namespace mlir {
namespace triton {

std::unique_ptr<OperationPass<ModuleOp>> createPidLoopPass(int64_t numPids = 0);

#define GEN_PASS_REGISTRATION
#include "privateuse/Transform/PidLoop/Passes.h.inc"

} // namespace triton
} // namespace mlir

#endif
