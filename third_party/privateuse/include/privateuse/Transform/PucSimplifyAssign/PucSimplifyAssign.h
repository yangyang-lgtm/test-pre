#ifndef TRITON_TRANSFORM_PUCSIMPLIFYASSIGN_PUCSIMPLIFYASSIGN_H
#define TRITON_TRANSFORM_PUCSIMPLIFYASSIGN_PUCSIMPLIFYASSIGN_H

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

namespace mlir {
namespace triton {

std::unique_ptr<OperationPass<ModuleOp>> createPucSimplifyAssignPass();

}
}

#endif
