#ifndef TRITON_TRANSFORM_PUCLOCALIZECARRIED_PUCLOCALIZECARRIED_H
#define TRITON_TRANSFORM_PUCLOCALIZECARRIED_PUCLOCALIZECARRIED_H

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

namespace mlir {
namespace triton {

std::unique_ptr<OperationPass<ModuleOp>> createPucLocalizeCarriedPass();

}
}

#endif
