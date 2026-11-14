#ifndef TRITON_TRANSFORM_PUCVOIDPTRARGS_PUCVOIDPTRARGS_H
#define TRITON_TRANSFORM_PUCVOIDPTRARGS_PUCVOIDPTRARGS_H

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

namespace mlir {
namespace triton {

std::unique_ptr<OperationPass<ModuleOp>> createPucVoidPtrArgsPass();

}
}

#endif
