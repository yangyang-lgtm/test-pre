#ifndef TRITON_TRANSFORM_PUCREPEAT_PUCREPEAT_H
#define TRITON_TRANSFORM_PUCREPEAT_PUCREPEAT_H

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

#include <cstdint>

namespace mlir {
namespace triton {

std::unique_ptr<OperationPass<ModuleOp>> createPucRepeatPass(int64_t iterations = 100);

}
}

#endif
