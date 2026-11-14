#ifndef PRIVATEUSE_CONVERSION_SCFTOPUMEM_TRITON_SCF_TO_PUMEM_H
#define PRIVATEUSE_CONVERSION_SCFTOPUMEM_TRITON_SCF_TO_PUMEM_H

#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir {
namespace triton {

std::unique_ptr<OperationPass<ModuleOp>> createScfToPumemPass();

}
}

#endif
