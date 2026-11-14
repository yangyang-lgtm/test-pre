#ifndef PRIVATEUSE_CONVERSION_ARITHTOPUARITH_TRITON_ARITH_TO_PUARITH_H
#define PRIVATEUSE_CONVERSION_ARITHTOPUARITH_TRITON_ARITH_TO_PUARITH_H

#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir {
namespace triton {

std::unique_ptr<OperationPass<ModuleOp>> createArithToPuarithPass();

}
}

#endif
