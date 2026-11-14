#ifndef PRIVATEUSE_CONVERSION_MATHTOPUMATH_TRITON_MATH_TO_PUARITH_H
#define PRIVATEUSE_CONVERSION_MATHTOPUMATH_TRITON_MATH_TO_PUARITH_H

#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir {
namespace triton {

std::unique_ptr<OperationPass<ModuleOp>> createMathToPumathPass();

}
}

#endif
