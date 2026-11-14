#ifndef PRIVATEUSE_DIALECT_PUMATH_IR_PUMATH_DIALECT_H_
#define PRIVATEUSE_DIALECT_PUMATH_IR_PUMATH_DIALECT_H_

#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Interfaces/VectorInterfaces.h"

#include "privateuse/Dialect/PuArith/IR/PuArithDialect.h"

#include "privateuse/Dialect/PuMath/IR/PuMathDialect.h.inc"

#define GET_OP_CLASSES
#include "privateuse/Dialect/PuMath/IR/PuMathOps.h.inc"

#endif
