#ifndef MLIR_DIALECT_PUMEM_IR_PUMEM_DIALECT_H_
#define MLIR_DIALECT_PUMEM_IR_PUMEM_DIALECT_H_

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#include "privateuse/Dialect/PUMem/IR/PUMemDialect.h.inc"

#define GET_OP_CLASSES
#include "privateuse/Dialect/PUMem/IR/PUMemOps.h.inc"

#endif
