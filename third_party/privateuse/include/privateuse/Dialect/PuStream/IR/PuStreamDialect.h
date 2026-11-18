#ifndef MLIR_DIALECT_PUSTREAM_IR_PUSTREAM_DIALECT_H_
#define MLIR_DIALECT_PUSTREAM_IR_PUSTREAM_DIALECT_H_

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/DialectRegistry.h"

#include "privateuse/Dialect/PuStream/IR/PuStreamDialect.h.inc"

#define GET_TYPEDEF_CLASSES
#include "privateuse/Dialect/PuStream/IR/PuStreamTypes.h.inc"

#define GET_OP_CLASSES
#include "privateuse/Dialect/PuStream/IR/PuStreamOps.h.inc"

#endif // MLIR_DIALECT_PUSTREAM_IR_PUSTREAM_DIALECT_H_