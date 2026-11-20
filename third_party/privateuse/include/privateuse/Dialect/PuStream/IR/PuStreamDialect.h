#ifndef PRIVATEUSE_DIALECT_PUSTREAM_IR_PUSTREAMDIALECT_H
#define PRIVATEUSE_DIALECT_PUSTREAM_IR_PUSTREAMDIALECT_H

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#include "privateuse/Dialect/PuStream/IR/PuStreamDialect.h.inc"

#define GET_TYPEDEF_CLASSES
#include "privateuse/Dialect/PuStream/IR/PuStreamTypes.h.inc"

#define GET_OP_CLASSES
#include "privateuse/Dialect/PuStream/IR/PuStreamOps.h.inc"

#endif
