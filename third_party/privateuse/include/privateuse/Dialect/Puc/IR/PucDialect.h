#ifndef MLIR_DIALECT_PUC_IR_PUC_DIALECT_H_
#define MLIR_DIALECT_PUC_IR_PUC_DIALECT_H_

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/Interfaces/CallInterfaces.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Interfaces/LoopLikeInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#include "privateuse/Dialect/Puc/IR/PucDialect.h.inc"

#define GET_OP_CLASSES
#include "privateuse/Dialect/Puc/IR/PucOps.h.inc"

#endif
