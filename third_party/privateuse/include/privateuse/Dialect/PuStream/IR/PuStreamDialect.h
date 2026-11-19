#ifndef MLIR_DIALECT_PUSTREAM_IR_PUSTREAM_DIALECT_H_
#define MLIR_DIALECT_PUSTREAM_IR_PUSTREAM_DIALECT_H_

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

namespace mlir::pustream {

// The single ordering resource shared by every asynchronous statement of a
// kernel. Copies of one direction are a global FIFO and the host is the only
// issuer, so no pass may reorder two async statements; encoding that as one
// side effect makes the illegal reorderings unrepresentable instead of relying
// on pass discipline.
struct Ordering : public SideEffects::Resource::Base<Ordering>
{
  StringRef getName() const override { return "pustream.ordering"; }
};

} // namespace mlir::pustream

#include "privateuse/Dialect/PuStream/IR/PuStreamDialect.h.inc"

#define GET_TYPEDEF_CLASSES
#include "privateuse/Dialect/PuStream/IR/PuStreamTypes.h.inc"

#define GET_OP_CLASSES
#include "privateuse/Dialect/PuStream/IR/PuStreamOps.h.inc"

#endif // MLIR_DIALECT_PUSTREAM_IR_PUSTREAM_DIALECT_H_
