#ifndef PRIVATEUSE_UTILITY_H
#define PRIVATEUSE_UTILITY_H

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "privateuse/Utils/Constants.h"

#include "llvm/ADT/ArrayRef.h"

#include <cstdint>

namespace mlir {

class OpBuilder;

namespace triton {

class LoadOp;

bool isPtrTypeLike(Type t);

Type storageElementType(Type element);

Value maskedFillValue(LoadOp load, Type elementType, OpBuilder& builder, Location loc);

Value createBufferWindow(
  OpBuilder& builder, Location loc, Value buf, ArrayRef<int64_t> shape, ArrayRef<Value> dimSizes
);

template <typename Fn> int64_t unsignedOp(int64_t lhs, int64_t rhs, Fn op)
{
  return static_cast<int64_t>(op(static_cast<uint64_t>(lhs), static_cast<uint64_t>(rhs)));
}

template <typename Fn> int64_t unsignedOp(int64_t value, Fn op)
{
  return static_cast<int64_t>(op(static_cast<uint64_t>(value)));
}

} // namespace triton
} // namespace mlir

#endif
