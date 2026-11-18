#ifndef PRIVATEUSE_UTILITY_H
#define PRIVATEUSE_UTILITY_H

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "privateuse/Utils/Constants.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLFunctionalExtras.h"

#include <cstdint>
#include <exception>
#include <string>
#include <utility>

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

/// Error thrown by a speculative rewrite that cannot convert its target. The
/// message is what runWithRestore reports when logging is enabled.
class RewriteError : public std::exception
{
public:
  explicit RewriteError(std::string message) : message_(std::move(message)) {}

  const char* what() const noexcept override { return message_.c_str(); }

private:
  std::string message_;
};

/// Runs `body`, which may rewrite `op` and throws `RewriteError` when the
/// rewrite fails. On failure `op` is replaced by a snapshot taken before the
/// call and the error is swallowed, so the caller keeps the original IR
/// instead of failing the pipeline. When `logError` is set the error message
/// is printed.
void runWithRestore(Operation* op, llvm::function_ref<void()> body, bool logError);

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
