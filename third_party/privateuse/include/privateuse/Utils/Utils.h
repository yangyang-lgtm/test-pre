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

class Int128
{
public:
  constexpr Int128() = default;

  static constexpr Int128 fromInt64(int64_t value)
  {
    Int128 result;
    result.hi_ = value < 0 ? ~uint64_t(0) : uint64_t(0);
    result.lo_ = static_cast<uint64_t>(value);
    return result;
  }

  bool isNegative() const { return static_cast<int64_t>(hi_) < 0; }
  bool fitsInt64() const { return hi_ == ((lo_ >> 63) ? ~uint64_t(0) : uint64_t(0)); }
  int64_t toInt64() const { return static_cast<int64_t>(lo_); }

  friend constexpr Int128 operator+(Int128 a, Int128 b)
  {
    Int128 result;
    result.lo_ = a.lo_ + b.lo_;
    result.hi_ = a.hi_ + b.hi_ + (result.lo_ < a.lo_ ? 1 : 0);
    return result;
  }

  friend constexpr Int128 operator-(Int128 a, Int128 b)
  {
    Int128 result;
    result.lo_ = a.lo_ - b.lo_;
    result.hi_ = a.hi_ - b.hi_ - (a.lo_ < b.lo_ ? 1 : 0);
    return result;
  }

  friend Int128 operator*(Int128 a, Int128 b)
  {
    uint64_t xHi, xLo, yHi, yLo;
    magnitude(a, xHi, xLo);
    magnitude(b, yHi, yLo);
    (void)xHi;
    (void)yHi;

    uint64_t x0  = xLo & 0xffffffff;
    uint64_t x1  = xLo >> 32;
    uint64_t y0  = yLo & 0xffffffff;
    uint64_t y1  = yLo >> 32;
    uint64_t p00 = x0 * y0;
    uint64_t p01 = x0 * y1;
    uint64_t p10 = x1 * y0;
    uint64_t p11 = x1 * y1;
    uint64_t mid = (p00 >> 32) + (p01 & 0xffffffff) + (p10 & 0xffffffff);

    Int128 result;
    result.lo_ = (mid << 32) | (p00 & 0xffffffff);
    result.hi_ = p11 + (p01 >> 32) + (p10 >> 32) + (mid >> 32);

    if (a.isNegative() != b.isNegative()) result = negate(result);
    return result;
  }

  friend bool operator==(Int128 a, Int128 b) { return a.hi_ == b.hi_ && a.lo_ == b.lo_; }
  friend bool operator!=(Int128 a, Int128 b) { return !(a == b); }

  friend bool operator<(Int128 a, Int128 b)
  {
    int64_t aHi = static_cast<int64_t>(a.hi_);
    int64_t bHi = static_cast<int64_t>(b.hi_);
    if (aHi != bHi) return aHi < bHi;
    return a.lo_ < b.lo_;
  }

  friend bool operator<=(Int128 a, Int128 b) { return !(b < a); }
  friend bool operator>(Int128 a, Int128 b) { return b < a; }
  friend bool operator>=(Int128 a, Int128 b) { return !(a < b); }

private:
  constexpr Int128(uint64_t hi, uint64_t lo) : hi_(hi), lo_(lo) {}

  static void magnitude(Int128 value, uint64_t& hi, uint64_t& lo)
  {
    if (!value.isNegative()) {
      hi = value.hi_;
      lo = value.lo_;
      return;
    }
    lo = ~value.lo_ + 1;
    hi = ~value.hi_ + (lo == 0 ? 1 : 0);
  }

  static Int128 negate(Int128 value)
  {
    Int128 result;
    result.lo_ = ~value.lo_ + 1;
    result.hi_ = ~value.hi_ + (result.lo_ == 0 ? 1 : 0);
    return result;
  }

  uint64_t hi_ = 0;
  uint64_t lo_ = 0;
};

inline bool checkedAdd(int64_t lhs, int64_t rhs, int64_t& result)
{
  Int128 sum = Int128::fromInt64(lhs) + Int128::fromInt64(rhs);
  if (!sum.fitsInt64()) return false;
  result = sum.toInt64();
  return true;
}

inline bool checkedSub(int64_t lhs, int64_t rhs, int64_t& result)
{
  Int128 diff = Int128::fromInt64(lhs) - Int128::fromInt64(rhs);
  if (!diff.fitsInt64()) return false;
  result = diff.toInt64();
  return true;
}

inline bool checkedMul(int64_t lhs, int64_t rhs, int64_t& result)
{
  Int128 product = Int128::fromInt64(lhs) * Int128::fromInt64(rhs);
  if (!product.fitsInt64()) return false;
  result = product.toInt64();
  return true;
}

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
