#include "privateuse/Analysis/ValueRange.h"

#include "privateuse/Utils/Utils.h"

#include <algorithm>
#include <array>
#include <iterator>
#include <limits>

namespace mlir {
namespace triton {

namespace {

ExprPtr makeLeaf(ScalarExpr::Kind kind, int64_t value, SymKind symKind, unsigned symId)
{
  auto expr     = std::make_shared<ScalarExpr>();
  expr->kind    = kind;
  expr->value   = value;
  expr->symKind = symKind;
  expr->symId   = symId;
  return expr;
}

int64_t unsignedMin(int64_t lhs, int64_t rhs)
{
  return unsignedOp(lhs, rhs, [](uint64_t a, uint64_t b) { return std::min(a, b); });
}

int64_t unsignedMax(int64_t lhs, int64_t rhs)
{
  return unsignedOp(lhs, rhs, [](uint64_t a, uint64_t b) { return std::max(a, b); });
}

template <typename T, typename MapT>
std::optional<T>
lookupSymbol(const std::array<std::optional<T>, 3>& pids, const MapT& loopIvs, SymKind kind, unsigned id)
{
  if (kind == SymKind::LoopIv) {
    auto it = loopIvs.find(id);
    if (it == loopIvs.end()) return std::nullopt;
    return it->second;
  }
  return pids[static_cast<unsigned>(kind)];
}

std::optional<int64_t> applyBinOp(ScalarExpr::Kind kind, int64_t lhs, int64_t rhs)
{
  int64_t result = 0;
  switch (kind) {
    case ScalarExpr::Kind::Add: return checkedAdd(lhs, rhs, result) ? std::optional<int64_t>(result) : std::nullopt;
    case ScalarExpr::Kind::Sub: return checkedSub(lhs, rhs, result) ? std::optional<int64_t>(result) : std::nullopt;
    case ScalarExpr::Kind::Mul: return checkedMul(lhs, rhs, result) ? std::optional<int64_t>(result) : std::nullopt;
    case ScalarExpr::Kind::DivS:
      if (rhs == 0 || (lhs == std::numeric_limits<int64_t>::min() && rhs == -1)) return std::nullopt;
      return lhs / rhs;
    case ScalarExpr::Kind::RemS:
      if (rhs == 0 || (lhs == std::numeric_limits<int64_t>::min() && rhs == -1)) return std::nullopt;
      return lhs % rhs;
    case ScalarExpr::Kind::DivU:
      if (rhs == 0) return std::nullopt;
      return unsignedOp(lhs, rhs, [](uint64_t a, uint64_t b) { return a / b; });
    case ScalarExpr::Kind::RemU:
      if (rhs == 0) return std::nullopt;
      return unsignedOp(lhs, rhs, [](uint64_t a, uint64_t b) { return a % b; });
    case ScalarExpr::Kind::Shl: {
      if (rhs < 0 || rhs >= 64) return std::nullopt;
      unsigned shift = static_cast<unsigned>(rhs);
      return unsignedOp(lhs, [shift](uint64_t value) { return value << shift; });
    }
    case ScalarExpr::Kind::ShrS:
      if (rhs < 0 || rhs >= 64) return std::nullopt;
      return lhs >> rhs;
    case ScalarExpr::Kind::ShrU: {
      if (rhs < 0 || rhs >= 64) return std::nullopt;
      unsigned shift = static_cast<unsigned>(rhs);
      return unsignedOp(lhs, [shift](uint64_t value) { return value >> shift; });
    }
    case ScalarExpr::Kind::MinS: return std::min(lhs, rhs);
    case ScalarExpr::Kind::MaxS: return std::max(lhs, rhs);
    case ScalarExpr::Kind::MinU: return unsignedMin(lhs, rhs);
    case ScalarExpr::Kind::MaxU: return unsignedMax(lhs, rhs);
    case ScalarExpr::Kind::And: return lhs & rhs;
    case ScalarExpr::Kind::Or: return lhs | rhs;
    case ScalarExpr::Kind::Xor: return lhs ^ rhs;
  }
  return std::nullopt;
}

const char* binOpName(ScalarExpr::Kind kind)
{
  switch (kind) {
    case ScalarExpr::Kind::Add: return "+";
    case ScalarExpr::Kind::Sub: return "-";
    case ScalarExpr::Kind::Mul: return "*";
    case ScalarExpr::Kind::DivS:
    case ScalarExpr::Kind::DivU: return "/";
    case ScalarExpr::Kind::RemS:
    case ScalarExpr::Kind::RemU: return "%";
    case ScalarExpr::Kind::Shl: return "<<";
    case ScalarExpr::Kind::ShrS:
    case ScalarExpr::Kind::ShrU: return ">>";
    case ScalarExpr::Kind::MinS:
    case ScalarExpr::Kind::MinU: return "minsi";
    case ScalarExpr::Kind::MaxS:
    case ScalarExpr::Kind::MaxU: return "maxsi";
    case ScalarExpr::Kind::And: return "&";
    case ScalarExpr::Kind::Or: return "|";
    case ScalarExpr::Kind::Xor: return "^";
  }
  return "?";
}

const char* symName(SymKind kind)
{
  if (kind == SymKind::PidX) return "pid.x";
  if (kind == SymKind::PidY) return "pid.y";
  if (kind == SymKind::PidZ) return "pid.z";
  if (kind == SymKind::LoopIv) return "iv";
  return "sym";
}

void printLeaf(llvm::raw_ostream& os, const ScalarExpr& expr)
{
  if (expr.kind == ScalarExpr::Kind::Const) {
    os << expr.value;
    return;
  }
  os << symName(expr.symKind);
  if (expr.symKind == SymKind::LoopIv) os << expr.symId;
}

bool fullyKnown(const Interval& interval)
{
  return interval.hasLo && interval.hasHi;
}

bool crossesSignBoundary(const Interval& interval)
{
  return interval.lo < 0 && interval.hi > 0;
}

std::optional<Interval> signedCorners(ScalarExpr::Kind kind, const Interval& lhs, const Interval& rhs)
{
  std::array<int64_t, 2> lhsValues = {lhs.lo, lhs.hi};
  std::array<int64_t, 2> rhsValues = {rhs.lo, rhs.hi};
  std::array<int64_t, 4> results{};
  for (size_t i = 0; i < lhsValues.size(); ++i) {
    for (size_t j = 0; j < rhsValues.size(); ++j) {
      std::optional<int64_t> value = applyBinOp(kind, lhsValues[i], rhsValues[j]);
      if (!value) return std::nullopt;
      results[i * 2 + j] = *value;
    }
  }
  return Interval{
    true, true, *std::min_element(results.begin(), results.end()), *std::max_element(results.begin(), results.end())
  };
}

std::optional<Interval> unsignedCorners(ScalarExpr::Kind kind, const Interval& lhs, const Interval& rhs)
{
  if (crossesSignBoundary(lhs) || crossesSignBoundary(rhs)) return std::nullopt;
  std::array<int64_t, 2> lhsValues = {lhs.lo, lhs.hi};
  std::array<int64_t, 2> rhsValues = {rhs.lo, rhs.hi};
  std::array<int64_t, 4> results{};
  for (size_t i = 0; i < lhsValues.size(); ++i) {
    for (size_t j = 0; j < rhsValues.size(); ++j) {
      std::optional<int64_t> value = applyBinOp(kind, lhsValues[i], rhsValues[j]);
      if (!value) return std::nullopt;
      results[i * 2 + j] = *value;
    }
  }
  int64_t lo = unsignedMin(unsignedMin(results[0], results[1]), unsignedMin(results[2], results[3]));
  int64_t hi = unsignedMax(unsignedMax(results[0], results[1]), unsignedMax(results[2], results[3]));
  if (lo < 0 && hi >= 0) return std::nullopt;
  return Interval{true, true, lo, hi};
}

int64_t magnitude(int64_t value)
{
  return value < 0 ? unsignedOp(value, [](uint64_t bits) { return uint64_t(0) - bits; }) : value;
}

int64_t pow2MinusOne(int64_t value)
{
  uint64_t bits  = static_cast<uint64_t>(value);
  bits          |= bits >> 1;
  bits          |= bits >> 2;
  bits          |= bits >> 4;
  bits          |= bits >> 8;
  bits          |= bits >> 16;
  bits          |= bits >> 32;
  return static_cast<int64_t>(bits);
}

Interval evaluateDivisionSigned(const Interval& lhs, const Interval& rhs)
{
  if (!fullyKnown(lhs) || !fullyKnown(rhs) || (rhs.lo <= 0 && rhs.hi >= 0)) return Interval::unknown();
  if (auto corners = signedCorners(ScalarExpr::Kind::DivS, lhs, rhs)) return *corners;
  return Interval::unknown();
}

Interval evaluateDivisionUnsigned(const Interval& lhs, const Interval& rhs)
{
  if (!fullyKnown(lhs) || !fullyKnown(rhs) || (rhs.lo <= 0 && rhs.hi >= 0)) return Interval::unknown();
  if (auto corners = unsignedCorners(ScalarExpr::Kind::DivU, lhs, rhs)) return *corners;
  return Interval::unknown();
}

Interval evaluateRemainderSigned(const Interval& lhs, const Interval& rhs)
{
  if (!fullyKnown(rhs) || (rhs.lo <= 0 && rhs.hi >= 0)) return Interval::unknown();
  int64_t maxRemainder =
    unsignedOp(unsignedMax(magnitude(rhs.lo), magnitude(rhs.hi)), [](uint64_t value) { return value - 1; });
  int64_t minRemainder = unsignedOp(maxRemainder, [](uint64_t value) { return uint64_t(0) - value; });
  if (lhs.hasLo && lhs.lo >= 0) {
    Interval result;
    result.hasLo = true;
    result.hasHi = true;
    result.lo    = 0;
    result.hi    = lhs.hasHi ? std::min(lhs.hi, maxRemainder) : maxRemainder;
    return result;
  }
  if (lhs.hasHi && lhs.hi <= 0) {
    Interval result;
    result.hasLo = true;
    result.hasHi = true;
    result.lo    = lhs.hasLo ? std::max(lhs.lo, minRemainder) : minRemainder;
    result.hi    = 0;
    return result;
  }
  return Interval{true, true, minRemainder, maxRemainder};
}

Interval evaluateRemainderUnsigned(const Interval& lhs, const Interval& rhs)
{
  if (!fullyKnown(lhs) || !fullyKnown(rhs) || (rhs.lo <= 0 && rhs.hi >= 0)) return Interval::unknown();
  if (crossesSignBoundary(lhs) || crossesSignBoundary(rhs)) return Interval::unknown();
  int64_t maxLhs = unsignedMax(lhs.lo, lhs.hi);
  int64_t maxRhs = unsignedMax(rhs.lo, rhs.hi);
  int64_t bound  = unsignedMin(maxLhs, unsignedOp(maxRhs, [](uint64_t value) { return value - 1; }));
  if (bound < 0) return Interval::unknown();
  return Interval{true, true, 0, bound};
}

Interval evaluateShiftLeft(const Interval& lhs, const Interval& rhs)
{
  if (!rhs.isPoint() || rhs.lo < 0 || rhs.lo > 63) return Interval::unknown();
  int64_t shift = rhs.lo;
  if (lhs.isPoint()) {
    std::optional<int64_t> value = applyBinOp(ScalarExpr::Kind::Shl, lhs.lo, shift);
    return value ? Interval::point(*value) : Interval::unknown();
  }
  if (shift == 63 || !fullyKnown(lhs)) return Interval::unknown();
  int64_t factor = int64_t(1) << shift;
  int64_t lo     = 0;
  int64_t hi     = 0;
  if (!checkedMul(lhs.lo, factor, lo) || !checkedMul(lhs.hi, factor, hi)) return Interval::unknown();
  return Interval{true, true, lo, hi};
}

Interval evaluateShiftRightSigned(const Interval& lhs, const Interval& rhs)
{
  if (!fullyKnown(lhs) || !fullyKnown(rhs) || rhs.lo < 0 || rhs.hi > 63) return Interval::unknown();
  if (auto corners = signedCorners(ScalarExpr::Kind::ShrS, lhs, rhs)) return *corners;
  return Interval::unknown();
}

Interval evaluateShiftRightUnsigned(const Interval& lhs, const Interval& rhs)
{
  if (!fullyKnown(lhs) || !fullyKnown(rhs) || rhs.lo < 0 || rhs.hi > 63) return Interval::unknown();
  if (auto corners = unsignedCorners(ScalarExpr::Kind::ShrU, lhs, rhs)) return *corners;
  return Interval::unknown();
}

Interval evaluateAnd(const Interval& lhs, const Interval& rhs)
{
  if (lhs.hasLo && lhs.lo >= 0 && rhs.hasLo && rhs.lo >= 0 && lhs.hasHi && rhs.hasHi)
    return Interval{true, true, 0, std::min(lhs.hi, rhs.hi)};
  if (lhs.hasLo && lhs.lo >= 0 && lhs.hasHi) return Interval{true, true, 0, lhs.hi};
  if (rhs.hasLo && rhs.lo >= 0 && rhs.hasHi) return Interval{true, true, 0, rhs.hi};
  if (lhs.hasHi && rhs.hasHi && lhs.hi < 0 && rhs.hi < 0)
    return Interval{true, true, std::numeric_limits<int64_t>::min(), std::min(lhs.hi, rhs.hi)};
  return Interval::unknown();
}

Interval evaluateOr(const Interval& lhs, const Interval& rhs)
{
  if (lhs.hasLo && lhs.hi < 0 && rhs.hasLo && rhs.hi < 0) return Interval{true, true, std::max(lhs.lo, rhs.lo), -1};
  if (lhs.hasLo && lhs.lo >= 0 && rhs.hasLo && rhs.lo >= 0 && lhs.hasHi && rhs.hasHi)
    return Interval{true, true, std::max(lhs.lo, rhs.lo), pow2MinusOne(std::max(lhs.hi, rhs.hi))};
  return Interval::unknown();
}

Interval evaluateXor(const Interval& lhs, const Interval& rhs)
{
  if (lhs.hasLo && lhs.lo >= 0 && rhs.hasLo && rhs.lo >= 0 && lhs.hasHi && rhs.hasHi)
    return Interval{true, true, 0, pow2MinusOne(std::max(lhs.hi, rhs.hi))};
  if (lhs.hasHi && rhs.hasHi && lhs.hi < 0 && rhs.hi < 0)
    return Interval{true, true, 0, std::numeric_limits<int64_t>::max()};
  return Interval::unknown();
}

} // namespace

ExprPtr ScalarExpr::cst(int64_t value)
{
  return makeLeaf(Kind::Const, value, SymKind::PidX, 0);
}

ExprPtr ScalarExpr::sym(SymKind kind, unsigned id)
{
  return makeLeaf(Kind::Sym, 0, kind, id);
}

ExprPtr ScalarExpr::bin(Kind kind, ExprPtr lhs, ExprPtr rhs)
{
  if (!lhs || !rhs) return nullptr;
  if (lhs->isConst() && rhs->isConst()) {
    if (auto folded = applyBinOp(kind, lhs->value, rhs->value)) return cst(*folded);
  }
  auto expr  = std::make_shared<ScalarExpr>();
  expr->kind = kind;
  expr->lhs  = std::move(lhs);
  expr->rhs  = std::move(rhs);
  return expr;
}

ExprPtr ScalarExpr::add(ExprPtr lhs, ExprPtr rhs)
{
  if (!lhs) return rhs;
  if (!rhs) return lhs;
  if (lhs->isConst() && lhs->value == 0) return rhs;
  if (rhs->isConst() && rhs->value == 0) return lhs;
  return bin(Kind::Add, std::move(lhs), std::move(rhs));
}

ExprPtr ScalarExpr::sub(ExprPtr lhs, ExprPtr rhs)
{
  if (!rhs) return lhs;
  if (!lhs) return neg(std::move(rhs));
  if (rhs->isConst() && rhs->value == 0) return lhs;
  return bin(Kind::Sub, std::move(lhs), std::move(rhs));
}

ExprPtr ScalarExpr::mul(ExprPtr lhs, ExprPtr rhs)
{
  if (!lhs || !rhs) return nullptr;
  if (lhs->isConst()) {
    if (lhs->value == 0) return cst(0);
    if (lhs->value == 1) return rhs;
    if (lhs->value == -1) return neg(std::move(rhs));
  }
  if (rhs->isConst()) {
    if (rhs->value == 0) return cst(0);
    if (rhs->value == 1) return lhs;
    if (rhs->value == -1) return neg(std::move(lhs));
  }
  return bin(Kind::Mul, std::move(lhs), std::move(rhs));
}

ExprPtr ScalarExpr::neg(ExprPtr value)
{
  if (!value) return nullptr;
  return bin(Kind::Sub, cst(0), std::move(value));
}

std::optional<Interval> SymbolBounds::lookup(SymKind kind, unsigned id) const
{
  return lookupSymbol(pids, loopIvs, kind, id);
}

std::optional<int64_t> SymbolValues::lookup(SymKind kind, unsigned id) const
{
  return lookupSymbol(pids, loopIvs, kind, id);
}

std::optional<int64_t> evaluate(const ScalarExpr& expr, const SymbolValues& values)
{
  if (expr.kind == ScalarExpr::Kind::Const) return expr.value;
  if (expr.kind == ScalarExpr::Kind::Sym) return values.lookup(expr.symKind, expr.symId);

  auto lhs = evaluate(*expr.lhs, values);
  auto rhs = evaluate(*expr.rhs, values);
  if (!lhs || !rhs) return std::nullopt;
  return applyBinOp(expr.kind, *lhs, *rhs);
}

Interval evaluateInterval(const ScalarExpr& expr, const SymbolBounds& bounds)
{
  if (expr.kind == ScalarExpr::Kind::Const) return Interval::point(expr.value);
  if (expr.kind == ScalarExpr::Kind::Sym) {
    auto bound = bounds.lookup(expr.symKind, expr.symId);
    return bound ? *bound : Interval::unknown();
  }

  Interval lhs = evaluateInterval(*expr.lhs, bounds);
  Interval rhs = evaluateInterval(*expr.rhs, bounds);

  switch (expr.kind) {
    case ScalarExpr::Kind::Add: {
      Interval result;
      int64_t value = 0;
      if (lhs.hasLo && rhs.hasLo && checkedAdd(lhs.lo, rhs.lo, value)) {
        result.hasLo = true;
        result.lo    = value;
      }
      if (lhs.hasHi && rhs.hasHi && checkedAdd(lhs.hi, rhs.hi, value)) {
        result.hasHi = true;
        result.hi    = value;
      }
      return result;
    }
    case ScalarExpr::Kind::Sub: {
      Interval result;
      int64_t value = 0;
      if (lhs.hasLo && rhs.hasHi && checkedSub(lhs.lo, rhs.hi, value)) {
        result.hasLo = true;
        result.lo    = value;
      }
      if (lhs.hasHi && rhs.hasLo && checkedSub(lhs.hi, rhs.lo, value)) {
        result.hasHi = true;
        result.hi    = value;
      }
      return result;
    }
    case ScalarExpr::Kind::Mul: {
      if (!lhs.hasLo || !lhs.hasHi || !rhs.hasLo || !rhs.hasHi) return Interval::unknown();
      std::array<int64_t, 2> lhsValues = {lhs.lo, lhs.hi};
      std::array<int64_t, 2> rhsValues = {rhs.lo, rhs.hi};
      std::array<int64_t, 4> products{};
      for (size_t i = 0; i < lhsValues.size(); ++i) {
        for (size_t j = 0; j < rhsValues.size(); ++j) {
          if (!checkedMul(lhsValues[i], rhsValues[j], products[i * 2 + j])) return Interval::unknown();
        }
      }
      Interval result;
      result.hasLo = true;
      result.hasHi = true;
      result.lo    = *std::min_element(products.begin(), products.end());
      result.hi    = *std::max_element(products.begin(), products.end());
      return result;
    }
    case ScalarExpr::Kind::DivS: return evaluateDivisionSigned(lhs, rhs);
    case ScalarExpr::Kind::DivU: return evaluateDivisionUnsigned(lhs, rhs);
    case ScalarExpr::Kind::RemS: return evaluateRemainderSigned(lhs, rhs);
    case ScalarExpr::Kind::RemU: return evaluateRemainderUnsigned(lhs, rhs);
    case ScalarExpr::Kind::Shl: return evaluateShiftLeft(lhs, rhs);
    case ScalarExpr::Kind::ShrS: return evaluateShiftRightSigned(lhs, rhs);
    case ScalarExpr::Kind::ShrU: return evaluateShiftRightUnsigned(lhs, rhs);
    case ScalarExpr::Kind::MinS:
    case ScalarExpr::Kind::MinU: {
      if (expr.kind == ScalarExpr::Kind::MinU && (crossesSignBoundary(lhs) || crossesSignBoundary(rhs)))
        return Interval::unknown();
      Interval result;
      if (lhs.hasLo && rhs.hasLo) {
        result.hasLo = true;
        result.lo    = expr.kind == ScalarExpr::Kind::MinS ? std::min(lhs.lo, rhs.lo) : unsignedMin(lhs.lo, rhs.lo);
      }
      if (lhs.hasHi && rhs.hasHi) {
        result.hasHi = true;
        result.hi    = expr.kind == ScalarExpr::Kind::MinS ? std::min(lhs.hi, rhs.hi) : unsignedMin(lhs.hi, rhs.hi);
      }
      return result;
    }
    case ScalarExpr::Kind::MaxS:
    case ScalarExpr::Kind::MaxU: {
      if (expr.kind == ScalarExpr::Kind::MaxU && (crossesSignBoundary(lhs) || crossesSignBoundary(rhs)))
        return Interval::unknown();
      Interval result;
      if (lhs.hasLo && rhs.hasLo) {
        result.hasLo = true;
        result.lo    = expr.kind == ScalarExpr::Kind::MaxS ? std::max(lhs.lo, rhs.lo) : unsignedMax(lhs.lo, rhs.lo);
      }
      if (lhs.hasHi && rhs.hasHi) {
        result.hasHi = true;
        result.hi    = expr.kind == ScalarExpr::Kind::MaxS ? std::max(lhs.hi, rhs.hi) : unsignedMax(lhs.hi, rhs.hi);
      }
      return result;
    }
    case ScalarExpr::Kind::And: return evaluateAnd(lhs, rhs);
    case ScalarExpr::Kind::Or: return evaluateOr(lhs, rhs);
    case ScalarExpr::Kind::Xor: return evaluateXor(lhs, rhs);
  }
  return Interval::unknown();
}

void printExpr(llvm::raw_ostream& os, const ScalarExpr& expr)
{
  if (expr.kind == ScalarExpr::Kind::Const || expr.kind == ScalarExpr::Kind::Sym) {
    printLeaf(os, expr);
    return;
  }
  os << '(';
  printExpr(os, *expr.lhs);
  os << ' ' << binOpName(expr.kind) << ' ';
  printExpr(os, *expr.rhs);
  os << ')';
}

void printExpr(llvm::raw_ostream& os, const ExprPtr& expr)
{
  if (!expr) {
    os << '0';
    return;
  }
  printExpr(os, *expr);
}

} // namespace triton
} // namespace mlir
