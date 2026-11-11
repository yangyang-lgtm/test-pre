#ifndef PRIVATEUSE_ANALYSIS_VALUERANGE_H
#define PRIVATEUSE_ANALYSIS_VALUERANGE_H

#include "llvm/ADT/DenseMap.h"
#include "llvm/Support/raw_ostream.h"

#include <array>
#include <cstdint>
#include <memory>
#include <optional>

namespace mlir {
namespace triton {

enum class SymKind : uint8_t { PidX, PidY, PidZ, LoopIv };

struct ScalarExpr;
using ExprPtr = std::shared_ptr<const ScalarExpr>;

struct ScalarExpr
{
  enum class Kind : uint8_t {
    Const,
    Sym,
    Add,
    Sub,
    Mul,
    DivS,
    DivU,
    RemS,
    RemU,
    Shl,
    ShrS,
    ShrU,
    MinS,
    MaxS,
    MinU,
    MaxU,
    And,
    Or,
    Xor,
  };

  Kind kind       = Kind::Const;
  int64_t value   = 0;
  SymKind symKind = SymKind::PidX;
  unsigned symId  = 0;
  ExprPtr lhs;
  ExprPtr rhs;

  static ExprPtr cst(int64_t value);
  static ExprPtr sym(SymKind kind, unsigned id = 0);
  static ExprPtr bin(Kind kind, ExprPtr lhs, ExprPtr rhs);
  static ExprPtr add(ExprPtr lhs, ExprPtr rhs);
  static ExprPtr sub(ExprPtr lhs, ExprPtr rhs);
  static ExprPtr mul(ExprPtr lhs, ExprPtr rhs);
  static ExprPtr neg(ExprPtr value);

  bool isConst() const { return kind == Kind::Const; }
};

struct SymbolValues
{
  std::array<std::optional<int64_t>, 3> pids;
  llvm::DenseMap<unsigned, int64_t> loopIvs;

  std::optional<int64_t> lookup(SymKind kind, unsigned id) const;
};

struct Interval
{
  bool hasLo = false;
  bool hasHi = false;
  int64_t lo = 0;
  int64_t hi = 0;

  static Interval point(int64_t value) { return {true, true, value, value}; }
  static Interval unknown() { return {}; }
  bool isPoint() const { return hasLo && hasHi && lo == hi; }
};

struct SymbolBounds
{
  std::array<std::optional<Interval>, 3> pids;
  llvm::DenseMap<unsigned, Interval> loopIvs;

  std::optional<Interval> lookup(SymKind kind, unsigned id) const;
};

std::optional<int64_t> evaluate(const ScalarExpr& expr, const SymbolValues& values);
Interval evaluateInterval(const ScalarExpr& expr, const SymbolBounds& bounds);

void printExpr(llvm::raw_ostream& os, const ScalarExpr& expr);
void printExpr(llvm::raw_ostream& os, const ExprPtr& expr);

} // namespace triton
} // namespace mlir

#endif
