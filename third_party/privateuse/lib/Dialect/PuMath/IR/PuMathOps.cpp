#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/CommonFolders.h"
#include "privateuse/Dialect/PuMath/IR/PuMathDialect.h"
#include "mlir/Dialect/UB/IR/UBOps.h"
#include "mlir/IR/Builders.h"
#include <optional>

using namespace mlir;
using namespace mlir::pumath;

#define GET_OP_CLASSES
#include "privateuse/Dialect/PuMath/IR/PuMathOps.cpp.inc"

OpFoldResult pumath::AbsFOp::fold(FoldAdaptor adaptor)
{
  return constFoldUnaryOp<FloatAttr>(adaptor.getOperands(), [](const APFloat& a) { return abs(a); });
}

OpFoldResult pumath::AbsIOp::fold(FoldAdaptor adaptor)
{
  return constFoldUnaryOp<IntegerAttr>(adaptor.getOperands(), [](const APInt& a) { return a.abs(); });
}

OpFoldResult pumath::AcosOp::fold(FoldAdaptor adaptor)
{
  return constFoldUnaryOpConditional<FloatAttr>(adaptor.getOperands(), [](const APFloat& a) -> std::optional<APFloat> {
    switch (APFloat::SemanticsToEnum(a.getSemantics())) {
      case APFloat::Semantics::S_IEEEdouble: return APFloat(acos(a.convertToDouble()));
      case APFloat::Semantics::S_IEEEsingle: return APFloat(acosf(a.convertToFloat()));
      default: return {};
    }
  });
}

OpFoldResult pumath::AcoshOp::fold(FoldAdaptor adaptor)
{
  return constFoldUnaryOpConditional<FloatAttr>(adaptor.getOperands(), [](const APFloat& a) -> std::optional<APFloat> {
    switch (APFloat::SemanticsToEnum(a.getSemantics())) {
      case APFloat::Semantics::S_IEEEdouble: return APFloat(acosh(a.convertToDouble()));
      case APFloat::Semantics::S_IEEEsingle: return APFloat(acoshf(a.convertToFloat()));
      default: return {};
    }
  });
}

OpFoldResult pumath::AsinOp::fold(FoldAdaptor adaptor)
{
  return constFoldUnaryOpConditional<FloatAttr>(adaptor.getOperands(), [](const APFloat& a) -> std::optional<APFloat> {
    switch (APFloat::SemanticsToEnum(a.getSemantics())) {
      case APFloat::Semantics::S_IEEEdouble: return APFloat(asin(a.convertToDouble()));
      case APFloat::Semantics::S_IEEEsingle: return APFloat(asinf(a.convertToFloat()));
      default: return {};
    }
  });
}

OpFoldResult pumath::AsinhOp::fold(FoldAdaptor adaptor)
{
  return constFoldUnaryOpConditional<FloatAttr>(adaptor.getOperands(), [](const APFloat& a) -> std::optional<APFloat> {
    switch (APFloat::SemanticsToEnum(a.getSemantics())) {
      case APFloat::Semantics::S_IEEEdouble: return APFloat(asinh(a.convertToDouble()));
      case APFloat::Semantics::S_IEEEsingle: return APFloat(asinhf(a.convertToFloat()));
      default: return {};
    }
  });
}

OpFoldResult pumath::AtanOp::fold(FoldAdaptor adaptor)
{
  return constFoldUnaryOpConditional<FloatAttr>(adaptor.getOperands(), [](const APFloat& a) -> std::optional<APFloat> {
    switch (APFloat::SemanticsToEnum(a.getSemantics())) {
      case APFloat::Semantics::S_IEEEdouble: return APFloat(atan(a.convertToDouble()));
      case APFloat::Semantics::S_IEEEsingle: return APFloat(atanf(a.convertToFloat()));
      default: return {};
    }
  });
}

OpFoldResult pumath::AtanhOp::fold(FoldAdaptor adaptor)
{
  return constFoldUnaryOpConditional<FloatAttr>(adaptor.getOperands(), [](const APFloat& a) -> std::optional<APFloat> {
    switch (APFloat::SemanticsToEnum(a.getSemantics())) {
      case APFloat::Semantics::S_IEEEdouble: return APFloat(atanh(a.convertToDouble()));
      case APFloat::Semantics::S_IEEEsingle: return APFloat(atanhf(a.convertToFloat()));
      default: return {};
    }
  });
}

OpFoldResult pumath::Atan2Op::fold(FoldAdaptor adaptor)
{
  return constFoldBinaryOpConditional<FloatAttr>(
    adaptor.getOperands(),
    [](const APFloat& a, const APFloat& b) -> std::optional<APFloat> {
      if (a.isZero() && b.isZero()) return llvm::APFloat::getNaN(a.getSemantics());

      switch (APFloat::SemanticsToEnum(a.getSemantics())) {
        case APFloat::Semantics::S_IEEEdouble: return APFloat(atan2(a.convertToDouble(), b.convertToDouble()));
        case APFloat::Semantics::S_IEEEsingle: return APFloat(atan2f(a.convertToFloat(), b.convertToFloat()));
        default: return {};
      }
    }
  );
}

OpFoldResult pumath::CbrtOp::fold(FoldAdaptor adaptor)
{
  return constFoldUnaryOpConditional<FloatAttr>(adaptor.getOperands(), [](const APFloat& a) -> std::optional<APFloat> {
    switch (APFloat::SemanticsToEnum(a.getSemantics())) {
      case APFloat::Semantics::S_IEEEdouble: return APFloat(cbrt(a.convertToDouble()));
      case APFloat::Semantics::S_IEEEsingle: return APFloat(cbrtf(a.convertToFloat()));
      default: return {};
    }
  });
}

OpFoldResult pumath::CeilOp::fold(FoldAdaptor adaptor)
{
  return constFoldUnaryOp<FloatAttr>(adaptor.getOperands(), [](const APFloat& a) {
    APFloat result(a);
    result.roundToIntegral(llvm::RoundingMode::TowardPositive);
    return result;
  });
}

OpFoldResult pumath::CopySignOp::fold(FoldAdaptor adaptor)
{
  return constFoldBinaryOp<FloatAttr>(adaptor.getOperands(), [](const APFloat& a, const APFloat& b) {
    APFloat result(a);
    result.copySign(b);
    return result;
  });
}

OpFoldResult pumath::CosOp::fold(FoldAdaptor adaptor)
{
  return constFoldUnaryOpConditional<FloatAttr>(adaptor.getOperands(), [](const APFloat& a) -> std::optional<APFloat> {
    switch (APFloat::SemanticsToEnum(a.getSemantics())) {
      case APFloat::Semantics::S_IEEEdouble: return APFloat(cos(a.convertToDouble()));
      case APFloat::Semantics::S_IEEEsingle: return APFloat(cosf(a.convertToFloat()));
      default: return {};
    }
  });
}

OpFoldResult pumath::CoshOp::fold(FoldAdaptor adaptor)
{
  return constFoldUnaryOpConditional<FloatAttr>(adaptor.getOperands(), [](const APFloat& a) -> std::optional<APFloat> {
    switch (APFloat::SemanticsToEnum(a.getSemantics())) {
      case APFloat::Semantics::S_IEEEdouble: return APFloat(cosh(a.convertToDouble()));
      case APFloat::Semantics::S_IEEEsingle: return APFloat(coshf(a.convertToFloat()));
      default: return {};
    }
  });
}

OpFoldResult pumath::SinOp::fold(FoldAdaptor adaptor)
{
  return constFoldUnaryOpConditional<FloatAttr>(adaptor.getOperands(), [](const APFloat& a) -> std::optional<APFloat> {
    switch (APFloat::SemanticsToEnum(a.getSemantics())) {
      case APFloat::Semantics::S_IEEEdouble: return APFloat(sin(a.convertToDouble()));
      case APFloat::Semantics::S_IEEEsingle: return APFloat(sinf(a.convertToFloat()));
      default: return {};
    }
  });
}

OpFoldResult pumath::SinhOp::fold(FoldAdaptor adaptor)
{
  return constFoldUnaryOpConditional<FloatAttr>(adaptor.getOperands(), [](const APFloat& a) -> std::optional<APFloat> {
    switch (APFloat::SemanticsToEnum(a.getSemantics())) {
      case APFloat::Semantics::S_IEEEdouble: return APFloat(sinh(a.convertToDouble()));
      case APFloat::Semantics::S_IEEEsingle: return APFloat(sinhf(a.convertToFloat()));
      default: return {};
    }
  });
}

std::optional<SmallVector<int64_t, 4>> pumath::SincosOp::getShapeForUnroll()
{
  if (auto vt = mlir::dyn_cast<VectorType>(getOperand().getType())) return llvm::to_vector<4>(vt.getShape());
  return std::nullopt;
}

LogicalResult pumath::SincosOp::fold(FoldAdaptor adaptor, SmallVectorImpl<OpFoldResult>& result)
{
  auto foldSincos =
    [](const APFloat& a, double (*fnDouble)(double), float (*fnFloat)(float)) -> std::optional<APFloat> {
    switch (APFloat::SemanticsToEnum(a.getSemantics())) {
      case APFloat::Semantics::S_IEEEdouble: return APFloat(fnDouble(a.convertToDouble()));
      case APFloat::Semantics::S_IEEEsingle: return APFloat(fnFloat(a.convertToFloat()));
      default: return {};
    }
  };

  Attribute sinRes = constFoldUnaryOpConditional<FloatAttr>(adaptor.getOperands(), [&](const APFloat& a) {
    return foldSincos(a, sin, sinf);
  });
  Attribute cosRes = constFoldUnaryOpConditional<FloatAttr>(adaptor.getOperands(), [&](const APFloat& a) {
    return foldSincos(a, cos, cosf);
  });

  if (sinRes && cosRes) {
    result.push_back(sinRes);
    result.push_back(cosRes);
    return success();
  }
  return failure();
}

OpFoldResult pumath::CountLeadingZerosOp::fold(FoldAdaptor adaptor)
{
  return constFoldUnaryOp<IntegerAttr>(adaptor.getOperands(), [](const APInt& a) {
    return APInt(a.getBitWidth(), a.countl_zero());
  });
}

OpFoldResult pumath::CountTrailingZerosOp::fold(FoldAdaptor adaptor)
{
  return constFoldUnaryOp<IntegerAttr>(adaptor.getOperands(), [](const APInt& a) {
    return APInt(a.getBitWidth(), a.countr_zero());
  });
}

OpFoldResult pumath::CtPopOp::fold(FoldAdaptor adaptor)
{
  return constFoldUnaryOp<IntegerAttr>(adaptor.getOperands(), [](const APInt& a) {
    return APInt(a.getBitWidth(), a.popcount());
  });
}

OpFoldResult pumath::ErfOp::fold(FoldAdaptor adaptor)
{
  return constFoldUnaryOpConditional<FloatAttr>(adaptor.getOperands(), [](const APFloat& a) -> std::optional<APFloat> {
    switch (APFloat::SemanticsToEnum(a.getSemantics())) {
      case APFloat::Semantics::S_IEEEdouble: return APFloat(erf(a.convertToDouble()));
      case APFloat::Semantics::S_IEEEsingle: return APFloat(erff(a.convertToFloat()));
      default: return {};
    }
  });
}

OpFoldResult pumath::ErfcOp::fold(FoldAdaptor adaptor)
{
  return constFoldUnaryOpConditional<FloatAttr>(adaptor.getOperands(), [](const APFloat& a) -> std::optional<APFloat> {
    switch (APFloat::SemanticsToEnum(a.getSemantics())) {
      case APFloat::Semantics::S_IEEEdouble: return APFloat(erfc(a.convertToDouble()));
      case APFloat::Semantics::S_IEEEsingle: return APFloat(erfcf(a.convertToFloat()));
      default: return {};
    }
  });
}

OpFoldResult pumath::IPowIOp::fold(FoldAdaptor adaptor)
{
  return constFoldBinaryOpConditional<IntegerAttr>(
    adaptor.getOperands(),
    [](const APInt& base, const APInt& power) -> std::optional<APInt> {
      unsigned width = base.getBitWidth();
      auto zeroValue = APInt::getZero(width);
      if (width == 1) return {};
      APInt oneValue{width, 1ULL, true};
      APInt minusOneValue{width, -1ULL, true};

      if (power.isZero()) return oneValue;

      if (power.isNegative()) {
        if (base.isZero()) return {};
        if (base.isOne()) return oneValue;
        if (base.ne(minusOneValue)) return zeroValue;
        if (power[0] == 1) return minusOneValue;

        return oneValue;
      }

      APInt result   = oneValue;
      APInt curBase  = base;
      APInt curPower = power;
      while (true) {
        if (curPower[0] == 1) result *= curBase;
        curPower.lshrInPlace(1);
        if (curPower.isZero()) return result;
        curBase *= curBase;
      }
    }
  );

  return Attribute();
}

OpFoldResult pumath::LogOp::fold(FoldAdaptor adaptor)
{
  return constFoldUnaryOpConditional<FloatAttr>(adaptor.getOperands(), [](const APFloat& a) -> std::optional<APFloat> {
    if (a.isNegative()) return {};

    switch (APFloat::SemanticsToEnum(a.getSemantics())) {
      case APFloat::Semantics::S_IEEEdouble: return APFloat(log(a.convertToDouble()));
      case APFloat::Semantics::S_IEEEsingle: return APFloat(logf(a.convertToFloat()));
      default: return {};
    }
  });
}

OpFoldResult pumath::Log2Op::fold(FoldAdaptor adaptor)
{
  return constFoldUnaryOpConditional<FloatAttr>(adaptor.getOperands(), [](const APFloat& a) -> std::optional<APFloat> {
    if (a.isNegative()) return {};

    switch (APFloat::SemanticsToEnum(a.getSemantics())) {
      case APFloat::Semantics::S_IEEEdouble: return APFloat(log2(a.convertToDouble()));
      case APFloat::Semantics::S_IEEEsingle: return APFloat(log2f(a.convertToFloat()));
      default: return {};
    }
  });
}

OpFoldResult pumath::Log10Op::fold(FoldAdaptor adaptor)
{
  return constFoldUnaryOpConditional<FloatAttr>(adaptor.getOperands(), [](const APFloat& a) -> std::optional<APFloat> {
    if (a.isNegative()) return {};

    switch (APFloat::SemanticsToEnum(a.getSemantics())) {
      case APFloat::Semantics::S_IEEEdouble: return APFloat(log10(a.convertToDouble()));
      case APFloat::Semantics::S_IEEEsingle: return APFloat(log10f(a.convertToFloat()));
      default: return {};
    }
  });
}

OpFoldResult pumath::Log1pOp::fold(FoldAdaptor adaptor)
{
  return constFoldUnaryOpConditional<FloatAttr>(adaptor.getOperands(), [](const APFloat& a) -> std::optional<APFloat> {
    switch (APFloat::SemanticsToEnum(a.getSemantics())) {
      case APFloat::Semantics::S_IEEEdouble:
        if ((a + APFloat(1.0)).isNegative()) return {};
        return APFloat(log1p(a.convertToDouble()));
      case APFloat::Semantics::S_IEEEsingle:
        if ((a + APFloat(1.0f)).isNegative()) return {};
        return APFloat(log1pf(a.convertToFloat()));
      default: return {};
    }
  });
}

OpFoldResult pumath::PowFOp::fold(FoldAdaptor adaptor)
{
  return constFoldBinaryOpConditional<FloatAttr>(
    adaptor.getOperands(),
    [](const APFloat& a, const APFloat& b) -> std::optional<APFloat> {
      switch (APFloat::SemanticsToEnum(a.getSemantics())) {
        case APFloat::Semantics::S_IEEEdouble: return APFloat(pow(a.convertToDouble(), b.convertToDouble()));
        case APFloat::Semantics::S_IEEEsingle: return APFloat(powf(a.convertToFloat(), b.convertToFloat()));
        default: return {};
      }
    }
  );
}

OpFoldResult pumath::RsqrtOp::fold(FoldAdaptor adaptor)
{
  return constFoldUnaryOpConditional<FloatAttr>(adaptor.getOperands(), [](const APFloat& a) -> std::optional<APFloat> {
    if (a.isNegative()) return {};

    APFloat one(a.getSemantics(), 1);
    switch (APFloat::SemanticsToEnum(a.getSemantics())) {
      case APFloat::Semantics::S_IEEEdouble: return one / APFloat(sqrt(a.convertToDouble()));
      case APFloat::Semantics::S_IEEEsingle: return one / APFloat(sqrtf(a.convertToFloat()));
      default: return {};
    }
  });
}

OpFoldResult pumath::SqrtOp::fold(FoldAdaptor adaptor)
{
  return constFoldUnaryOpConditional<FloatAttr>(adaptor.getOperands(), [](const APFloat& a) -> std::optional<APFloat> {
    if (a.isNegative()) return {};

    switch (APFloat::SemanticsToEnum(a.getSemantics())) {
      case APFloat::Semantics::S_IEEEdouble: return APFloat(sqrt(a.convertToDouble()));
      case APFloat::Semantics::S_IEEEsingle: return APFloat(sqrtf(a.convertToFloat()));
      default: return {};
    }
  });
}

OpFoldResult pumath::ExpOp::fold(FoldAdaptor adaptor)
{
  return constFoldUnaryOpConditional<FloatAttr>(adaptor.getOperands(), [](const APFloat& a) -> std::optional<APFloat> {
    switch (APFloat::SemanticsToEnum(a.getSemantics())) {
      case APFloat::Semantics::S_IEEEdouble: return APFloat(exp(a.convertToDouble()));
      case APFloat::Semantics::S_IEEEsingle: return APFloat(expf(a.convertToFloat()));
      default: return {};
    }
  });
}

OpFoldResult pumath::Exp2Op::fold(FoldAdaptor adaptor)
{
  return constFoldUnaryOpConditional<FloatAttr>(adaptor.getOperands(), [](const APFloat& a) -> std::optional<APFloat> {
    switch (APFloat::SemanticsToEnum(a.getSemantics())) {
      case APFloat::Semantics::S_IEEEdouble: return APFloat(exp2(a.convertToDouble()));
      case APFloat::Semantics::S_IEEEsingle: return APFloat(exp2f(a.convertToFloat()));
      default: return {};
    }
  });
}

OpFoldResult pumath::ExpM1Op::fold(FoldAdaptor adaptor)
{
  return constFoldUnaryOpConditional<FloatAttr>(adaptor.getOperands(), [](const APFloat& a) -> std::optional<APFloat> {
    switch (APFloat::SemanticsToEnum(a.getSemantics())) {
      case APFloat::Semantics::S_IEEEdouble: return APFloat(expm1(a.convertToDouble()));
      case APFloat::Semantics::S_IEEEsingle: return APFloat(expm1f(a.convertToFloat()));
      default: return {};
    }
  });
}

OpFoldResult pumath::IsFiniteOp::fold(FoldAdaptor adaptor)
{
  if (auto val = dyn_cast_or_null<FloatAttr>(adaptor.getOperand())) {
    return BoolAttr::get(val.getContext(), val.getValue().isFinite());
  }
  if (auto splat = dyn_cast_or_null<SplatElementsAttr>(adaptor.getOperand())) {
    return DenseElementsAttr::get(cast<ShapedType>(getType()), APInt(1, splat.getSplatValue<APFloat>().isFinite()));
  }
  return {};
}

OpFoldResult pumath::IsInfOp::fold(FoldAdaptor adaptor)
{
  if (auto val = dyn_cast_or_null<FloatAttr>(adaptor.getOperand())) {
    return BoolAttr::get(val.getContext(), val.getValue().isInfinity());
  }
  if (auto splat = dyn_cast_or_null<SplatElementsAttr>(adaptor.getOperand())) {
    return DenseElementsAttr::get(cast<ShapedType>(getType()), APInt(1, splat.getSplatValue<APFloat>().isInfinity()));
  }
  return {};
}

OpFoldResult pumath::IsNaNOp::fold(FoldAdaptor adaptor)
{
  if (auto val = dyn_cast_or_null<FloatAttr>(adaptor.getOperand())) {
    return BoolAttr::get(val.getContext(), val.getValue().isNaN());
  }
  if (auto splat = dyn_cast_or_null<SplatElementsAttr>(adaptor.getOperand())) {
    return DenseElementsAttr::get(cast<ShapedType>(getType()), APInt(1, splat.getSplatValue<APFloat>().isNaN()));
  }
  return {};
}

OpFoldResult pumath::IsNormalOp::fold(FoldAdaptor adaptor)
{
  if (auto val = dyn_cast_or_null<FloatAttr>(adaptor.getOperand())) {
    return BoolAttr::get(val.getContext(), val.getValue().isNormal());
  }
  if (auto splat = dyn_cast_or_null<SplatElementsAttr>(adaptor.getOperand())) {
    return DenseElementsAttr::get(cast<ShapedType>(getType()), APInt(1, splat.getSplatValue<APFloat>().isNormal()));
  }
  return {};
}

OpFoldResult pumath::TanOp::fold(FoldAdaptor adaptor)
{
  return constFoldUnaryOpConditional<FloatAttr>(adaptor.getOperands(), [](const APFloat& a) -> std::optional<APFloat> {
    switch (APFloat::SemanticsToEnum(a.getSemantics())) {
      case APFloat::Semantics::S_IEEEdouble: return APFloat(tan(a.convertToDouble()));
      case APFloat::Semantics::S_IEEEsingle: return APFloat(tanf(a.convertToFloat()));
      default: return {};
    }
  });
}

OpFoldResult pumath::TanhOp::fold(FoldAdaptor adaptor)
{
  return constFoldUnaryOpConditional<FloatAttr>(adaptor.getOperands(), [](const APFloat& a) -> std::optional<APFloat> {
    switch (APFloat::SemanticsToEnum(a.getSemantics())) {
      case APFloat::Semantics::S_IEEEdouble: return APFloat(tanh(a.convertToDouble()));
      case APFloat::Semantics::S_IEEEsingle: return APFloat(tanhf(a.convertToFloat()));
      default: return {};
    }
  });
}

OpFoldResult pumath::RoundEvenOp::fold(FoldAdaptor adaptor)
{
  return constFoldUnaryOp<FloatAttr>(adaptor.getOperands(), [](const APFloat& a) {
    APFloat result(a);
    result.roundToIntegral(llvm::RoundingMode::NearestTiesToEven);
    return result;
  });
}

OpFoldResult pumath::FloorOp::fold(FoldAdaptor adaptor)
{
  return constFoldUnaryOp<FloatAttr>(adaptor.getOperands(), [](const APFloat& a) {
    APFloat result(a);
    result.roundToIntegral(llvm::RoundingMode::TowardNegative);
    return result;
  });
}

OpFoldResult pumath::RoundOp::fold(FoldAdaptor adaptor)
{
  return constFoldUnaryOpConditional<FloatAttr>(adaptor.getOperands(), [](const APFloat& a) -> std::optional<APFloat> {
    switch (APFloat::SemanticsToEnum(a.getSemantics())) {
      case APFloat::Semantics::S_IEEEdouble: return APFloat(round(a.convertToDouble()));
      case APFloat::Semantics::S_IEEEsingle: return APFloat(roundf(a.convertToFloat()));
      default: return {};
    }
  });
}

OpFoldResult pumath::TruncOp::fold(FoldAdaptor adaptor)
{
  return constFoldUnaryOpConditional<FloatAttr>(adaptor.getOperands(), [](const APFloat& a) -> std::optional<APFloat> {
    switch (APFloat::SemanticsToEnum(a.getSemantics())) {
      case APFloat::Semantics::S_IEEEdouble: return APFloat(trunc(a.convertToDouble()));
      case APFloat::Semantics::S_IEEEsingle: return APFloat(truncf(a.convertToFloat()));
      default: return {};
    }
  });
}

OpFoldResult pumath::FPowIOp::fold(FoldAdaptor adaptor)
{
  return constFoldBinaryOpConditional<FloatAttr, IntegerAttr>(
    adaptor.getOperands(),
    [](const APFloat& base, const APInt& exp) -> std::optional<APFloat> {
      switch (APFloat::SemanticsToEnum(base.getSemantics())) {
        case APFloat::Semantics::S_IEEEdouble:
        case APFloat::Semantics::S_IEEEsingle: break;
        default: return {};
      }

      const llvm::fltSemantics& sem = base.getSemantics();
      APInt magnitude               = exp.abs();
      APFloat res                   = APFloat::getOne(sem);
      APFloat curSquare             = base;
      while (!magnitude.isZero()) {
        if (magnitude[0]) res = res * curSquare;
        curSquare = curSquare * curSquare;
        magnitude.lshrInPlace(1);
      }
      if (exp.isNegative()) res = APFloat::getOne(sem) / res;
      return res;
    }
  );
}

Operation* pumath::PuMathDialect::materializeConstant(OpBuilder& builder, Attribute value, Type type, Location loc)
{
  if (auto poison = dyn_cast<ub::PoisonAttr>(value)) return ub::PoisonOp::create(builder, loc, type, poison);

  return puarith::ConstantOp::materialize(builder, value, type, loc);
}
