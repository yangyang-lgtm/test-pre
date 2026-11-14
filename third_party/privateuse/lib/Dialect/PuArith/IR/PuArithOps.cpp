#include <cassert>
#include <cstdint>
#include <functional>
#include <optional>
#include <utility>

#include "privateuse/Dialect/PuArith/IR/PuArithDialect.h"
#include "mlir/Dialect/CommonFolders.h"
#include "mlir/Dialect/UB/IR/UBMatchers.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributeInterfaces.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Support/LogicalResult.h"

#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/APSInt.h"
#include "llvm/ADT/FloatingPointMode.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/ErrorHandling.h"

using namespace mlir;
using namespace mlir::puarith;

static constexpr llvm::RoundingMode kDefaultRoundingMode = llvm::RoundingMode::NearestTiesToEven;

static IntegerAttr applyToIntegerAttrs(
  PatternRewriter& builder, Value res, Attribute lhs, Attribute rhs,
  function_ref<APInt(const APInt&, const APInt&)> binFn
)
{
  const APInt& lhsVal = llvm::cast<IntegerAttr>(lhs).getValue();
  const APInt& rhsVal = llvm::cast<IntegerAttr>(rhs).getValue();
  APInt value         = binFn(lhsVal, rhsVal);
  return IntegerAttr::get(res.getType(), value);
}

static IntegerAttr addIntegerAttrs(PatternRewriter& builder, Value res, Attribute lhs, Attribute rhs)
{
  return applyToIntegerAttrs(builder, res, lhs, rhs, std::plus<APInt>());
}

static IntegerAttr subIntegerAttrs(PatternRewriter& builder, Value res, Attribute lhs, Attribute rhs)
{
  return applyToIntegerAttrs(builder, res, lhs, rhs, std::minus<APInt>());
}

static IntegerAttr mulIntegerAttrs(PatternRewriter& builder, Value res, Attribute lhs, Attribute rhs)
{
  return applyToIntegerAttrs(builder, res, lhs, rhs, std::multiplies<APInt>());
}

static IntegerAttr andIntegerAttrs(PatternRewriter& builder, Value res, Attribute lhs, Attribute rhs)
{
  return applyToIntegerAttrs(builder, res, lhs, rhs, std::bit_and<APInt>());
}

static IntegerAttr orIntegerAttrs(PatternRewriter& builder, Value res, Attribute lhs, Attribute rhs)
{
  return applyToIntegerAttrs(builder, res, lhs, rhs, std::bit_or<APInt>());
}

static IntegerAttr xorIntegerAttrs(PatternRewriter& builder, Value res, Attribute lhs, Attribute rhs)
{
  return applyToIntegerAttrs(builder, res, lhs, rhs, std::bit_xor<APInt>());
}

static IntegerOverflowFlagsAttr mergeOverflowFlags(IntegerOverflowFlagsAttr val1, IntegerOverflowFlagsAttr val2)
{
  return IntegerOverflowFlagsAttr::get(val1.getContext(), val1.getValue() & val2.getValue());
}

puarith::CmpIPredicate puarith::invertPredicate(puarith::CmpIPredicate pred)
{
  switch (pred) {
    case puarith::CmpIPredicate::eq: return puarith::CmpIPredicate::ne;
    case puarith::CmpIPredicate::ne: return puarith::CmpIPredicate::eq;
    case puarith::CmpIPredicate::slt: return puarith::CmpIPredicate::sge;
    case puarith::CmpIPredicate::sle: return puarith::CmpIPredicate::sgt;
    case puarith::CmpIPredicate::sgt: return puarith::CmpIPredicate::sle;
    case puarith::CmpIPredicate::sge: return puarith::CmpIPredicate::slt;
    case puarith::CmpIPredicate::ult: return puarith::CmpIPredicate::uge;
    case puarith::CmpIPredicate::ule: return puarith::CmpIPredicate::ugt;
    case puarith::CmpIPredicate::ugt: return puarith::CmpIPredicate::ule;
    case puarith::CmpIPredicate::uge: return puarith::CmpIPredicate::ult;
  }
  llvm_unreachable("unknown cmpi predicate kind");
}

static llvm::RoundingMode convertArithRoundingModeToLLVMIR(std::optional<RoundingMode> roundingMode)
{
  if (!roundingMode) return kDefaultRoundingMode;
  switch (*roundingMode) {
    case RoundingMode::downward: return llvm::RoundingMode::TowardNegative;
    case RoundingMode::to_nearest_away: return llvm::RoundingMode::NearestTiesToAway;
    case RoundingMode::to_nearest_even: return llvm::RoundingMode::NearestTiesToEven;
    case RoundingMode::toward_zero: return llvm::RoundingMode::TowardZero;
    case RoundingMode::upward: return llvm::RoundingMode::TowardPositive;
  }
  llvm_unreachable("Unhandled rounding mode");
}

static puarith::CmpIPredicateAttr invertPredicate(puarith::CmpIPredicateAttr pred)
{
  return puarith::CmpIPredicateAttr::get(pred.getContext(), invertPredicate(pred.getValue()));
}

static int64_t getScalarOrElementWidth(Type type)
{
  Type elemTy = getElementTypeOrSelf(type);
  if (elemTy.isIntOrFloat()) return elemTy.getIntOrFloatBitWidth();

  return -1;
}

static int64_t getScalarOrElementWidth(Value value)
{
  return getScalarOrElementWidth(value.getType());
}

static FailureOr<APInt> getIntOrSplatIntValue(Attribute attr)
{
  APInt value;
  if (matchPattern(attr, m_ConstantInt(&value))) return value;

  return failure();
}

static Attribute getBoolAttribute(Type type, bool value)
{
  auto boolAttr         = BoolAttr::get(type.getContext(), value);
  ShapedType shapedType = dyn_cast_or_null<ShapedType>(type);
  if (!shapedType) return boolAttr;
  if (!shapedType.hasStaticShape()) return {};
  return DenseElementsAttr::get(shapedType, boolAttr);
}

static Attribute getIntegerAttrOfType(Type type, int64_t value)
{
  auto scalarAttr       = IntegerAttr::get(getElementTypeOrSelf(type), value);
  ShapedType shapedType = dyn_cast<ShapedType>(type);
  if (!shapedType) return scalarAttr;
  if (!shapedType.hasStaticShape()) return {};
  return DenseElementsAttr::get(shapedType, scalarAttr);
}

namespace {
#include "privateuse/Dialect/PuArith/IR/PuArithCanonicalization.h.inc"
}

void puarith::ConstantOp::getAsmResultNames(function_ref<void(Value, StringRef)> setNameFn)
{
  auto type = getType();
  if (auto intCst = dyn_cast<IntegerAttr>(getValue())) {
    auto intType = dyn_cast<IntegerType>(type);

    if (intType && intType.getWidth() == 1) return setNameFn(getResult(), (intCst.getInt() ? "true" : "false"));

    SmallString<32> specialNameBuffer;
    llvm::raw_svector_ostream specialName(specialNameBuffer);
    specialName << 'c' << intCst.getValue();
    if (intType) specialName << '_' << type;
    setNameFn(getResult(), specialName.str());
  } else {
    setNameFn(getResult(), "cst");
  }
}

LogicalResult puarith::ConstantOp::verify()
{
  auto type = getType();
  if (auto intType = dyn_cast<IntegerType>(getElementTypeOrSelf(type)); intType && !intType.isSignless())
    return emitOpError("integer return type must be signless");
  if (!llvm::isa<IntegerAttr, FloatAttr, ElementsAttr>(getValue())) {
    return emitOpError("value must be an integer, float, or elements attribute");
  }

  if (isa<ScalableVectorType>(type) && !isa<SplatElementsAttr>(getValue()))
    return emitOpError("initializing scalable vectors with elements attribute is not supported"
                       " unless it's a vector splat");
  return success();
}

bool puarith::ConstantOp::isBuildableWith(Attribute value, Type type)
{
  auto typedAttr = dyn_cast<TypedAttr>(value);
  if (!typedAttr || typedAttr.getType() != type) return false;
  if (auto intType = dyn_cast<IntegerType>(getElementTypeOrSelf(type))) {
    if (!intType.isSignless()) return false;
  }
  return llvm::isa<IntegerAttr, FloatAttr, ElementsAttr>(value);
}

ConstantOp puarith::ConstantOp::materialize(OpBuilder& builder, Attribute value, Type type, Location loc)
{
  if (isBuildableWith(value, type)) return puarith::ConstantOp::create(builder, loc, cast<TypedAttr>(value));
  return nullptr;
}

OpFoldResult puarith::ConstantOp::fold(FoldAdaptor adaptor)
{
  return getValue();
}

void puarith::ConstantIntOp::build(OpBuilder& builder, OperationState& result, int64_t value, unsigned width)
{
  auto type = builder.getIntegerType(width);
  puarith::ConstantOp::build(builder, result, type, builder.getIntegerAttr(type, value));
}

puarith::ConstantIntOp
puarith::ConstantIntOp::create(OpBuilder& builder, Location location, int64_t value, unsigned width)
{
  mlir::OperationState state(location, getOperationName());
  build(builder, state, value, width);
  auto result = dyn_cast<ConstantIntOp>(builder.create(state));
  if (!(result)) llvm::report_fatal_error("builder didn't return the right type");
  return result;
}

puarith::ConstantIntOp puarith::ConstantIntOp::create(ImplicitLocOpBuilder& builder, int64_t value, unsigned width)
{
  return create(builder, builder.getLoc(), value, width);
}

void puarith::ConstantIntOp::build(OpBuilder& builder, OperationState& result, Type type, int64_t value)
{
  puarith::ConstantOp::build(builder, result, type, builder.getIntegerAttr(type, value));
}

puarith::ConstantIntOp puarith::ConstantIntOp::create(OpBuilder& builder, Location location, Type type, int64_t value)
{
  mlir::OperationState state(location, getOperationName());
  build(builder, state, type, value);
  auto result = dyn_cast<ConstantIntOp>(builder.create(state));
  if (!(result)) llvm::report_fatal_error("builder didn't return the right type");
  return result;
}

puarith::ConstantIntOp puarith::ConstantIntOp::create(ImplicitLocOpBuilder& builder, Type type, int64_t value)
{
  return create(builder, builder.getLoc(), type, value);
}

void puarith::ConstantIntOp::build(OpBuilder& builder, OperationState& result, Type type, const APInt& value)
{
  puarith::ConstantOp::build(builder, result, type, builder.getIntegerAttr(type, value));
}

puarith::ConstantIntOp
puarith::ConstantIntOp::create(OpBuilder& builder, Location location, Type type, const APInt& value)
{
  mlir::OperationState state(location, getOperationName());
  build(builder, state, type, value);
  auto result = dyn_cast<ConstantIntOp>(builder.create(state));
  if (!(result)) llvm::report_fatal_error("builder didn't return the right type");
  return result;
}

puarith::ConstantIntOp puarith::ConstantIntOp::create(ImplicitLocOpBuilder& builder, Type type, const APInt& value)
{
  return create(builder, builder.getLoc(), type, value);
}

bool puarith::ConstantIntOp::classof(Operation* op)
{
  if (auto constOp = dyn_cast_or_null<puarith::ConstantOp>(op)) return constOp.getType().isSignlessInteger();
  return false;
}

void puarith::ConstantFloatOp::build(OpBuilder& builder, OperationState& result, FloatType type, const APFloat& value)
{
  puarith::ConstantOp::build(builder, result, type, builder.getFloatAttr(type, value));
}

puarith::ConstantFloatOp
puarith::ConstantFloatOp::create(OpBuilder& builder, Location location, FloatType type, const APFloat& value)
{
  mlir::OperationState state(location, getOperationName());
  build(builder, state, type, value);
  auto result = dyn_cast<ConstantFloatOp>(builder.create(state));
  if (!(result)) llvm::report_fatal_error("builder didn't return the right type");
  return result;
}

puarith::ConstantFloatOp
puarith::ConstantFloatOp::create(ImplicitLocOpBuilder& builder, FloatType type, const APFloat& value)
{
  return create(builder, builder.getLoc(), type, value);
}

bool puarith::ConstantFloatOp::classof(Operation* op)
{
  if (auto constOp = dyn_cast_or_null<puarith::ConstantOp>(op)) return llvm::isa<FloatType>(constOp.getType());
  return false;
}

void puarith::ConstantIndexOp::build(OpBuilder& builder, OperationState& result, int64_t value)
{
  puarith::ConstantOp::build(builder, result, builder.getIndexType(), builder.getIndexAttr(value));
}

puarith::ConstantIndexOp puarith::ConstantIndexOp::create(OpBuilder& builder, Location location, int64_t value)
{
  mlir::OperationState state(location, getOperationName());
  build(builder, state, value);
  auto result = dyn_cast<ConstantIndexOp>(builder.create(state));
  if (!(result)) llvm::report_fatal_error("builder didn't return the right type");
  return result;
}

puarith::ConstantIndexOp puarith::ConstantIndexOp::create(ImplicitLocOpBuilder& builder, int64_t value)
{
  return create(builder, builder.getLoc(), value);
}

bool puarith::ConstantIndexOp::classof(Operation* op)
{
  if (auto constOp = dyn_cast_or_null<puarith::ConstantOp>(op)) return constOp.getType().isIndex();
  return false;
}

Value mlir::puarith::getZeroConstant(OpBuilder& builder, Location loc, Type type)
{
  if (!(!isa<Float8E8M0FNUType>(getElementTypeOrSelf(type))))
    llvm::report_fatal_error("type doesn't have a zero representation");
  TypedAttr zeroAttr = builder.getZeroAttr(type);
  if (!(zeroAttr)) llvm::report_fatal_error("unsupported type for zero attribute");
  return puarith::ConstantOp::create(builder, loc, zeroAttr);
}

OpFoldResult puarith::AddIOp::fold(FoldAdaptor adaptor)
{
  if (matchPattern(adaptor.getRhs(), m_Zero())) return getLhs();

  if (auto sub = getLhs().getDefiningOp<SubIOp>())
    if (getRhs() == sub.getRhs()) return sub.getLhs();

  if (auto sub = getRhs().getDefiningOp<SubIOp>())
    if (getLhs() == sub.getRhs()) return sub.getLhs();

  return constFoldBinaryOp<IntegerAttr>(adaptor.getOperands(), [](APInt a, const APInt& b) {
    return std::move(a) + b;
  });
}

void puarith::AddIOp::getCanonicalizationPatterns(RewritePatternSet& patterns, MLIRContext* context)
{
  // clang-format off
  patterns.add<
      AddIAddConstant,
      AddISubConstantRHS,
      AddISubConstantLHS,
      AddIMulNegativeOneRhs,
      AddIMulNegativeOneLhs
  >(
    context
  );
  // clang-format on
}

std::optional<SmallVector<int64_t, 4>> puarith::AddUIExtendedOp::getShapeForUnroll()
{
  if (auto vt = dyn_cast<VectorType>(getType(0))) return llvm::to_vector<4>(vt.getShape());
  return std::nullopt;
}

static APInt calculateUnsignedOverflow(const APInt& sum, const APInt& operand)
{
  return sum.ult(operand) ? APInt::getAllOnes(1) : APInt::getZero(1);
}

LogicalResult puarith::AddUIExtendedOp::fold(FoldAdaptor adaptor, SmallVectorImpl<OpFoldResult>& results)
{
  Type overflowTy = getOverflow().getType();
  if (matchPattern(getRhs(), m_Zero())) {
    Builder builder(getContext());
    auto falseValue = builder.getZeroAttr(overflowTy);

    results.push_back(getLhs());
    results.push_back(falseValue);
    return success();
  }

  if (Attribute sumAttr = constFoldBinaryOp<IntegerAttr>(adaptor.getOperands(), [](APInt a, const APInt& b) {
        return std::move(a) + b;
      })) {
    if (matchPattern(sumAttr, ub::m_Poison())) {
      results.push_back(sumAttr);
      results.push_back(sumAttr);
      return success();
    }
    Attribute overflowAttr = constFoldBinaryOp<IntegerAttr>(
      ArrayRef({sumAttr, adaptor.getLhs()}), getI8SameShape(llvm::cast<TypedAttr>(sumAttr).getType()),
      calculateUnsignedOverflow
    );
    if (!overflowAttr) return failure();

    results.push_back(sumAttr);
    results.push_back(overflowAttr);
    return success();
  }

  return failure();
}

void puarith::AddUIExtendedOp::getCanonicalizationPatterns(RewritePatternSet& patterns, MLIRContext* context)
{
  patterns.add<AddUIExtendedToAddI>(context);
}

std::optional<SmallVector<int64_t, 4>> puarith::SubUIExtendedOp::getShapeForUnroll()
{
  if (auto vt = dyn_cast<VectorType>(getType(0))) return llvm::to_vector<4>(vt.getShape());
  return std::nullopt;
}

static APInt calculateUnsignedBorrow(const APInt& lhs, const APInt& rhs)
{
  return lhs.ult(rhs) ? APInt::getAllOnes(1) : APInt::getZero(1);
}

LogicalResult puarith::SubUIExtendedOp::fold(FoldAdaptor adaptor, SmallVectorImpl<OpFoldResult>& results)
{
  Type borrowTy = getBorrow().getType();
  if (matchPattern(getRhs(), m_Zero())) {
    Builder builder(getContext());
    auto falseValue = builder.getZeroAttr(borrowTy);

    results.push_back(getLhs());
    results.push_back(falseValue);
    return success();
  }

  if (getLhs() == getRhs()) {
    auto shapedType = dyn_cast<ShapedType>(getDiff().getType());
    if (shapedType && !shapedType.hasStaticShape()) return failure();
    Builder builder(getContext());
    auto zeroDiff   = builder.getZeroAttr(getDiff().getType());
    auto falseValue = builder.getZeroAttr(borrowTy);
    if (!zeroDiff) return failure();

    results.push_back(zeroDiff);
    results.push_back(falseValue);
    return success();
  }

  if (Attribute diffAttr = constFoldBinaryOp<IntegerAttr>(adaptor.getOperands(), [](APInt a, const APInt& b) {
        return std::move(a) - b;
      })) {
    if (matchPattern(diffAttr, ub::m_Poison())) {
      results.push_back(diffAttr);
      results.push_back(diffAttr);
      return success();
    }
    Attribute borrowAttr = constFoldBinaryOp<IntegerAttr>(
      adaptor.getOperands(), getI8SameShape(llvm::cast<TypedAttr>(diffAttr).getType()), calculateUnsignedBorrow
    );
    if (!borrowAttr) return failure();

    results.push_back(diffAttr);
    results.push_back(borrowAttr);
    return success();
  }

  return failure();
}

void puarith::SubUIExtendedOp::getCanonicalizationPatterns(RewritePatternSet& patterns, MLIRContext* context)
{
  patterns.add<SubUIExtendedToSubI>(context);
}

OpFoldResult puarith::SubIOp::fold(FoldAdaptor adaptor)
{
  if (getOperand(0) == getOperand(1)) {
    auto shapedType = dyn_cast<ShapedType>(getType());
    if (!shapedType || shapedType.hasStaticShape()) return Builder(getContext()).getZeroAttr(getType());
  }
  if (matchPattern(adaptor.getRhs(), m_Zero())) return getLhs();

  if (auto add = getLhs().getDefiningOp<AddIOp>()) {
    if (getRhs() == add.getRhs()) return add.getLhs();
    if (getRhs() == add.getLhs()) return add.getRhs();
  }

  if (auto sub = getRhs().getDefiningOp<SubIOp>())
    if (getLhs() == sub.getLhs()) return sub.getRhs();

  return constFoldBinaryOp<IntegerAttr>(adaptor.getOperands(), [](APInt a, const APInt& b) {
    return std::move(a) - b;
  });
}

void puarith::SubIOp::getCanonicalizationPatterns(RewritePatternSet& patterns, MLIRContext* context)
{
  // clang-format off
  patterns.add<
      SubIRHSAddConstant,
      SubILHSAddConstant,
      SubIRHSSubConstantRHS,
      SubIRHSSubConstantLHS,
      SubILHSSubConstantRHS,
      SubILHSSubConstantLHS,
      SubISubILHSRHSLHS
  >(context);
  // clang-format on
}

OpFoldResult puarith::MulIOp::fold(FoldAdaptor adaptor)
{
  if (matchPattern(adaptor.getRhs(), m_Zero())) return getRhs();
  if (matchPattern(adaptor.getRhs(), m_One())) return getLhs();
  return constFoldBinaryOp<IntegerAttr>(adaptor.getOperands(), [](const APInt& a, const APInt& b) { return a * b; });
}

void puarith::MulIOp::getAsmResultNames(function_ref<void(Value, StringRef)> setNameFn)
{
  if (!isa<IndexType>(getType())) return;

  auto isVscale = [](Operation* op) { return op && op->getName().getStringRef() == "vector.vscale"; };

  IntegerAttr baseValue;
  auto isVscaleExpr = [&](Value a, Value b) {
    return matchPattern(a, m_Constant(&baseValue)) && isVscale(b.getDefiningOp());
  };

  if (!isVscaleExpr(getLhs(), getRhs()) && !isVscaleExpr(getRhs(), getLhs())) return;

  SmallString<32> specialNameBuffer;
  llvm::raw_svector_ostream specialName(specialNameBuffer);
  specialName << 'c' << baseValue.getInt() << "_vscale";
  setNameFn(getResult(), specialName.str());
}

void puarith::MulIOp::getCanonicalizationPatterns(RewritePatternSet& patterns, MLIRContext* context)
{
  patterns.add<MulIMulIConstant>(context);
}

std::optional<SmallVector<int64_t, 4>> puarith::MulSIExtendedOp::getShapeForUnroll()
{
  if (auto vt = dyn_cast<VectorType>(getType(0))) return llvm::to_vector<4>(vt.getShape());
  return std::nullopt;
}

LogicalResult puarith::MulSIExtendedOp::fold(FoldAdaptor adaptor, SmallVectorImpl<OpFoldResult>& results)
{
  if (matchPattern(adaptor.getRhs(), m_Zero())) {
    Attribute zero = adaptor.getRhs();
    results.push_back(zero);
    results.push_back(zero);
    return success();
  }

  if (Attribute lowAttr =
        constFoldBinaryOp<IntegerAttr>(adaptor.getOperands(), [](const APInt& a, const APInt& b) { return a * b; })) {
    Attribute highAttr = constFoldBinaryOp<IntegerAttr>(adaptor.getOperands(), llvm::APIntOps::mulhs);
    if (!(highAttr)) llvm::report_fatal_error("Unexpected constant-folding failure");

    results.push_back(lowAttr);
    results.push_back(highAttr);
    return success();
  }

  return failure();
}

void puarith::MulSIExtendedOp::getCanonicalizationPatterns(RewritePatternSet& patterns, MLIRContext* context)
{
  patterns.add<MulSIExtendedToMulI, MulSIExtendedRHSOne>(context);
}

std::optional<SmallVector<int64_t, 4>> puarith::MulUIExtendedOp::getShapeForUnroll()
{
  if (auto vt = dyn_cast<VectorType>(getType(0))) return llvm::to_vector<4>(vt.getShape());
  return std::nullopt;
}

LogicalResult puarith::MulUIExtendedOp::fold(FoldAdaptor adaptor, SmallVectorImpl<OpFoldResult>& results)
{
  if (matchPattern(adaptor.getRhs(), m_Zero())) {
    Attribute zero = adaptor.getRhs();
    results.push_back(zero);
    results.push_back(zero);
    return success();
  }

  if (matchPattern(adaptor.getRhs(), m_One())) {
    Builder builder(getContext());
    Attribute zero = builder.getZeroAttr(getLhs().getType());
    results.push_back(getLhs());
    results.push_back(zero);
    return success();
  }

  if (Attribute lowAttr =
        constFoldBinaryOp<IntegerAttr>(adaptor.getOperands(), [](const APInt& a, const APInt& b) { return a * b; })) {
    Attribute highAttr = constFoldBinaryOp<IntegerAttr>(adaptor.getOperands(), llvm::APIntOps::mulhu);
    if (!(highAttr)) llvm::report_fatal_error("Unexpected constant-folding failure");

    results.push_back(lowAttr);
    results.push_back(highAttr);
    return success();
  }

  return failure();
}

void puarith::MulUIExtendedOp::getCanonicalizationPatterns(RewritePatternSet& patterns, MLIRContext* context)
{
  patterns.add<MulUIExtendedToMulI>(context);
}

static Value foldDivMul(Value lhs, Value rhs, puarith::IntegerOverflowFlags ovfFlags)
{
  auto mul = lhs.getDefiningOp<mlir::puarith::MulIOp>();
  if (!mul || !bitEnumContainsAll(mul.getOverflowFlags(), ovfFlags)) return {};

  if (mul.getLhs() == rhs) return mul.getRhs();

  if (mul.getRhs() == rhs) return mul.getLhs();

  return {};
}

OpFoldResult puarith::DivUIOp::fold(FoldAdaptor adaptor)
{
  if (matchPattern(adaptor.getRhs(), m_One())) return getLhs();

  if (matchPattern(adaptor.getLhs(), m_Zero())) return getLhs();

  if (getLhs() == getRhs()) return getIntegerAttrOfType(getType(), 1);

  if (Value val = foldDivMul(getLhs(), getRhs(), IntegerOverflowFlags::nuw)) return val;

  bool div0   = false;
  auto result = constFoldBinaryOp<IntegerAttr>(adaptor.getOperands(), [&](APInt a, const APInt& b) {
    if (div0 || !b) {
      div0 = true;
      return a;
    }
    return a.udiv(b);
  });

  return div0 ? Attribute() : result;
}

static Speculation::Speculatability getDivUISpeculatability(Value divisor)
{
  if (matchPattern(divisor, m_IntRangeWithoutZeroU())) return Speculation::Speculatable;

  return Speculation::NotSpeculatable;
}

Speculation::Speculatability puarith::DivUIOp::getSpeculatability()
{
  return getDivUISpeculatability(getRhs());
}

OpFoldResult puarith::DivSIOp::fold(FoldAdaptor adaptor)
{
  if (matchPattern(adaptor.getRhs(), m_One())) return getLhs();

  if (matchPattern(adaptor.getLhs(), m_Zero())) return getLhs();

  if (getLhs() == getRhs()) return getIntegerAttrOfType(getType(), 1);

  if (Value val = foldDivMul(getLhs(), getRhs(), IntegerOverflowFlags::nsw)) return val;

  bool overflowOrDiv0 = false;
  auto result         = constFoldBinaryOp<IntegerAttr>(adaptor.getOperands(), [&](APInt a, const APInt& b) {
    if (overflowOrDiv0 || !b) {
      overflowOrDiv0 = true;
      return a;
    }
    return a.sdiv_ov(b, overflowOrDiv0);
  });

  return overflowOrDiv0 ? Attribute() : result;
}

static Speculation::Speculatability getDivSISpeculatability(Value divisor)
{
  if (matchPattern(divisor, m_IntRangeWithoutZeroS()) && matchPattern(divisor, m_IntRangeWithoutNegOneS()))
    return Speculation::Speculatable;

  return Speculation::NotSpeculatable;
}

Speculation::Speculatability puarith::DivSIOp::getSpeculatability()
{
  return getDivSISpeculatability(getRhs());
}

static APInt signedCeilNonnegInputs(const APInt& a, const APInt& b, bool& overflow)
{
  APInt one(a.getBitWidth(), 1, true);
  APInt val = a.ssub_ov(one, overflow).sdiv_ov(b, overflow);
  return val.sadd_ov(one, overflow);
}

OpFoldResult puarith::CeilDivUIOp::fold(FoldAdaptor adaptor)
{
  if (matchPattern(adaptor.getRhs(), m_One())) return getLhs();

  if (matchPattern(adaptor.getLhs(), m_Zero())) return getLhs();

  if (getLhs() == getRhs()) return getIntegerAttrOfType(getType(), 1);

  bool overflowOrDiv0 = false;
  auto result         = constFoldBinaryOp<IntegerAttr>(adaptor.getOperands(), [&](APInt a, const APInt& b) {
    if (overflowOrDiv0 || !b) {
      overflowOrDiv0 = true;
      return a;
    }
    APInt quotient = a.udiv(b);
    if (!a.urem(b)) return quotient;
    APInt one(a.getBitWidth(), 1, true);
    return quotient.uadd_ov(one, overflowOrDiv0);
  });

  return overflowOrDiv0 ? Attribute() : result;
}

Speculation::Speculatability puarith::CeilDivUIOp::getSpeculatability()
{
  return getDivUISpeculatability(getRhs());
}

OpFoldResult puarith::CeilDivSIOp::fold(FoldAdaptor adaptor)
{
  if (matchPattern(adaptor.getRhs(), m_One())) return getLhs();

  if (matchPattern(adaptor.getLhs(), m_Zero())) return getLhs();

  if (getLhs() == getRhs()) return getIntegerAttrOfType(getType(), 1);

  bool overflowOrDiv0 = false;
  auto result         = constFoldBinaryOp<IntegerAttr>(adaptor.getOperands(), [&](APInt a, const APInt& b) {
    if (overflowOrDiv0 || !b) {
      overflowOrDiv0 = true;
      return a;
    }
    if (!a) return a;
    unsigned bits = a.getBitWidth();
    APInt zero    = APInt::getZero(bits);
    bool aGtZero  = a.sgt(zero);
    bool bGtZero  = b.sgt(zero);
    if (aGtZero && bGtZero) {
      return signedCeilNonnegInputs(a, b, overflowOrDiv0);
    }

    bool overflowNegA   = false;
    bool overflowNegB   = false;
    bool overflowDiv    = false;
    bool overflowNegRes = false;
    if (!aGtZero && !bGtZero) {
      APInt posA     = zero.ssub_ov(a, overflowNegA);
      APInt posB     = zero.ssub_ov(b, overflowNegB);
      APInt res      = signedCeilNonnegInputs(posA, posB, overflowDiv);
      overflowOrDiv0 = (overflowNegA || overflowNegB || overflowDiv);
      return res;
    }
    if (!aGtZero && bGtZero) {
      APInt posA     = zero.ssub_ov(a, overflowNegA);
      APInt div      = posA.sdiv_ov(b, overflowDiv);
      APInt res      = zero.ssub_ov(div, overflowNegRes);
      overflowOrDiv0 = (overflowNegA || overflowDiv || overflowNegRes);
      return res;
    }
    APInt posB = zero.ssub_ov(b, overflowNegB);
    APInt div  = a.sdiv_ov(posB, overflowDiv);
    APInt res  = zero.ssub_ov(div, overflowNegRes);

    overflowOrDiv0 = (overflowNegB || overflowDiv || overflowNegRes);
    return res;
  });

  return overflowOrDiv0 ? Attribute() : result;
}

Speculation::Speculatability puarith::CeilDivSIOp::getSpeculatability()
{
  return getDivSISpeculatability(getRhs());
}

OpFoldResult puarith::FloorDivSIOp::fold(FoldAdaptor adaptor)
{
  if (matchPattern(adaptor.getRhs(), m_One())) return getLhs();

  if (matchPattern(adaptor.getLhs(), m_Zero())) return getLhs();

  if (getLhs() == getRhs()) return getIntegerAttrOfType(getType(), 1);

  bool overflowOrDiv = false;
  auto result        = constFoldBinaryOp<IntegerAttr>(adaptor.getOperands(), [&](APInt a, const APInt& b) {
    if (b.isZero()) {
      overflowOrDiv = true;
      return a;
    }
    return a.sfloordiv_ov(b, overflowOrDiv);
  });

  return overflowOrDiv ? Attribute() : result;
}

OpFoldResult puarith::RemUIOp::fold(FoldAdaptor adaptor)
{
  if (matchPattern(adaptor.getRhs(), m_One())) return getIntegerAttrOfType(getType(), 0);

  if (matchPattern(adaptor.getLhs(), m_Zero()) || getLhs() == getRhs()) return getIntegerAttrOfType(getType(), 0);

  bool div0   = false;
  auto result = constFoldBinaryOp<IntegerAttr>(adaptor.getOperands(), [&](APInt a, const APInt& b) {
    if (div0 || b.isZero()) {
      div0 = true;
      return a;
    }
    return a.urem(b);
  });

  return div0 ? Attribute() : result;
}

Speculation::Speculatability puarith::RemUIOp::getSpeculatability()
{
  return getDivUISpeculatability(getRhs());
}

OpFoldResult puarith::RemSIOp::fold(FoldAdaptor adaptor)
{
  if (matchPattern(adaptor.getRhs(), m_One())) return getIntegerAttrOfType(getType(), 0);

  if (matchPattern(adaptor.getLhs(), m_Zero()) || getLhs() == getRhs()) return getIntegerAttrOfType(getType(), 0);

  bool div0   = false;
  auto result = constFoldBinaryOp<IntegerAttr>(adaptor.getOperands(), [&](APInt a, const APInt& b) {
    if (div0 || b.isZero()) {
      div0 = true;
      return a;
    }
    return a.srem(b);
  });

  return div0 ? Attribute() : result;
}

Speculation::Speculatability puarith::RemSIOp::getSpeculatability()
{
  if (matchPattern(getRhs(), m_IntRangeWithoutZeroS())) return Speculation::Speculatable;

  return Speculation::NotSpeculatable;
}

static Value foldAndIofAndI(puarith::AndIOp op)
{
  for (bool reversePrev : {false, true}) {
    auto prev = (reversePrev ? op.getRhs() : op.getLhs()).getDefiningOp<puarith::AndIOp>();
    if (!prev) continue;

    Value other = (reversePrev ? op.getLhs() : op.getRhs());
    if (other != prev.getLhs() && other != prev.getRhs()) continue;

    return prev.getResult();
  }
  return {};
}

OpFoldResult puarith::AndIOp::fold(FoldAdaptor adaptor)
{
  if (matchPattern(adaptor.getRhs(), m_Zero())) return getRhs();
  APInt intValue;
  if (matchPattern(adaptor.getRhs(), m_ConstantInt(&intValue)) && intValue.isAllOnes()) return getLhs();
  if (matchPattern(getRhs(), m_Op<XOrIOp>(matchers::m_Val(getLhs()), m_ConstantInt(&intValue))) && intValue.isAllOnes())
    return Builder(getContext()).getZeroAttr(getType());
  if (matchPattern(getLhs(), m_Op<XOrIOp>(matchers::m_Val(getRhs()), m_ConstantInt(&intValue))) && intValue.isAllOnes())
    return Builder(getContext()).getZeroAttr(getType());

  if (Value result = foldAndIofAndI(*this)) return result;

  return constFoldBinaryOp<IntegerAttr>(adaptor.getOperands(), [](APInt a, const APInt& b) {
    return std::move(a) & b;
  });
}

OpFoldResult puarith::OrIOp::fold(FoldAdaptor adaptor)
{
  if (APInt rhsVal; matchPattern(adaptor.getRhs(), m_ConstantInt(&rhsVal))) {
    if (rhsVal.isZero()) return getLhs();
    if (rhsVal.isAllOnes()) return adaptor.getRhs();
  }

  APInt intValue;
  if (matchPattern(getRhs(), m_Op<XOrIOp>(matchers::m_Val(getLhs()), m_ConstantInt(&intValue))) && intValue.isAllOnes())
    return getRhs().getDefiningOp<XOrIOp>().getRhs();
  if (matchPattern(getLhs(), m_Op<XOrIOp>(matchers::m_Val(getRhs()), m_ConstantInt(&intValue))) && intValue.isAllOnes())
    return getLhs().getDefiningOp<XOrIOp>().getRhs();

  return constFoldBinaryOp<IntegerAttr>(adaptor.getOperands(), [](APInt a, const APInt& b) {
    return std::move(a) | b;
  });
}

OpFoldResult puarith::XOrIOp::fold(FoldAdaptor adaptor)
{
  if (matchPattern(adaptor.getRhs(), m_Zero())) return getLhs();
  if (getLhs() == getRhs()) {
    auto shapedType = dyn_cast<ShapedType>(getType());
    if (!shapedType || shapedType.hasStaticShape()) return Builder(getContext()).getZeroAttr(getType());
  }
  if (puarith::XOrIOp prev = getLhs().getDefiningOp<puarith::XOrIOp>()) {
    if (prev.getRhs() == getRhs()) return prev.getLhs();
    if (prev.getLhs() == getRhs()) return prev.getRhs();
  }
  if (puarith::XOrIOp prev = getRhs().getDefiningOp<puarith::XOrIOp>()) {
    if (prev.getRhs() == getLhs()) return prev.getLhs();
    if (prev.getLhs() == getLhs()) return prev.getRhs();
  }

  return constFoldBinaryOp<IntegerAttr>(adaptor.getOperands(), [](APInt a, const APInt& b) {
    return std::move(a) ^ b;
  });
}

void puarith::XOrIOp::getCanonicalizationPatterns(RewritePatternSet& patterns, MLIRContext* context)
{
  // clang-format off
  patterns.add<
      XOrIXOrIConstant,
      XOrINotCmpI,
      XOrIOfExtUI,
      XOrIOfExtSI
  >(context);
  // clang-format on
}

OpFoldResult puarith::NegFOp::fold(FoldAdaptor adaptor)
{
  if (auto op = this->getOperand().getDefiningOp<puarith::NegFOp>()) return op.getOperand();
  return constFoldUnaryOp<FloatAttr>(adaptor.getOperands(), [](const APFloat& a) { return -a; });
}

OpFoldResult puarith::FlushDenormalsOp::fold(FoldAdaptor adaptor)
{
  if (auto op = this->getOperand().getDefiningOp<puarith::FlushDenormalsOp>()) return op.getResult();

  return constFoldUnaryOp<FloatAttr>(adaptor.getOperands(), [](const APFloat& a) {
    if (a.isDenormal()) return APFloat::getZero(a.getSemantics(), a.isNegative());
    return a;
  });
}

OpFoldResult puarith::AddFOp::fold(FoldAdaptor adaptor)
{
  if (matchPattern(adaptor.getRhs(), m_NegZeroFloat())) return getLhs();

  auto rm = getRoundingmode();
  return constFoldBinaryOp<FloatAttr>(adaptor.getOperands(), [rm](const APFloat& a, const APFloat& b) {
    APFloat result(a);
    result.add(b, convertArithRoundingModeToLLVMIR(rm));
    return result;
  });
}

void puarith::AddFOp::getCanonicalizationPatterns(RewritePatternSet& patterns, MLIRContext* context)
{
  patterns.add<AddFOfNegFLhs, AddFOfNegFRhs>(context);
}

OpFoldResult puarith::SubFOp::fold(FoldAdaptor adaptor)
{
  if (matchPattern(adaptor.getRhs(), m_PosZeroFloat())) return getLhs();

  auto rm = getRoundingmode();
  return constFoldBinaryOp<FloatAttr>(adaptor.getOperands(), [rm](const APFloat& a, const APFloat& b) {
    APFloat result(a);
    result.subtract(b, convertArithRoundingModeToLLVMIR(rm));
    return result;
  });
}

void puarith::SubFOp::getCanonicalizationPatterns(RewritePatternSet& patterns, MLIRContext* context)
{
  patterns.add<SubFOfNegZero>(context);
}

OpFoldResult puarith::MaximumFOp::fold(FoldAdaptor adaptor)
{
  if (getLhs() == getRhs()) return getRhs();

  if (matchPattern(adaptor.getRhs(), m_NegInfFloat())) return getLhs();

  return constFoldBinaryOp<FloatAttr>(adaptor.getOperands(), llvm::maximum);
}

OpFoldResult puarith::MaxNumFOp::fold(FoldAdaptor adaptor)
{
  if (getLhs() == getRhs()) return getRhs();

  if (matchPattern(adaptor.getRhs(), m_NaNFloat())) return getLhs();

  return constFoldBinaryOp<FloatAttr>(adaptor.getOperands(), llvm::maxnum);
}

OpFoldResult MaxSIOp::fold(FoldAdaptor adaptor)
{
  if (getLhs() == getRhs()) return getRhs();

  if (APInt intValue; matchPattern(adaptor.getRhs(), m_ConstantInt(&intValue))) {
    if (intValue.isMaxSignedValue()) return getRhs();
    if (intValue.isMinSignedValue()) return getLhs();
  }

  return constFoldBinaryOp<IntegerAttr>(adaptor.getOperands(), llvm::APIntOps::smax);
}

OpFoldResult MaxUIOp::fold(FoldAdaptor adaptor)
{
  if (getLhs() == getRhs()) return getRhs();

  if (APInt intValue; matchPattern(adaptor.getRhs(), m_ConstantInt(&intValue))) {
    if (intValue.isMaxValue()) return getRhs();
    if (intValue.isMinValue()) return getLhs();
  }

  return constFoldBinaryOp<IntegerAttr>(adaptor.getOperands(), llvm::APIntOps::umax);
}

OpFoldResult puarith::MinimumFOp::fold(FoldAdaptor adaptor)
{
  if (getLhs() == getRhs()) return getRhs();

  if (matchPattern(adaptor.getRhs(), m_PosInfFloat())) return getLhs();

  return constFoldBinaryOp<FloatAttr>(adaptor.getOperands(), llvm::minimum);
}

OpFoldResult puarith::MinNumFOp::fold(FoldAdaptor adaptor)
{
  if (getLhs() == getRhs()) return getRhs();

  if (matchPattern(adaptor.getRhs(), m_NaNFloat())) return getLhs();

  return constFoldBinaryOp<FloatAttr>(adaptor.getOperands(), llvm::minnum);
}

OpFoldResult MinSIOp::fold(FoldAdaptor adaptor)
{
  if (getLhs() == getRhs()) return getRhs();

  if (APInt intValue; matchPattern(adaptor.getRhs(), m_ConstantInt(&intValue))) {
    if (intValue.isMinSignedValue()) return getRhs();
    if (intValue.isMaxSignedValue()) return getLhs();
  }

  return constFoldBinaryOp<IntegerAttr>(adaptor.getOperands(), llvm::APIntOps::smin);
}

OpFoldResult MinUIOp::fold(FoldAdaptor adaptor)
{
  if (getLhs() == getRhs()) return getRhs();

  if (APInt intValue; matchPattern(adaptor.getRhs(), m_ConstantInt(&intValue))) {
    if (intValue.isMinValue()) return getRhs();
    if (intValue.isMaxValue()) return getLhs();
  }

  return constFoldBinaryOp<IntegerAttr>(adaptor.getOperands(), llvm::APIntOps::umin);
}

OpFoldResult puarith::MulFOp::fold(FoldAdaptor adaptor)
{
  if (matchPattern(adaptor.getRhs(), m_OneFloat())) return getLhs();

  if (puarith::bitEnumContainsAll(getFastmath(), puarith::FastMathFlags::nnan | puarith::FastMathFlags::nsz)) {
    if (matchPattern(adaptor.getRhs(), m_AnyZeroFloat())) return getRhs();
  }

  auto rm = getRoundingmode();
  return constFoldBinaryOp<FloatAttr>(adaptor.getOperands(), [rm](const APFloat& a, const APFloat& b) {
    APFloat result(a);
    result.multiply(b, convertArithRoundingModeToLLVMIR(rm));
    return result;
  });
}

void puarith::MulFOp::getCanonicalizationPatterns(RewritePatternSet& patterns, MLIRContext* context)
{
  patterns.add<MulFOfNegF>(context);
}

OpFoldResult puarith::DivFOp::fold(FoldAdaptor adaptor)
{
  if (matchPattern(adaptor.getRhs(), m_OneFloat())) return getLhs();

  auto rm = getRoundingmode();
  return constFoldBinaryOp<FloatAttr>(adaptor.getOperands(), [rm](const APFloat& a, const APFloat& b) {
    APFloat result(a);
    result.divide(b, convertArithRoundingModeToLLVMIR(rm));
    return result;
  });
}

void puarith::DivFOp::getCanonicalizationPatterns(RewritePatternSet& patterns, MLIRContext* context)
{
  patterns.add<DivFOfNegF>(context);
}

OpFoldResult puarith::RemFOp::fold(FoldAdaptor adaptor)
{
  return constFoldBinaryOp<FloatAttr>(adaptor.getOperands(), [](const APFloat& a, const APFloat& b) {
    APFloat result(a);
    (void)result.mod(b);
    return result;
  });
}

template <typename... Types> using type_list = std::tuple<Types...>*;

template <typename... ShapedTypes, typename... ElementTypes>
static Type getUnderlyingType(Type type, type_list<ShapedTypes...>, type_list<ElementTypes...>)
{
  if (llvm::isa<ShapedType>(type) && !llvm::isa<ShapedTypes...>(type)) return {};

  auto underlyingType = getElementTypeOrSelf(type);
  if (!llvm::isa<ElementTypes...>(underlyingType)) return {};

  return underlyingType;
}

template <typename... ElementTypes> static Type getTypeIfLike(Type type)
{
  return getUnderlyingType(type, type_list<VectorType, TensorType>(), type_list<ElementTypes...>());
}

template <typename... ElementTypes> static Type getTypeIfLikeOrMemRef(Type type)
{
  return getUnderlyingType(type, type_list<VectorType, TensorType, MemRefType>(), type_list<ElementTypes...>());
}

static bool hasSameEncoding(Type typeA, Type typeB)
{
  auto rankedTensorA = dyn_cast<RankedTensorType>(typeA);
  auto rankedTensorB = dyn_cast<RankedTensorType>(typeB);
  if (!rankedTensorA || !rankedTensorB) return true;
  return rankedTensorA.getEncoding() == rankedTensorB.getEncoding();
}

static bool areValidCastInputsAndOutputs(TypeRange inputs, TypeRange outputs)
{
  if (inputs.size() != 1 || outputs.size() != 1) return false;
  if (!hasSameEncoding(inputs.front(), outputs.front())) return false;
  return succeeded(verifyCompatibleShapes(inputs.front(), outputs.front()));
}

template <typename ValType, typename Op> static LogicalResult verifyExtOp(Op op)
{
  Type srcType = getElementTypeOrSelf(op.getIn().getType());
  Type dstType = getElementTypeOrSelf(op.getType());

  if (llvm::cast<ValType>(srcType).getWidth() >= llvm::cast<ValType>(dstType).getWidth())
    return op.emitError("result type ") << dstType << " must be wider than operand type " << srcType;

  return success();
}

template <typename ValType, typename Op> static LogicalResult verifyTruncateOp(Op op)
{
  Type srcType = getElementTypeOrSelf(op.getIn().getType());
  Type dstType = getElementTypeOrSelf(op.getType());

  if (llvm::cast<ValType>(srcType).getWidth() <= llvm::cast<ValType>(dstType).getWidth())
    return op.emitError("result type ") << dstType << " must be shorter than operand type " << srcType;

  return success();
}

template <template <typename> class WidthComparator, typename... ElementTypes>
static bool checkWidthChangeCast(TypeRange inputs, TypeRange outputs)
{
  if (!areValidCastInputsAndOutputs(inputs, outputs)) return false;

  auto srcType = getTypeIfLike<ElementTypes...>(inputs.front());
  auto dstType = getTypeIfLike<ElementTypes...>(outputs.front());
  if (!srcType || !dstType) return false;

  return WidthComparator<unsigned>()(dstType.getIntOrFloatBitWidth(), srcType.getIntOrFloatBitWidth());
}

static FailureOr<APFloat> convertFloatValue(
  APFloat sourceValue, const llvm::fltSemantics& targetSemantics, llvm::RoundingMode roundingMode = kDefaultRoundingMode
)
{
  using fltNonfiniteBehavior = llvm::fltNonfiniteBehavior;
  if (sourceValue.isInfinity() && (targetSemantics.nonFiniteBehavior == fltNonfiniteBehavior::NanOnly ||
                                   targetSemantics.nonFiniteBehavior == fltNonfiniteBehavior::FiniteOnly))
    return failure();
  if (sourceValue.isNaN() && targetSemantics.nonFiniteBehavior == fltNonfiniteBehavior::FiniteOnly) return failure();

  bool losesInfo = false;
  auto status    = sourceValue.convert(targetSemantics, roundingMode, &losesInfo);
  if (losesInfo || status != APFloat::opOK) return failure();

  return sourceValue;
}

OpFoldResult puarith::ExtUIOp::fold(FoldAdaptor adaptor)
{
  if (auto lhs = getIn().getDefiningOp<ExtUIOp>()) {
    getInMutable().assign(lhs.getIn());
    return getResult();
  }

  Type resType      = getElementTypeOrSelf(getType());
  unsigned bitWidth = llvm::cast<IntegerType>(resType).getWidth();
  return constFoldCastOp<IntegerAttr, IntegerAttr>(
    adaptor.getOperands(), getType(), [bitWidth](const APInt& a, bool& castStatus) { return a.zext(bitWidth); }
  );
}

bool puarith::ExtUIOp::areCastCompatible(TypeRange inputs, TypeRange outputs)
{
  return checkWidthChangeCast<std::greater, IntegerType>(inputs, outputs);
}

LogicalResult puarith::ExtUIOp::verify()
{
  return verifyExtOp<IntegerType>(*this);
}

OpFoldResult puarith::ExtSIOp::fold(FoldAdaptor adaptor)
{
  if (auto lhs = getIn().getDefiningOp<ExtSIOp>()) {
    getInMutable().assign(lhs.getIn());
    return getResult();
  }

  Type resType      = getElementTypeOrSelf(getType());
  unsigned bitWidth = llvm::cast<IntegerType>(resType).getWidth();
  return constFoldCastOp<IntegerAttr, IntegerAttr>(
    adaptor.getOperands(), getType(), [bitWidth](const APInt& a, bool& castStatus) { return a.sext(bitWidth); }
  );
}

bool puarith::ExtSIOp::areCastCompatible(TypeRange inputs, TypeRange outputs)
{
  return checkWidthChangeCast<std::greater, IntegerType>(inputs, outputs);
}

void puarith::ExtSIOp::getCanonicalizationPatterns(RewritePatternSet& patterns, MLIRContext* context)
{
  patterns.add<ExtSIOfExtUI>(context);
}

LogicalResult puarith::ExtSIOp::verify()
{
  return verifyExtOp<IntegerType>(*this);
}

OpFoldResult puarith::ExtFOp::fold(FoldAdaptor adaptor)
{
  if (auto truncFOp = getOperand().getDefiningOp<TruncFOp>()) {
    if (truncFOp.getOperand().getType() == getType()) {
      puarith::FastMathFlags truncFMF = truncFOp.getFastmath().value_or(puarith::FastMathFlags::none);
      bool isTruncContract            = bitEnumContainsAll(truncFMF, puarith::FastMathFlags::contract);
      puarith::FastMathFlags extFMF   = getFastmath().value_or(puarith::FastMathFlags::none);
      bool isExtContract              = bitEnumContainsAll(extFMF, puarith::FastMathFlags::contract);
      if (isTruncContract && isExtContract) {
        return truncFOp.getOperand();
      }
    }
  }

  auto resElemType                          = cast<FloatType>(getElementTypeOrSelf(getType()));
  const llvm::fltSemantics& targetSemantics = resElemType.getFloatSemantics();
  return constFoldCastOp<FloatAttr, FloatAttr>(
    adaptor.getOperands(), getType(),
    [&targetSemantics](const APFloat& a, bool& castStatus) {
      FailureOr<APFloat> result = convertFloatValue(a, targetSemantics);
      if (failed(result)) {
        castStatus = false;
        return a;
      }
      return *result;
    }
  );
}

bool puarith::ExtFOp::areCastCompatible(TypeRange inputs, TypeRange outputs)
{
  return checkWidthChangeCast<std::greater, FloatType>(inputs, outputs);
}

LogicalResult puarith::ExtFOp::verify()
{
  return verifyExtOp<FloatType>(*this);
}

bool puarith::ScalingExtFOp::areCastCompatible(TypeRange inputs, TypeRange outputs)
{
  return checkWidthChangeCast<std::greater, FloatType>(inputs.front(), outputs);
}

LogicalResult puarith::ScalingExtFOp::verify()
{
  return verifyExtOp<FloatType>(*this);
}

OpFoldResult puarith::TruncIOp::fold(FoldAdaptor adaptor)
{
  if (matchPattern(getOperand(), m_Op<puarith::ExtUIOp>()) || matchPattern(getOperand(), m_Op<puarith::ExtSIOp>())) {
    Value src    = getOperand().getDefiningOp()->getOperand(0);
    Type srcType = getElementTypeOrSelf(src.getType());
    Type dstType = getElementTypeOrSelf(getType());
    if (llvm::cast<IntegerType>(srcType).getWidth() > llvm::cast<IntegerType>(dstType).getWidth()) {
      setOperand(src);
      return getResult();
    }

    if (srcType == dstType) return src;
  }

  if (matchPattern(getOperand(), m_Op<puarith::TruncIOp>())) {
    setOperand(getOperand().getDefiningOp()->getOperand(0));
    return getResult();
  }

  Type resType      = getElementTypeOrSelf(getType());
  unsigned bitWidth = llvm::cast<IntegerType>(resType).getWidth();
  return constFoldCastOp<IntegerAttr, IntegerAttr>(
    adaptor.getOperands(), getType(), [bitWidth](const APInt& a, bool& castStatus) { return a.trunc(bitWidth); }
  );
}

bool puarith::TruncIOp::areCastCompatible(TypeRange inputs, TypeRange outputs)
{
  return checkWidthChangeCast<std::less, IntegerType>(inputs, outputs);
}

void puarith::TruncIOp::getCanonicalizationPatterns(RewritePatternSet& patterns, MLIRContext* context)
{
  // clang-format off
  patterns.add<
      TruncIExtSIToExtSI,
      TruncIExtUIToExtUI,
      TruncIShrSIToTrunciShrUI
  >(context);
  // clang-format on
}

LogicalResult puarith::TruncIOp::verify()
{
  return verifyTruncateOp<IntegerType>(*this);
}

OpFoldResult puarith::TruncFOp::fold(FoldAdaptor adaptor)
{
  auto resElemType = cast<FloatType>(getElementTypeOrSelf(getType()));
  if (auto extOp = getOperand().getDefiningOp<puarith::ExtFOp>()) {
    Value src             = extOp.getIn();
    auto srcType          = cast<FloatType>(getElementTypeOrSelf(src.getType()));
    auto intermediateType = cast<FloatType>(getElementTypeOrSelf(extOp.getType()));
    if (llvm::APFloatBase::isRepresentableBy(srcType.getFloatSemantics(), intermediateType.getFloatSemantics())) {
      if (srcType.getWidth() > resElemType.getWidth()) {
        setOperand(src);
        return getResult();
      }

      if (srcType == resElemType) return src;
    }
  }

  const llvm::fltSemantics& targetSemantics = resElemType.getFloatSemantics();
  return constFoldCastOp<FloatAttr, FloatAttr>(
    adaptor.getOperands(), getType(),
    [this, &targetSemantics](const APFloat& a, bool& castStatus) {
      llvm::RoundingMode llvmRoundingMode = convertArithRoundingModeToLLVMIR(getRoundingmode());
      FailureOr<APFloat> result           = convertFloatValue(a, targetSemantics, llvmRoundingMode);
      if (failed(result)) {
        castStatus = false;
        return a;
      }
      return *result;
    }
  );
}

void puarith::TruncFOp::getCanonicalizationPatterns(RewritePatternSet& patterns, MLIRContext* context)
{
  patterns.add<TruncFSIToFPToSIToFP, TruncFUIToFPToUIToFP>(context);
}

bool puarith::TruncFOp::areCastCompatible(TypeRange inputs, TypeRange outputs)
{
  return checkWidthChangeCast<std::less, FloatType>(inputs, outputs);
}

LogicalResult puarith::TruncFOp::verify()
{
  return verifyTruncateOp<FloatType>(*this);
}

OpFoldResult puarith::ConvertFOp::fold(FoldAdaptor adaptor)
{
  auto resElemType                          = cast<FloatType>(getElementTypeOrSelf(getType()));
  const llvm::fltSemantics& targetSemantics = resElemType.getFloatSemantics();
  return constFoldCastOp<FloatAttr, FloatAttr>(
    adaptor.getOperands(), getType(),
    [this, &targetSemantics](const APFloat& a, bool& castStatus) {
      llvm::RoundingMode llvmRoundingMode = convertArithRoundingModeToLLVMIR(getRoundingmode());
      FailureOr<APFloat> result           = convertFloatValue(a, targetSemantics, llvmRoundingMode);
      if (failed(result)) {
        castStatus = false;
        return a;
      }
      return *result;
    }
  );
}

bool puarith::ConvertFOp::areCastCompatible(TypeRange inputs, TypeRange outputs)
{
  if (!areValidCastInputsAndOutputs(inputs, outputs)) return false;
  auto srcType = getTypeIfLike<FloatType>(inputs.front());
  auto dstType = getTypeIfLike<FloatType>(outputs.front());
  if (!srcType || !dstType) return false;
  return srcType != dstType && srcType.getIntOrFloatBitWidth() == dstType.getIntOrFloatBitWidth();
}

LogicalResult puarith::ConvertFOp::verify()
{
  auto srcType = cast<FloatType>(getElementTypeOrSelf(getIn().getType()));
  auto dstType = cast<FloatType>(getElementTypeOrSelf(getType()));
  if (srcType == dstType)
    return emitError("result element type ") << dstType << " must be different from operand element type " << srcType;
  if (srcType.getWidth() != dstType.getWidth())
    return emitError("result element type ")
           << dstType << " must have the same bitwidth as operand element type " << srcType;
  return success();
}

bool puarith::ScalingTruncFOp::areCastCompatible(TypeRange inputs, TypeRange outputs)
{
  return checkWidthChangeCast<std::less, FloatType>(inputs.front(), outputs);
}

LogicalResult puarith::ScalingTruncFOp::verify()
{
  return verifyTruncateOp<FloatType>(*this);
}

void puarith::AndIOp::getCanonicalizationPatterns(RewritePatternSet& patterns, MLIRContext* context)
{
  // clang-format off
  patterns.add<
      AndIAndIConstant,
      AndOfExtUI,
      AndOfExtSI
  >(context);
  // clang-format on
}

void puarith::OrIOp::getCanonicalizationPatterns(RewritePatternSet& patterns, MLIRContext* context)
{
  // clang-format off
  patterns.add<
      OrIOrIConstant,
      OrOfExtUI,
      OrOfExtSI
  >(context);
  // clang-format on
}

template <typename From, typename To> static bool checkIntFloatCast(TypeRange inputs, TypeRange outputs)
{
  if (!areValidCastInputsAndOutputs(inputs, outputs)) return false;

  auto srcType = getTypeIfLike<From>(inputs.front());
  auto dstType = getTypeIfLike<To>(outputs.back());

  return srcType && dstType;
}

bool puarith::UIToFPOp::areCastCompatible(TypeRange inputs, TypeRange outputs)
{
  return checkIntFloatCast<IntegerType, FloatType>(inputs, outputs);
}

OpFoldResult puarith::UIToFPOp::fold(FoldAdaptor adaptor)
{
  Type resEleType = getElementTypeOrSelf(getType());
  return constFoldCastOp<IntegerAttr, FloatAttr>(
    adaptor.getOperands(), getType(),
    [&resEleType](const APInt& a, bool& castStatus) {
      FloatType floatTy = llvm::cast<FloatType>(resEleType);
      APFloat apf(floatTy.getFloatSemantics(), APInt::getZero(floatTy.getWidth()));
      apf.convertFromAPInt(a, false, APFloat::rmNearestTiesToEven);
      return apf;
    }
  );
}

void puarith::UIToFPOp::getCanonicalizationPatterns(RewritePatternSet& patterns, MLIRContext* context)
{
  patterns.add<UIToFPOfExtUI>(context);
}

bool puarith::SIToFPOp::areCastCompatible(TypeRange inputs, TypeRange outputs)
{
  return checkIntFloatCast<IntegerType, FloatType>(inputs, outputs);
}

OpFoldResult puarith::SIToFPOp::fold(FoldAdaptor adaptor)
{
  Type resEleType = getElementTypeOrSelf(getType());
  return constFoldCastOp<IntegerAttr, FloatAttr>(
    adaptor.getOperands(), getType(),
    [&resEleType](const APInt& a, bool& castStatus) {
      FloatType floatTy = llvm::cast<FloatType>(resEleType);
      APFloat apf(floatTy.getFloatSemantics(), APInt::getZero(floatTy.getWidth()));
      apf.convertFromAPInt(a, true, APFloat::rmNearestTiesToEven);
      return apf;
    }
  );
}

void puarith::SIToFPOp::getCanonicalizationPatterns(RewritePatternSet& patterns, MLIRContext* context)
{
  patterns.add<SIToFPOfExtSI, SIToFPOfExtUI>(context);
}

bool puarith::FPToUIOp::areCastCompatible(TypeRange inputs, TypeRange outputs)
{
  return checkIntFloatCast<FloatType, IntegerType>(inputs, outputs);
}

OpFoldResult puarith::FPToUIOp::fold(FoldAdaptor adaptor)
{
  Type resType      = getElementTypeOrSelf(getType());
  unsigned bitWidth = llvm::cast<IntegerType>(resType).getWidth();
  return constFoldCastOp<FloatAttr, IntegerAttr>(
    adaptor.getOperands(), getType(),
    [&bitWidth](const APFloat& a, bool& castStatus) {
      bool ignored;
      APSInt api(bitWidth, true);
      castStatus = APFloat::opInvalidOp != a.convertToInteger(api, APFloat::rmTowardZero, &ignored);
      return api;
    }
  );
}

bool puarith::FPToSIOp::areCastCompatible(TypeRange inputs, TypeRange outputs)
{
  return checkIntFloatCast<FloatType, IntegerType>(inputs, outputs);
}

OpFoldResult puarith::FPToSIOp::fold(FoldAdaptor adaptor)
{
  Type resType      = getElementTypeOrSelf(getType());
  unsigned bitWidth = llvm::cast<IntegerType>(resType).getWidth();
  return constFoldCastOp<FloatAttr, IntegerAttr>(
    adaptor.getOperands(), getType(),
    [&bitWidth](const APFloat& a, bool& castStatus) {
      bool ignored;
      APSInt api(bitWidth, false);
      castStatus = APFloat::opInvalidOp != a.convertToInteger(api, APFloat::rmTowardZero, &ignored);
      return api;
    }
  );
}

static unsigned getIndexCastWidth(Type t)
{
  if (auto intTy = dyn_cast<IntegerType>(getElementTypeOrSelf(t))) return intTy.getWidth();
  return IndexType::kInternalStorageBitWidth;
}

static bool areIndexCastCompatible(TypeRange inputs, TypeRange outputs)
{
  if (!areValidCastInputsAndOutputs(inputs, outputs)) return false;

  auto srcType = getTypeIfLikeOrMemRef<IntegerType, IndexType>(inputs.front());
  auto dstType = getTypeIfLikeOrMemRef<IntegerType, IndexType>(outputs.front());
  if (!srcType || !dstType) return false;

  return (srcType.isIndex() && dstType.isSignlessInteger()) || (srcType.isSignlessInteger() && dstType.isIndex());
}

bool puarith::IndexCastOp::areCastCompatible(TypeRange inputs, TypeRange outputs)
{
  return areIndexCastCompatible(inputs, outputs);
}

OpFoldResult puarith::IndexCastOp::fold(FoldAdaptor adaptor)
{
  unsigned resultBitwidth = 64;
  if (auto intTy = dyn_cast<IntegerType>(getElementTypeOrSelf(getType()))) resultBitwidth = intTy.getWidth();

  if (auto foldResult = constFoldCastOp<IntegerAttr, IntegerAttr>(
        adaptor.getOperands(), getType(),
        [resultBitwidth](const APInt& a, bool&) { return a.sextOrTrunc(resultBitwidth); }
      ))
    return foldResult;

  if (auto inner = getOperand().getDefiningOp<puarith::IndexCastOp>()) {
    Value x = inner.getOperand();
    if (x.getType() == getType()) {
      if (getIndexCastWidth(inner.getType()) >= getIndexCastWidth(x.getType())) return x;
    }
  }
  return {};
}

void puarith::IndexCastOp::getCanonicalizationPatterns(RewritePatternSet& patterns, MLIRContext* context)
{
  patterns.add<IndexCastOfExtSI>(context);
}

bool puarith::IndexCastUIOp::areCastCompatible(TypeRange inputs, TypeRange outputs)
{
  return areIndexCastCompatible(inputs, outputs);
}

OpFoldResult puarith::IndexCastUIOp::fold(FoldAdaptor adaptor)
{
  unsigned resultBitwidth = 64;
  if (auto intTy = dyn_cast<IntegerType>(getElementTypeOrSelf(getType()))) resultBitwidth = intTy.getWidth();

  if (auto foldResult = constFoldCastOp<IntegerAttr, IntegerAttr>(
        adaptor.getOperands(), getType(),
        [resultBitwidth](const APInt& a, bool&) { return a.zextOrTrunc(resultBitwidth); }
      ))
    return foldResult;

  if (auto inner = getOperand().getDefiningOp<puarith::IndexCastUIOp>()) {
    Value x = inner.getOperand();
    if (x.getType() == getType()) {
      if (getIndexCastWidth(inner.getType()) >= getIndexCastWidth(x.getType())) return x;
    }
  }
  return {};
}

void puarith::IndexCastUIOp::getCanonicalizationPatterns(RewritePatternSet& patterns, MLIRContext* context)
{
  patterns.add<IndexCastUIOfExtUI>(context);
}

bool puarith::BitcastOp::areCastCompatible(TypeRange inputs, TypeRange outputs)
{
  if (!areValidCastInputsAndOutputs(inputs, outputs)) return false;

  auto srcType = getTypeIfLikeOrMemRef<IntegerType, FloatType>(inputs.front());
  auto dstType = getTypeIfLikeOrMemRef<IntegerType, FloatType>(outputs.front());
  if (!srcType || !dstType) return false;

  return srcType.getIntOrFloatBitWidth() == dstType.getIntOrFloatBitWidth();
}

OpFoldResult puarith::BitcastOp::fold(FoldAdaptor adaptor)
{
  auto resType = getType();
  auto operand = adaptor.getIn();
  if (!operand) return {};

  if (auto denseAttr = dyn_cast_or_null<DenseElementsAttr>(operand))
    return denseAttr.bitcast(llvm::cast<ShapedType>(resType).getElementType());
  if (llvm::isa<ShapedType>(resType)) return {};

  if (matchPattern(operand, ub::m_Poison())) return ub::PoisonAttr::get(getContext());

  if (!llvm::isa<FloatAttr, IntegerAttr>(operand)) return {};

  APInt bits = llvm::isa<FloatAttr>(operand) ? llvm::cast<FloatAttr>(operand).getValue().bitcastToAPInt()
                                             : llvm::cast<IntegerAttr>(operand).getValue();
  if (!(resType.getIntOrFloatBitWidth() == bits.getBitWidth()))
    llvm::report_fatal_error("trying to fold on broken IR: operands have incompatible types");

  if (auto resFloatType = dyn_cast<FloatType>(resType))
    return FloatAttr::get(resType, APFloat(resFloatType.getFloatSemantics(), bits));
  return IntegerAttr::get(resType, bits);
}

void puarith::BitcastOp::getCanonicalizationPatterns(RewritePatternSet& patterns, MLIRContext* context)
{
  patterns.add<BitcastOfBitcast>(context);
}

bool mlir::puarith::applyCmpPredicate(puarith::CmpIPredicate predicate, const APInt& lhs, const APInt& rhs)
{
  switch (predicate) {
    case puarith::CmpIPredicate::eq: return lhs.eq(rhs);
    case puarith::CmpIPredicate::ne: return lhs.ne(rhs);
    case puarith::CmpIPredicate::slt: return lhs.slt(rhs);
    case puarith::CmpIPredicate::sle: return lhs.sle(rhs);
    case puarith::CmpIPredicate::sgt: return lhs.sgt(rhs);
    case puarith::CmpIPredicate::sge: return lhs.sge(rhs);
    case puarith::CmpIPredicate::ult: return lhs.ult(rhs);
    case puarith::CmpIPredicate::ule: return lhs.ule(rhs);
    case puarith::CmpIPredicate::ugt: return lhs.ugt(rhs);
    case puarith::CmpIPredicate::uge: return lhs.uge(rhs);
  }
  llvm_unreachable("unknown cmpi predicate kind");
}

static bool applyCmpPredicateToEqualOperands(puarith::CmpIPredicate predicate)
{
  switch (predicate) {
    case puarith::CmpIPredicate::eq:
    case puarith::CmpIPredicate::sle:
    case puarith::CmpIPredicate::sge:
    case puarith::CmpIPredicate::ule:
    case puarith::CmpIPredicate::uge: return true;
    case puarith::CmpIPredicate::ne:
    case puarith::CmpIPredicate::slt:
    case puarith::CmpIPredicate::sgt:
    case puarith::CmpIPredicate::ult:
    case puarith::CmpIPredicate::ugt: return false;
  }
  llvm_unreachable("unknown cmpi predicate kind");
}

static std::optional<int64_t> getIntegerWidth(Type t)
{
  if (auto intType = dyn_cast<IntegerType>(t)) {
    return intType.getWidth();
  }
  if (auto vectorIntType = dyn_cast<VectorType>(t)) {
    return llvm::cast<IntegerType>(vectorIntType.getElementType()).getWidth();
  }
  return std::nullopt;
}

static LogicalResult verifyCompareOp(Operation* op, Type lhsTy, Type resTy)
{
  auto lhsShaped   = dyn_cast<ShapedType>(lhsTy);
  auto resShaped   = dyn_cast<ShapedType>(resTy);
  int64_t lhsLanes = lhsShaped ? lhsShaped.getNumElements() : 1;
  int64_t resLanes = resShaped ? resShaped.getNumElements() : 1;
  if (lhsLanes != resLanes) return op->emitOpError("result shape must match the operands");
  Type lhsElem = getElementTypeOrSelf(lhsTy);
  Type resElem = getElementTypeOrSelf(resTy);
  if (!resElem.isSignlessInteger(8) && resElem != lhsElem)
    return op->emitOpError("result element type must be i8 or the operand element type, got ") << resTy;
  return success();
}

LogicalResult puarith::CmpIOp::verify()
{
  return verifyCompareOp(*this, getLhs().getType(), getResult().getType());
}

LogicalResult puarith::CmpFOp::verify()
{
  return verifyCompareOp(*this, getLhs().getType(), getResult().getType());
}

OpFoldResult puarith::CmpIOp::fold(FoldAdaptor adaptor)
{
  if (getLhs() == getRhs()) {
    auto val = applyCmpPredicateToEqualOperands(getPredicate());
    return getBoolAttribute(getType(), val);
  }

  if (matchPattern(adaptor.getRhs(), m_Zero())) {
    if (auto extOp = getLhs().getDefiningOp<ExtSIOp>()) {
      std::optional<int64_t> integerWidth = getIntegerWidth(extOp.getOperand().getType());
      if (integerWidth && integerWidth.value() == 1 && getPredicate() == puarith::CmpIPredicate::ne)
        return extOp.getOperand();
    }
    if (auto extOp = getLhs().getDefiningOp<ExtUIOp>()) {
      std::optional<int64_t> integerWidth = getIntegerWidth(extOp.getOperand().getType());
      if (integerWidth && integerWidth.value() == 1 && getPredicate() == puarith::CmpIPredicate::ne)
        return extOp.getOperand();
    }

    if (getElementTypeOrSelf(getLhs().getType()).isInteger(1) && getPredicate() == puarith::CmpIPredicate::ne)
      return getLhs();
  }

  if (matchPattern(adaptor.getRhs(), m_One())) {
    if (getElementTypeOrSelf(getLhs().getType()).isInteger(1) && getPredicate() == puarith::CmpIPredicate::eq)
      return getLhs();
  }

  if (adaptor.getLhs() && !adaptor.getRhs()) {
    using Pred                             = CmpIPredicate;
    const std::pair<Pred, Pred> invPreds[] = {
      {Pred::slt, Pred::sgt}, {Pred::sgt, Pred::slt}, {Pred::sle, Pred::sge}, {Pred::sge, Pred::sle},
      {Pred::ult, Pred::ugt}, {Pred::ugt, Pred::ult}, {Pred::ule, Pred::uge}, {Pred::uge, Pred::ule},
      {Pred::eq, Pred::eq},   {Pred::ne, Pred::ne},
    };
    Pred origPred = getPredicate();
    for (auto pred : invPreds) {
      if (origPred == pred.first) {
        setPredicate(pred.second);
        Value lhs = getLhs();
        Value rhs = getRhs();
        getLhsMutable().assign(rhs);
        getRhsMutable().assign(lhs);
        return getResult();
      }
    }
    llvm_unreachable("unknown cmpi predicate kind");
  }

  if (auto lhs = dyn_cast_if_present<TypedAttr>(adaptor.getLhs())) {
    return constFoldBinaryOp<IntegerAttr>(
      adaptor.getOperands(), getI8SameShape(lhs.getType()),
      [pred = getPredicate()](const APInt& lhs, const APInt& rhs) {
        return APInt(8, static_cast<int64_t>(applyCmpPredicate(pred, lhs, rhs)));
      }
    );
  }

  return {};
}

void puarith::CmpIOp::getCanonicalizationPatterns(RewritePatternSet& patterns, MLIRContext* context)
{
  patterns.insert<CmpIExtSI, CmpIExtUI>(context);
}

bool mlir::puarith::applyCmpPredicate(puarith::CmpFPredicate predicate, const APFloat& lhs, const APFloat& rhs)
{
  auto cmpResult = lhs.compare(rhs);
  switch (predicate) {
    case puarith::CmpFPredicate::AlwaysFalse: return false;
    case puarith::CmpFPredicate::OEQ: return cmpResult == APFloat::cmpEqual;
    case puarith::CmpFPredicate::OGT: return cmpResult == APFloat::cmpGreaterThan;
    case puarith::CmpFPredicate::OGE: return cmpResult == APFloat::cmpGreaterThan || cmpResult == APFloat::cmpEqual;
    case puarith::CmpFPredicate::OLT: return cmpResult == APFloat::cmpLessThan;
    case puarith::CmpFPredicate::OLE: return cmpResult == APFloat::cmpLessThan || cmpResult == APFloat::cmpEqual;
    case puarith::CmpFPredicate::ONE: return cmpResult != APFloat::cmpUnordered && cmpResult != APFloat::cmpEqual;
    case puarith::CmpFPredicate::ORD: return cmpResult != APFloat::cmpUnordered;
    case puarith::CmpFPredicate::UEQ: return cmpResult == APFloat::cmpUnordered || cmpResult == APFloat::cmpEqual;
    case puarith::CmpFPredicate::UGT: return cmpResult == APFloat::cmpUnordered || cmpResult == APFloat::cmpGreaterThan;
    case puarith::CmpFPredicate::UGE:
      return cmpResult == APFloat::cmpUnordered || cmpResult == APFloat::cmpGreaterThan ||
             cmpResult == APFloat::cmpEqual;
    case puarith::CmpFPredicate::ULT: return cmpResult == APFloat::cmpUnordered || cmpResult == APFloat::cmpLessThan;
    case puarith::CmpFPredicate::ULE:
      return cmpResult == APFloat::cmpUnordered || cmpResult == APFloat::cmpLessThan || cmpResult == APFloat::cmpEqual;
    case puarith::CmpFPredicate::UNE: return cmpResult != APFloat::cmpEqual;
    case puarith::CmpFPredicate::UNO: return cmpResult == APFloat::cmpUnordered;
    case puarith::CmpFPredicate::AlwaysTrue: return true;
  }
  llvm_unreachable("unknown cmpf predicate kind");
}

OpFoldResult puarith::CmpFOp::fold(FoldAdaptor adaptor)
{
  auto lhs = dyn_cast_if_present<FloatAttr>(adaptor.getLhs());
  auto rhs = dyn_cast_if_present<FloatAttr>(adaptor.getRhs());

  if (lhs && lhs.getValue().isNaN()) rhs = lhs;
  if (rhs && rhs.getValue().isNaN()) lhs = rhs;

  if (!lhs || !rhs) return {};

  auto val = applyCmpPredicate(getPredicate(), lhs.getValue(), rhs.getValue());
  return BoolAttr::get(getContext(), val);
}

class CmpFIntToFPConst final : public OpRewritePattern<CmpFOp>
{
public:
  CmpFIntToFPConst(MLIRContext* context) : OpRewritePattern<CmpFOp>(context) {}

  static CmpIPredicate convertToIntegerPredicate(CmpFPredicate pred, bool isUnsigned)
  {
    using namespace puarith;
    switch (pred) {
      case CmpFPredicate::UEQ:
      case CmpFPredicate::OEQ: return CmpIPredicate::eq;
      case CmpFPredicate::UGT:
      case CmpFPredicate::OGT: return isUnsigned ? CmpIPredicate::ugt : CmpIPredicate::sgt;
      case CmpFPredicate::UGE:
      case CmpFPredicate::OGE: return isUnsigned ? CmpIPredicate::uge : CmpIPredicate::sge;
      case CmpFPredicate::ULT:
      case CmpFPredicate::OLT: return isUnsigned ? CmpIPredicate::ult : CmpIPredicate::slt;
      case CmpFPredicate::ULE:
      case CmpFPredicate::OLE: return isUnsigned ? CmpIPredicate::ule : CmpIPredicate::sle;
      case CmpFPredicate::UNE:
      case CmpFPredicate::ONE: return CmpIPredicate::ne;
      default: llvm_unreachable("Unexpected predicate!");
    }
  }

  LogicalResult matchAndRewrite(CmpFOp op, PatternRewriter& rewriter) const override
  {
    FloatAttr flt;
    if (!matchPattern(op.getRhs(), m_Constant(&flt))) return failure();

    const APFloat& rhs = flt.getValue();

    if (rhs.isNaN()) return failure();

    FloatType floatTy = llvm::cast<FloatType>(op.getRhs().getType());
    int mantissaWidth = floatTy.getFPMantissaWidth();
    if (mantissaWidth <= 0) return failure();

    bool isUnsigned;
    Value intVal;

    if (auto si = op.getLhs().getDefiningOp<SIToFPOp>()) {
      isUnsigned = false;
      intVal     = si.getIn();
    } else if (auto ui = op.getLhs().getDefiningOp<UIToFPOp>()) {
      isUnsigned = true;
      intVal     = ui.getIn();
    } else {
      return failure();
    }

    auto intTy    = llvm::cast<IntegerType>(intVal.getType());
    auto intWidth = intTy.getWidth();

    auto valueBits = isUnsigned ? intWidth : (intWidth - 1);

    if ((int)intWidth > mantissaWidth) {
      int exponent = ilogb(rhs);
      if (exponent == APFloat::IEK_Inf) {
        int maxExponent = ilogb(APFloat::getLargest(rhs.getSemantics()));
        if (maxExponent < (int)valueBits) {
          return failure();
        }
      } else {
        if (mantissaWidth <= exponent && exponent <= (int)valueBits) {
          return failure();
        }
      }
    }

    CmpIPredicate pred;
    switch (op.getPredicate()) {
      case CmpFPredicate::ORD: rewriter.replaceOpWithNewOp<ConstantIntOp>(op, true, 1); return success();
      case CmpFPredicate::UNO: rewriter.replaceOpWithNewOp<ConstantIntOp>(op, false, 1); return success();
      default: pred = convertToIntegerPredicate(op.getPredicate(), isUnsigned); break;
    }

    if (!isUnsigned) {
      APFloat signedMax(rhs.getSemantics());
      signedMax.convertFromAPInt(APInt::getSignedMaxValue(intWidth), true, APFloat::rmNearestTiesToEven);
      if (signedMax < rhs) {
        if (pred == CmpIPredicate::ne || pred == CmpIPredicate::slt || pred == CmpIPredicate::sle)
          rewriter.replaceOpWithNewOp<ConstantIntOp>(op, true, 1);
        else
          rewriter.replaceOpWithNewOp<ConstantIntOp>(op, false, 1);
        return success();
      }
    } else {
      APFloat unsignedMax(rhs.getSemantics());
      unsignedMax.convertFromAPInt(APInt::getMaxValue(intWidth), false, APFloat::rmNearestTiesToEven);
      if (unsignedMax < rhs) {
        if (pred == CmpIPredicate::ne || pred == CmpIPredicate::ult || pred == CmpIPredicate::ule)
          rewriter.replaceOpWithNewOp<ConstantIntOp>(op, true, 1);
        else
          rewriter.replaceOpWithNewOp<ConstantIntOp>(op, false, 1);
        return success();
      }
    }

    if (!isUnsigned) {
      APFloat signedMin(rhs.getSemantics());
      signedMin.convertFromAPInt(APInt::getSignedMinValue(intWidth), true, APFloat::rmNearestTiesToEven);
      if (signedMin > rhs) {
        if (pred == CmpIPredicate::ne || pred == CmpIPredicate::sgt || pred == CmpIPredicate::sge)
          rewriter.replaceOpWithNewOp<ConstantIntOp>(op, true, 1);
        else
          rewriter.replaceOpWithNewOp<ConstantIntOp>(op, false, 1);
        return success();
      }
    } else {
      APFloat unsignedMin(rhs.getSemantics());
      unsignedMin.convertFromAPInt(APInt::getMinValue(intWidth), false, APFloat::rmNearestTiesToEven);
      if (unsignedMin > rhs) {
        if (pred == CmpIPredicate::ne || pred == CmpIPredicate::ugt || pred == CmpIPredicate::uge)
          rewriter.replaceOpWithNewOp<ConstantIntOp>(op, true, 1);
        else
          rewriter.replaceOpWithNewOp<ConstantIntOp>(op, false, 1);
        return success();
      }
    }

    bool ignored;
    APSInt rhsInt(intWidth, isUnsigned);
    if (APFloat::opInvalidOp == rhs.convertToInteger(rhsInt, APFloat::rmTowardZero, &ignored)) {
      return failure();
    }

    if (!rhs.isZero()) {
      APFloat apf(floatTy.getFloatSemantics(), APInt::getZero(floatTy.getWidth()));
      apf.convertFromAPInt(rhsInt, !isUnsigned, APFloat::rmNearestTiesToEven);

      bool equal = apf == rhs;
      if (!equal) {
        switch (pred) {
          case CmpIPredicate::ne: rewriter.replaceOpWithNewOp<ConstantIntOp>(op, true, 1); return success();
          case CmpIPredicate::eq: rewriter.replaceOpWithNewOp<ConstantIntOp>(op, false, 1); return success();
          case CmpIPredicate::ule:
            if (rhs.isNegative()) {
              rewriter.replaceOpWithNewOp<ConstantIntOp>(op, false, 1);
              return success();
            }
            break;
          case CmpIPredicate::sle:
            if (rhs.isNegative()) pred = CmpIPredicate::slt;
            break;
          case CmpIPredicate::ult:
            if (rhs.isNegative()) {
              rewriter.replaceOpWithNewOp<ConstantIntOp>(op, false, 1);
              return success();
            }
            pred = CmpIPredicate::ule;
            break;
          case CmpIPredicate::slt:
            if (!rhs.isNegative()) pred = CmpIPredicate::sle;
            break;
          case CmpIPredicate::ugt:
            if (rhs.isNegative()) {
              rewriter.replaceOpWithNewOp<ConstantIntOp>(op, true, 1);
              return success();
            }
            break;
          case CmpIPredicate::sgt:
            if (rhs.isNegative()) pred = CmpIPredicate::sge;
            break;
          case CmpIPredicate::uge:
            if (rhs.isNegative()) {
              rewriter.replaceOpWithNewOp<ConstantIntOp>(op, true, 1);
              return success();
            }
            pred = CmpIPredicate::ugt;
            break;
          case CmpIPredicate::sge:
            if (!rhs.isNegative()) pred = CmpIPredicate::sgt;
            break;
        }
      }
    }

    rewriter.replaceOpWithNewOp<CmpIOp>(
      op, getI8SameShape(intVal.getType()), intVal,
      ConstantOp::create(rewriter, op.getLoc(), intVal.getType(), rewriter.getIntegerAttr(intVal.getType(), rhsInt)),
      pred
    );
    return success();
  }
};

void puarith::CmpFOp::getCanonicalizationPatterns(RewritePatternSet& patterns, MLIRContext* context) {}

struct SelectToExtUI : public OpRewritePattern<puarith::SelectOp>
{
  using Base::Base;

  LogicalResult matchAndRewrite(puarith::SelectOp op, PatternRewriter& rewriter) const override
  {
    if (!llvm::isa<IntegerType>(op.getType()) || op.getType().isInteger(1)) return failure();

    if (matchPattern(op.getTrueValue(), m_One()) && matchPattern(op.getFalseValue(), m_Zero())) {
      rewriter.replaceOpWithNewOp<puarith::ExtUIOp>(op, op.getType(), op.getCondition());
      return success();
    }

    if (matchPattern(op.getTrueValue(), m_Zero()) && matchPattern(op.getFalseValue(), m_One())) {
      rewriter.replaceOpWithNewOp<puarith::ExtUIOp>(
        op, op.getType(),
        puarith::XOrIOp::create(
          rewriter, op.getLoc(), op.getCondition(),
          puarith::ConstantIntOp::create(rewriter, op.getLoc(), op.getCondition().getType(), 1)
        )
      );
      return success();
    }

    return failure();
  }
};

void puarith::SelectOp::getCanonicalizationPatterns(RewritePatternSet& results, MLIRContext* context) {}

OpFoldResult puarith::SelectOp::fold(FoldAdaptor adaptor)
{
  Value trueVal  = getTrueValue();
  Value falseVal = getFalseValue();
  if (trueVal == falseVal) return trueVal;

  Value condition = getCondition();

  if (matchPattern(adaptor.getCondition(), m_One())) return trueVal;

  if (matchPattern(adaptor.getCondition(), m_Zero())) return falseVal;

  if (matchPattern(adaptor.getTrueValue(), ub::m_Poison())) return falseVal;

  if (matchPattern(adaptor.getFalseValue(), ub::m_Poison())) return trueVal;

  if (getType().isSignlessInteger(1) && matchPattern(adaptor.getTrueValue(), m_One()) &&
      matchPattern(adaptor.getFalseValue(), m_Zero()))
    return condition;

  if (auto cmp = condition.getDefiningOp<puarith::CmpIOp>()) {
    auto pred = cmp.getPredicate();
    if (pred == puarith::CmpIPredicate::eq || pred == puarith::CmpIPredicate::ne) {
      auto cmpLhs = cmp.getLhs();
      auto cmpRhs = cmp.getRhs();

      if ((cmpLhs == trueVal && cmpRhs == falseVal) || (cmpRhs == trueVal && cmpLhs == falseVal))
        return pred == puarith::CmpIPredicate::ne ? trueVal : falseVal;
    }
  }

  if (auto cond = dyn_cast_if_present<DenseElementsAttr>(adaptor.getCondition())) {
    if (!(cond.getType().hasStaticShape())) llvm::report_fatal_error("DenseElementsAttr must have static shape");
    if (auto lhs = dyn_cast_if_present<DenseElementsAttr>(adaptor.getTrueValue())) {
      if (auto rhs = dyn_cast_if_present<DenseElementsAttr>(adaptor.getFalseValue())) {
        SmallVector<Attribute> results;
        results.reserve(static_cast<size_t>(cond.getNumElements()));
        auto condVals = llvm::make_range(cond.value_begin<BoolAttr>(), cond.value_end<BoolAttr>());
        auto lhsVals  = llvm::make_range(lhs.value_begin<Attribute>(), lhs.value_end<Attribute>());
        auto rhsVals  = llvm::make_range(rhs.value_begin<Attribute>(), rhs.value_end<Attribute>());

        for (auto [condVal, lhsVal, rhsVal] : llvm::zip_equal(condVals, lhsVals, rhsVals))
          results.push_back(condVal.getValue() ? lhsVal : rhsVal);

        return DenseElementsAttr::get(lhs.getType(), results);
      }
    }
  }

  return nullptr;
}

ParseResult SelectOp::parse(OpAsmParser& parser, OperationState& result)
{
  Type conditionType, resultType;
  SmallVector<OpAsmParser::UnresolvedOperand, 3> operands;
  if (parser.parseOperandList(operands, 3) || parser.parseOptionalAttrDict(result.attributes) ||
      parser.parseColonType(resultType))
    return failure();

  if (succeeded(parser.parseOptionalComma())) {
    conditionType = resultType;
    if (parser.parseType(resultType)) return failure();
  } else {
    conditionType = parser.getBuilder().getI8Type();
  }

  result.addTypes(resultType);
  return parser.resolveOperands(
    operands, {conditionType, resultType, resultType}, parser.getNameLoc(), result.operands
  );
}

void puarith::SelectOp::print(OpAsmPrinter& p)
{
  p << " " << getOperands();
  p.printOptionalAttrDict((*this)->getAttrs());
  p << " : ";
  if (ShapedType condType = dyn_cast<ShapedType>(getCondition().getType())) p << condType << ", ";
  p << getType();
}

LogicalResult puarith::SelectOp::verify()
{
  Type conditionType = getCondition().getType();
  if (conditionType.isSignlessInteger(8)) return success();

  if (conditionType == getType()) return success();

  Type resultType = getType();
  if (!llvm::isa<TensorType, VectorType>(resultType))
    return emitOpError() << "expected condition to be a signless i8, but got " << conditionType;
  Type shapedConditionType = getI8SameShape(resultType);
  if (conditionType != shapedConditionType) {
    return emitOpError() << "expected condition type to have the same shape "
                            "as the result type, expected "
                         << shapedConditionType << ", but got " << conditionType;
  }
  return success();
}
OpFoldResult puarith::ShLIOp::fold(FoldAdaptor adaptor)
{
  if (matchPattern(adaptor.getRhs(), m_Zero())) return getLhs();
  if (matchPattern(adaptor.getLhs(), m_Zero())) return getLhs();
  bool bounded = false;
  auto result  = constFoldBinaryOp<IntegerAttr>(adaptor.getOperands(), [&](const APInt& a, const APInt& b) {
    bounded = b.ult(b.getBitWidth());
    return a.shl(b);
  });
  return bounded ? result : Attribute();
}

OpFoldResult puarith::ShRUIOp::fold(FoldAdaptor adaptor)
{
  if (matchPattern(adaptor.getRhs(), m_Zero())) return getLhs();
  if (matchPattern(adaptor.getLhs(), m_Zero())) return getLhs();
  if (getLhs() == getRhs()) return getIntegerAttrOfType(getType(), 0);
  bool bounded = false;
  auto result  = constFoldBinaryOp<IntegerAttr>(adaptor.getOperands(), [&](const APInt& a, const APInt& b) {
    bounded = b.ult(b.getBitWidth());
    return a.lshr(b);
  });
  return bounded ? result : Attribute();
}

OpFoldResult puarith::ShRSIOp::fold(FoldAdaptor adaptor)
{
  if (matchPattern(adaptor.getRhs(), m_Zero())) return getLhs();
  if (matchPattern(adaptor.getLhs(), m_Zero())) return getLhs();
  if (getLhs() == getRhs()) return getIntegerAttrOfType(getType(), 0);
  if (APInt val; matchPattern(adaptor.getLhs(), m_ConstantInt(&val)) && val.isAllOnes()) return getLhs();
  bool bounded = false;
  auto result  = constFoldBinaryOp<IntegerAttr>(adaptor.getOperands(), [&](const APInt& a, const APInt& b) {
    bounded = b.ult(b.getBitWidth());
    return a.ashr(b);
  });
  return bounded ? result : Attribute();
}

#define GET_OP_CLASSES
#include "privateuse/Dialect/PuArith/IR/PuArithOps.cpp.inc"

#include "privateuse/Dialect/PuArith/IR/PuArithOpsEnums.cpp.inc"
