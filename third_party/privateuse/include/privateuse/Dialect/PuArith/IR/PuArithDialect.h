#ifndef PRIVATEUSE_DIALECT_PUARTH_IR_PUARTH_DIALECT_H_
#define PRIVATEUSE_DIALECT_PUARTH_IR_PUARTH_DIALECT_H_

#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Interfaces/CastInterfaces.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Interfaces/VectorInterfaces.h"
#include "llvm/ADT/StringExtras.h"

#include "privateuse/Dialect/PuArith/IR/PuArithDialect.h.inc"

#include "privateuse/Dialect/PuArith/IR/PuArithOpsEnums.h.inc"
#define GET_ATTRDEF_CLASSES
#include "privateuse/Dialect/PuArith/IR/PuArithOpsAttributes.h.inc"

#include "privateuse/Dialect/PuArith/IR/PuArithOpsInterfaces.h.inc"

namespace mlir {
namespace puarith {

inline Type getI8SameShape(Type type)
{
  auto i8Type = IntegerType::get(type.getContext(), 8);
  if (auto shapedType = dyn_cast<ShapedType>(type)) return shapedType.cloneWith(std::nullopt, i8Type);
  if (llvm::isa<UnrankedTensorType>(type)) return UnrankedTensorType::get(i8Type);
  return i8Type;
}

}
}

#define GET_OP_CLASSES
#include "privateuse/Dialect/PuArith/IR/PuArithOps.h.inc"

namespace mlir {
namespace puarith {

class ConstantIntOp : public puarith::ConstantOp
{
public:
  using puarith::ConstantOp::ConstantOp;
  static ::mlir::TypeID resolveTypeID() { return TypeID::get<ConstantOp>(); }

  static void build(OpBuilder& builder, OperationState& result, int64_t value, unsigned width);
  static ConstantIntOp create(OpBuilder& builder, Location location, int64_t value, unsigned width);
  static ConstantIntOp create(ImplicitLocOpBuilder& builder, int64_t value, unsigned width);

  static void build(OpBuilder& builder, OperationState& result, Type type, int64_t value);
  static ConstantIntOp create(OpBuilder& builder, Location location, Type type, int64_t value);
  static ConstantIntOp create(ImplicitLocOpBuilder& builder, Type type, int64_t value);

  static void build(OpBuilder& builder, OperationState& result, Type type, const APInt& value);
  static ConstantIntOp create(OpBuilder& builder, Location location, Type type, const APInt& value);
  static ConstantIntOp create(ImplicitLocOpBuilder& builder, Type type, const APInt& value);

  inline int64_t value() { return cast<IntegerAttr>(puarith::ConstantOp::getValue()).getInt(); }

  static bool classof(Operation* op);
};

class ConstantFloatOp : public puarith::ConstantOp
{
public:
  using puarith::ConstantOp::ConstantOp;
  static ::mlir::TypeID resolveTypeID() { return TypeID::get<ConstantOp>(); }

  static void build(OpBuilder& builder, OperationState& result, FloatType type, const APFloat& value);
  static ConstantFloatOp create(OpBuilder& builder, Location location, FloatType type, const APFloat& value);
  static ConstantFloatOp create(ImplicitLocOpBuilder& builder, FloatType type, const APFloat& value);

  inline APFloat value() { return cast<FloatAttr>(puarith::ConstantOp::getValue()).getValue(); }

  static bool classof(Operation* op);
};

class ConstantIndexOp : public puarith::ConstantOp
{
public:
  using puarith::ConstantOp::ConstantOp;
  static ::mlir::TypeID resolveTypeID() { return TypeID::get<ConstantOp>(); }
  static void build(OpBuilder& builder, OperationState& result, int64_t value);
  static ConstantIndexOp create(OpBuilder& builder, Location location, int64_t value);
  static ConstantIndexOp create(ImplicitLocOpBuilder& builder, int64_t value);

  inline int64_t value() { return cast<IntegerAttr>(puarith::ConstantOp::getValue()).getInt(); }

  static bool classof(Operation* op);
};

}
}

namespace mlir {
namespace puarith {

bool applyCmpPredicate(puarith::CmpIPredicate predicate, const APInt& lhs, const APInt& rhs);

bool applyCmpPredicate(puarith::CmpFPredicate predicate, const APFloat& lhs, const APFloat& rhs);

puarith::CmpIPredicate invertPredicate(puarith::CmpIPredicate pred);

Value getZeroConstant(OpBuilder& builder, Location loc, Type type);

}
}

#endif
