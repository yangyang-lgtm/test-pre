#include "privateuse/Dialect/PuArith/IR/PuArithDialect.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/Transforms/InliningUtils.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace mlir::puarith;

#include "privateuse/Dialect/PuArith/IR/PuArithDialect.cpp.inc"
#include "privateuse/Dialect/PuArith/IR/PuArithOpsInterfaces.cpp.inc"
#define GET_ATTRDEF_CLASSES
#include "privateuse/Dialect/PuArith/IR/PuArithOpsAttributes.cpp.inc"

namespace {
struct PuArithInlinerInterface : public DialectInlinerInterface
{
  using DialectInlinerInterface::DialectInlinerInterface;

  bool isLegalToInline(Operation*, Region*, bool, IRMapping&) const final { return true; }
};
}

void puarith::PuArithDialect::initialize()
{
  addOperations<
#define GET_OP_LIST
#include "privateuse/Dialect/PuArith/IR/PuArithOps.cpp.inc"
    >();
  addAttributes<
#define GET_ATTRDEF_LIST
#include "privateuse/Dialect/PuArith/IR/PuArithOpsAttributes.cpp.inc"
    >();
  addInterfaces<PuArithInlinerInterface>();
}

Operation* puarith::PuArithDialect::materializeConstant(OpBuilder& builder, Attribute value, Type type, Location loc)
{
  return ConstantOp::materialize(builder, value, type, loc);
}
