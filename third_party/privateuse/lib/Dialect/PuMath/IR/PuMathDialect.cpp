#include "privateuse/Dialect/PuMath/IR/PuMathDialect.h"

#include "mlir/Transforms/InliningUtils.h"

using namespace mlir;
using namespace mlir::pumath;

#include "privateuse/Dialect/PuMath/IR/PuMathDialect.cpp.inc"

namespace {
struct PuMathInlinerInterface : public DialectInlinerInterface
{
  using DialectInlinerInterface::DialectInlinerInterface;

  bool isLegalToInline(Operation*, Region*, bool, IRMapping&) const final { return true; }
};
}

void mlir::pumath::PuMathDialect::initialize()
{
  addOperations<
#define GET_OP_LIST
#include "privateuse/Dialect/PuMath/IR/PuMathOps.cpp.inc"
    >();
  addInterfaces<PuMathInlinerInterface>();
}
