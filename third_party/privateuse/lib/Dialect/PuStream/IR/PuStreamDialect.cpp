#include "privateuse/Dialect/PuStream/IR/PuStreamDialect.h"

#include "llvm/ADT/TypeSwitch.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpImplementation.h"

#define GET_DIALECT_DEFS
#include "privateuse/Dialect/PuStream/IR/PuStreamDialect.cpp.inc"

#define GET_TYPEDEF_CLASSES
#include "privateuse/Dialect/PuStream/IR/PuStreamTypes.cpp.inc"

#define GET_OP_CLASSES
#include "privateuse/Dialect/PuStream/IR/PuStreamOps.cpp.inc"

using namespace mlir;
using namespace mlir::pustream;

void mlir::pustream::PuStreamDialect::initialize()
{
  addTypes<
#define GET_TYPEDEF_LIST
#include "privateuse/Dialect/PuStream/IR/PuStreamTypes.cpp.inc"
    >();
  addOperations<
#define GET_OP_LIST
#include "privateuse/Dialect/PuStream/IR/PuStreamOps.cpp.inc"
    >();
}

LogicalResult mlir::pustream::AsyncCopyOp::verify()
{
  auto sourceType = dyn_cast<MemRefType>(getSource().getType());
  if (!sourceType) return emitOpError("source must be a ranked memref");
  auto targetType = dyn_cast<MemRefType>(getTarget().getType());
  if (!targetType) return emitOpError("target must be a ranked memref");

  if (sourceType.getElementType() != targetType.getElementType())
    return emitOpError("source and target element types must match");
  if (sourceType.getMemorySpace() == targetType.getMemorySpace())
    return emitOpError("requires source and target in different memory spaces");
  return success();
}

LogicalResult mlir::pustream::AsyncDispatchOp::verify()
{
  int64_t numArgs = static_cast<int64_t>(getArgs().size());
  auto verifyIndex = [&](int64_t index, StringRef list) -> LogicalResult {
    if (index < 0 || index >= numArgs)
      return emitOpError() << list << " index " << index << " is out of range";
    if (!isa<MemRefType>(getArgs()[index].getType()))
      return emitOpError() << list << " index " << index << " does not refer to a memref argument";
    return success();
  };
  for (int64_t index : getReads())
    if (failed(verifyIndex(index, "reads"))) return failure();
  for (int64_t index : getWrites())
    if (failed(verifyIndex(index, "writes"))) return failure();
  return success();
}