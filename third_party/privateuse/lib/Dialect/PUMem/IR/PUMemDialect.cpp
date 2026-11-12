#include "privateuse/Dialect/PUMem/IR/PUMemDialect.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpImplementation.h"

#define GET_DIALECT_DEFS
#include "privateuse/Dialect/PUMem/IR/PUMemDialect.cpp.inc"

#define GET_OP_CLASSES
#include "privateuse/Dialect/PUMem/IR/PUMemOps.cpp.inc"

using namespace mlir;

void mlir::pumem::PUMemDialect::initialize()
{
  addOperations<
#define GET_OP_LIST
#include "privateuse/Dialect/PUMem/IR/PUMemOps.cpp.inc"
    >();
}

LogicalResult mlir::pumem::GatherOp::verify()
{
  auto source      = dyn_cast<MemRefType>(getSource().getType());
  auto indices     = dyn_cast<MemRefType>(getIndices().getType());
  auto destination = dyn_cast<MemRefType>(getDestination().getType());
  if (!source || !indices || !destination) return emitOpError("requires ranked memref operands");
  if (source.getRank() != 1) return emitOpError("source must be a rank-1 memref");
  if (!indices.getElementType().isIntOrIndex()) return emitOpError("indices must have integer element type");
  if (indices.getShape() != destination.getShape()) return emitOpError("indices and destination shapes must match");
  if (source.getElementType() != destination.getElementType())
    return emitOpError("source and destination element types must match");
  return success();
}

LogicalResult mlir::pumem::ScatterOp::verify()
{
  auto source      = dyn_cast<MemRefType>(getSource().getType());
  auto indices     = dyn_cast<MemRefType>(getIndices().getType());
  auto destination = dyn_cast<MemRefType>(getDestination().getType());
  if (!source || !indices || !destination) return emitOpError("requires ranked memref operands");
  if (destination.getRank() != 1) return emitOpError("destination must be a rank-1 memref");
  if (!indices.getElementType().isIntOrIndex()) return emitOpError("indices must have integer element type");
  if (indices.getShape() != source.getShape()) return emitOpError("indices and source shapes must match");
  if (source.getElementType() != destination.getElementType())
    return emitOpError("source and destination element types must match");
  return success();
}
