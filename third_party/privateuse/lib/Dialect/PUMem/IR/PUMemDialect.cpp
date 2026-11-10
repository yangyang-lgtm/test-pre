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
