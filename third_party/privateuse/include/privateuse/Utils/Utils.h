#ifndef PRIVATEUSE_UTILITY_H
#define PRIVATEUSE_UTILITY_H

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Operation.h"
#include "privateuse/Utils/Constants.h"

namespace mlir {
namespace triton {
bool isPtrTypeLike(Type t);

}

}

#endif
