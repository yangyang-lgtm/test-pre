#ifndef TRITON_TRANSFORM_SPECIALIZEKERNEL_SPECIALIZEKERNEL_H
#define TRITON_TRANSFORM_SPECIALIZEKERNEL_SPECIALIZEKERNEL_H

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

#include <cstdint>
#include <utility>
#include <variant>
#include <vector>

namespace mlir {
namespace triton {

using SpecializeConstant = std::pair<int64_t, std::variant<int64_t, double>>;

std::unique_ptr<OperationPass<ModuleOp>>
createSpecializeKernelPass(std::vector<SpecializeConstant> constants = {}, std::vector<int64_t> grid = {});

}
}

#endif
