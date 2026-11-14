#ifndef PRIVATEUSE_TRANSFORM_PUCLAUNCHKERNEL_PASSES_H
#define PRIVATEUSE_TRANSFORM_PUCLAUNCHKERNEL_PASSES_H

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

#include <memory>
#include <string>

namespace mlir {
namespace triton {

std::unique_ptr<OperationPass<ModuleOp>>
createPucLaunchKernelPass(std::string launcherAttrValue = "extern \"C\"", std::string kernelAttrValue = "GLOBAL");

#define GEN_PASS_REGISTRATION
#include "privateuse/Transform/PucLaunchKernel/Passes.h.inc"

} // namespace triton
} // namespace mlir

#endif
