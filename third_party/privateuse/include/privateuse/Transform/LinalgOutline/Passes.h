#ifndef PRIVATEUSE_TRANSFORM_LINALGOUTLINE_PASSES_H
#define PRIVATEUSE_TRANSFORM_LINALGOUTLINE_PASSES_H

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

#include <memory>
#include <string>

namespace mlir {
namespace triton {

std::unique_ptr<OperationPass<ModuleOp>> createLinalgOutlinePass();
std::unique_ptr<OperationPass<ModuleOp>> createLinalgOutlinePass(std::string outlinedAttr);

#define GEN_PASS_REGISTRATION
#include "privateuse/Transform/LinalgOutline/Passes.h.inc"

} // namespace triton
} // namespace mlir

#endif
