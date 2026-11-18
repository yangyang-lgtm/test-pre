#ifndef PRIVATEUSE_TRANSFORM_OUTLINETODISPATCH_PASSES_H
#define PRIVATEUSE_TRANSFORM_OUTLINETODISPATCH_PASSES_H

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

#include <memory>
#include <string>

namespace mlir {
namespace triton {

std::unique_ptr<OperationPass<ModuleOp>> createOutlineToDispatchPass(std::string outlinedAttrValue = "");

#define GEN_PASS_DECL
#include "privateuse/Transform/OutlineToDispatch/Passes.h.inc"

#define GEN_PASS_REGISTRATION
#include "privateuse/Transform/OutlineToDispatch/Passes.h.inc"

} // namespace triton
} // namespace mlir

#endif
