#ifndef PRIVATEUSE_CONVERSION_TOPUC_PASSES_H
#define PRIVATEUSE_CONVERSION_TOPUC_PASSES_H

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

#include <memory>

namespace mlir {
namespace triton {

std::unique_ptr<OperationPass<ModuleOp>> createToPucPass();

#define GEN_PASS_DECL
#include "privateuse/Conversion/ToPuc/Passes.h.inc"

#define GEN_PASS_REGISTRATION
#include "privateuse/Conversion/ToPuc/Passes.h.inc"

} // namespace triton
} // namespace mlir

#endif
