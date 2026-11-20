#ifndef PRIVATEUSE_CONVERSION_PUSTREAMTOPUMEM_PASSES_H
#define PRIVATEUSE_CONVERSION_PUSTREAMTOPUMEM_PASSES_H

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

#include <memory>

namespace mlir {
namespace triton {

std::unique_ptr<OperationPass<ModuleOp>> createPustreamToPumemPass();

#define GEN_PASS_DECL
#include "privateuse/Conversion/PustreamToPumem/Passes.h.inc"

#define GEN_PASS_REGISTRATION
#include "privateuse/Conversion/PustreamToPumem/Passes.h.inc"

} // namespace triton
} // namespace mlir

#endif
