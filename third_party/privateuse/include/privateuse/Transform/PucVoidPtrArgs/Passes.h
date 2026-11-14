#ifndef PUC_VOID_PTR_ARGS_TRANSFORM_PASSES_H
#define PUC_VOID_PTR_ARGS_TRANSFORM_PASSES_H

#include "privateuse/Transform/PucVoidPtrArgs/PucVoidPtrArgs.h"

namespace mlir {
namespace triton {

#define GEN_PASS_REGISTRATION
#include "privateuse/Transform/PucVoidPtrArgs/Passes.h.inc"

}
}

#endif
