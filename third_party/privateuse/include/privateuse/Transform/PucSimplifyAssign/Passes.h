#ifndef PUC_SIMPLIFY_ASSIGN_TRANSFORM_PASSES_H
#define PUC_SIMPLIFY_ASSIGN_TRANSFORM_PASSES_H

#include "privateuse/Transform/PucSimplifyAssign/PucSimplifyAssign.h"

namespace mlir {
namespace triton {

#define GEN_PASS_REGISTRATION
#include "privateuse/Transform/PucSimplifyAssign/Passes.h.inc"

}
}

#endif
