#ifndef PUC_REPEAT_TRANSFORM_PASSES_H
#define PUC_REPEAT_TRANSFORM_PASSES_H

#include "privateuse/Transform/PucRepeat/PucRepeat.h"

namespace mlir {
namespace triton {

#define GEN_PASS_REGISTRATION
#include "privateuse/Transform/PucRepeat/Passes.h.inc"

}
}

#endif
