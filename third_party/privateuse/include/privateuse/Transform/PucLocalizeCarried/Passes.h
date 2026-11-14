#ifndef PUC_LOCALIZE_CARRIED_TRANSFORM_PASSES_H
#define PUC_LOCALIZE_CARRIED_TRANSFORM_PASSES_H

#include "privateuse/Transform/PucLocalizeCarried/PucLocalizeCarried.h"

namespace mlir {
namespace triton {

#define GEN_PASS_REGISTRATION
#include "privateuse/Transform/PucLocalizeCarried/Passes.h.inc"

}
}

#endif
