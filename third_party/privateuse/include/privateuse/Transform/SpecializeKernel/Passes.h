#ifndef SPECIALIZE_KERNEL_TRANSFORM_PASSES_H
#define SPECIALIZE_KERNEL_TRANSFORM_PASSES_H

#include "privateuse/Transform/SpecializeKernel/SpecializeKernel.h"

namespace mlir {
namespace triton {

#define GEN_PASS_REGISTRATION
#include "privateuse/Transform/SpecializeKernel/Passes.h.inc"

}
}

#endif
