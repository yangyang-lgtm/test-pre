#ifndef PRIVATEUSE_CONVERSION_SCFTOPUMEM_PASSES_H
#define PRIVATEUSE_CONVERSION_SCFTOPUMEM_PASSES_H

#include "privateuse/Conversion/ScfToPumem/ScfToPumem.h"

namespace mlir {
namespace triton {

#define GEN_PASS_REGISTRATION
#include "privateuse/Conversion/ScfToPumem/Passes.h.inc"

}
}

#endif
