#ifndef PRIVATEUSE_CONVERSION_MATHTOPUMATH_PASSES_H
#define PRIVATEUSE_CONVERSION_MATHTOPUMATH_PASSES_H

#include "privateuse/Conversion/MathToPumath/MathToPumath.h"

namespace mlir {
namespace triton {

#define GEN_PASS_REGISTRATION
#include "privateuse/Conversion/MathToPumath/Passes.h.inc"

}
}

#endif
