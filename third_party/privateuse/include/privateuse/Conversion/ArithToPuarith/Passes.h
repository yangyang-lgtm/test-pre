#ifndef PRIVATEUSE_CONVERSION_ARITHTOPUARITH_PASSES_H
#define PRIVATEUSE_CONVERSION_ARITHTOPUARITH_PASSES_H

#include "privateuse/Conversion/ArithToPuarith/ArithToPuarith.h"

namespace mlir {
namespace triton {

#define GEN_PASS_REGISTRATION
#include "privateuse/Conversion/ArithToPuarith/Passes.h.inc"

}
}

#endif
