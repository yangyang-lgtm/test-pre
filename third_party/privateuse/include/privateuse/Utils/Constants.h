#ifndef PRIVATEUSE_UTILS_CONSTANTS_H
#define PRIVATEUSE_UTILS_CONSTANTS_H

#include "llvm/ADT/StringRef.h"

#include <cstdint>

namespace mlir {
namespace triton {

inline constexpr int64_t kSramSpace        = 128;
inline constexpr int64_t kDramSpace        = 0;
inline constexpr int64_t kTrivialSpace     = -1;
inline constexpr int64_t kDefaultAlignment = 64;

inline constexpr llvm::StringLiteral kGridAttr = "privateuse.grid";
}
}

#endif
