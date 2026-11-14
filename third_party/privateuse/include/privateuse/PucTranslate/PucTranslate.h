#ifndef PRIVATEUSE_PUCTRANSLATE_PUCTRANSLATE_H
#define PRIVATEUSE_PUCTRANSLATE_PUCTRANSLATE_H

#include "mlir/Support/LogicalResult.h"
#include "llvm/Support/raw_ostream.h"

#include <cstdint>
#include <string>

namespace mlir {
class Operation;

namespace puc {

LogicalResult
translatePucToCpp(Operation* op, llvm::raw_ostream& os, llvm::StringRef backend = "cpu", int64_t numPids = 32);

bool translatePucToCppString(Operation* op, std::string& out, llvm::StringRef backend = "cpu", int64_t numPids = 32);

} // namespace puc
} // namespace mlir

#endif
