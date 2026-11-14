#ifndef PRIVATEUSE_CONVERSION_TOPUC_PUCPATTERNS_H
#define PRIVATEUSE_CONVERSION_TOPUC_PUCPATTERNS_H

#include "mlir/Transforms/DialectConversion.h"

namespace mlir {
namespace triton {

void populatePumemToPucPatterns(RewritePatternSet& patterns, TypeConverter& typeConverter);

void populateScalarToPucPatterns(RewritePatternSet& patterns, TypeConverter& typeConverter);

void populateFuncToPucPatterns(RewritePatternSet& patterns, TypeConverter& typeConverter);

} // namespace triton
} // namespace mlir

#endif
