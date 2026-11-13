#ifndef PRIVATEUSE_CONVERSION_ARITHTOLINALG_CONVERSIONTOOLS_H
#define PRIVATEUSE_CONVERSION_ARITHTOLINALG_CONVERSIONTOOLS_H

#include "mlir/Dialect/Utils/StructuredOpsUtils.h"
#include "mlir/IR/AffineMap.h"
#include "llvm/Support/ErrorHandling.h"

namespace mlir {
namespace triton {

static inline SmallVector<utils::IteratorType> getNParallelLoopsAttrs(unsigned n)
{
  return SmallVector<utils::IteratorType>(n, utils::IteratorType::parallel);
}

static inline AffineMap
getBroadcastAffineMap(MLIRContext* context, ArrayRef<int64_t> inputShape, ArrayRef<int64_t> broadcastToShape)
{
  if (!(broadcastToShape.size() >= inputShape.size()))
    llvm::report_fatal_error("broadcastToShape.size() >= inputShape.size()");

  SmallVector<AffineExpr> outExpr;

  size_t diff = broadcastToShape.size() - inputShape.size();
  for (size_t i = 0; i < broadcastToShape.size(); i++) {
    if (i < diff) {
      continue;
    }
    size_t j = i - diff;
    if (inputShape[j] == 1) {
      outExpr.push_back(mlir::getAffineConstantExpr(0, context));
      continue;
    }
    outExpr.push_back(mlir::getAffineDimExpr(i, context));
  }
  return AffineMap::get(broadcastToShape.size(), 0, outExpr, context);
}

} // namespace triton
} // namespace mlir

#endif
