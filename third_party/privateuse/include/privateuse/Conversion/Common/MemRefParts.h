#ifndef PRIVATEUSE_CONVERSION_COMMON_MEMREFPARTS_H
#define PRIVATEUSE_CONVERSION_COMMON_MEMREFPARTS_H

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Interfaces/DataLayoutInterfaces.h"
#include "privateuse/Dialect/PUMem/IR/PUMemDialect.h"
#include "privateuse/Utils/Constants.h"

namespace mlir::pumem {

// A memref once converted: a pointer, an element offset, and one size and one
// stride per dimension. Every conversion in the tree agrees on this shape, so it
// is spelled once here.
struct MemRefParts
{
  Value base;
  Value offset;
  SmallVector<Value> sizes;
  SmallVector<Value> strides;
};

inline MemRefParts unpackMemRef(ValueRange flat, MemRefType type)
{
  MemRefParts parts;
  unsigned rank = type.getRank();
  parts.base    = flat[0];
  parts.offset  = flat[1];
  parts.sizes.append(flat.begin() + 2, flat.begin() + 2 + rank);
  parts.strides.append(flat.begin() + 2 + rank, flat.end());
  return parts;
}

inline SmallVector<Value> packMemRef(const MemRefParts& parts)
{
  SmallVector<Value> flat{parts.base, parts.offset};
  flat.append(parts.sizes.begin(), parts.sizes.end());
  flat.append(parts.strides.begin(), parts.strides.end());
  return flat;
}

inline Value constantIndex(OpBuilder& builder, Location loc, int64_t value)
{
  return arith::ConstantIndexOp::create(builder, loc, value);
}

inline FailureOr<int64_t> getMemRefSpace(MemRefType type)
{
  Attribute space = type.getMemorySpace();
  if (!space) return triton::kDramSpace;
  if (auto intSpace = dyn_cast<IntegerAttr>(space)) return intSpace.getInt();
  return failure();
}

inline FailureOr<int64_t> getElementByteSize(Type type, Operation* op)
{
  auto module = op->getParentOfType<ModuleOp>();
  if (!module) return failure();
  llvm::TypeSize size = DataLayout(module).getTypeSizeInBits(type);
  if (size.isScalable() || size.getFixedValue() == 0) return failure();
  return (size.getFixedValue() + 7) / 8;
}

// A pointer to the first element of the box, with the offset folded in.
inline Value materializeElementPtr(OpBuilder& builder, Location loc, const MemRefParts& parts, ValueRange indices)
{
  assert((indices.empty() || indices.size() == parts.strides.size()) && "index count must match the rank");
  Value offset = parts.offset;
  if (!indices.empty())
    for (auto [index, stride] : llvm::zip_equal(indices, parts.strides))
      offset = arith::AddIOp::create(builder, loc, offset, arith::MulIOp::create(builder, loc, index, stride));
  if (indices.empty()) {
    if (auto constant = getConstantIntValue(offset); constant && *constant == 0) return parts.base;
  }
  return AddPtrOp::create(builder, loc, parts.base.getType(), parts.base, offset);
}

} // namespace mlir::pumem

#endif
