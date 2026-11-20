#ifndef PRIVATEUSE_CONVERSION_COMMON_MEMREFPARTS_H
#define PRIVATEUSE_CONVERSION_COMMON_MEMREFPARTS_H

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Interfaces/DataLayoutInterfaces.h"
#include "privateuse/Conversion/Common/I1ToI8.h"
#include "privateuse/Dialect/PUMem/IR/PUMemDialect.h"
#include "privateuse/Utils/Constants.h"

namespace mlir::pumem {

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

struct MemRefParts
{
  Value base;
  Value offset;
  SmallVector<Value> sizes;
  SmallVector<Value> strides;

  static unsigned countFor(MemRefType type) { return 2 + 2 * type.getRank(); }

  static FailureOr<SmallVector<Type>> typesFor(MemRefType type, MLIRContext* context)
  {
    FailureOr<int64_t> space = getMemRefSpace(type);
    if (failed(space)) return failure();
    SmallVector<Type> types;
    types.push_back(PtrType::get(context, triton::i1ToI8(type.getElementType()), *space));
    types.append(1 + 2 * type.getRank(), IndexType::get(context));
    return types;
  }

  static FailureOr<MemRefParts> unpack(ValueRange flat, MemRefType type, Operation* onError)
  {
    if (flat.size() != countFor(type)) {
      onError->emitOpError() << "expected " << countFor(type) << " values for " << type << ", got " << flat.size();
      return failure();
    }
    MemRefParts parts;
    unsigned rank = type.getRank();
    parts.base    = flat[0];
    parts.offset  = flat[1];
    parts.sizes.append(flat.begin() + 2, flat.begin() + 2 + rank);
    parts.strides.append(flat.begin() + 2 + rank, flat.end());
    return parts;
  }

  SmallVector<Value> pack() const
  {
    SmallVector<Value> flat{base, offset};
    flat.append(sizes.begin(), sizes.end());
    flat.append(strides.begin(), strides.end());
    return flat;
  }

  FailureOr<Value> elementPtr(OpBuilder& builder, Location loc, ValueRange indices = {}) const
  {
    if (!indices.empty() && indices.size() != strides.size()) {
      emitError(loc) << "expected " << strides.size() << " indices for this memref, got " << indices.size();
      return failure();
    }
    Value combined = offset;
    if (!indices.empty())
      for (auto [index, stride] : llvm::zip_equal(indices, strides))
        combined = arith::AddIOp::create(builder, loc, combined, arith::MulIOp::create(builder, loc, index, stride));
    if (indices.empty()) {
      if (auto constant = getConstantIntValue(combined); constant && *constant == 0) return base;
    }
    return Value(AddPtrOp::create(builder, loc, base.getType(), base, combined));
  }
};

struct ViewDescriptor
{
  SmallVector<Value> sizes;
  SmallVector<Value> strides;

  static ViewDescriptor read(OpBuilder& builder, Location loc, Value view)
  {
    auto metadata = memref::ExtractStridedMetadataOp::create(builder, loc, view);
    ViewDescriptor descriptor;
    descriptor.sizes.assign(metadata.getSizes().begin(), metadata.getSizes().end());
    descriptor.strides.assign(metadata.getStrides().begin(), metadata.getStrides().end());
    return descriptor;
  }
};

} // namespace mlir::pumem

#endif
