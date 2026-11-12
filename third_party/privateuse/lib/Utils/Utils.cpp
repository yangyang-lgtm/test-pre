#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "triton/Dialect/Triton/IR/Dialect.h"

#include "llvm/ADT/STLExtras.h"

namespace mlir {
namespace triton {
bool isPtrTypeLike(Type t)
{
  if (auto tensorType = dyn_cast<RankedTensorType>(t)) {
    return isa<triton::PointerType>(tensorType.getElementType());
  }
  return isa<triton::PointerType>(t);
}

Type storageElementType(Type element)
{
  if (element.isInteger(1)) return IntegerType::get(element.getContext(), 8);
  return element;
}

Value maskedFillValue(LoadOp load, Type elementType, OpBuilder& builder, Location loc)
{
  Value other = load.getOther();
  if (!other) return arith::ConstantOp::create(builder, loc, elementType, builder.getZeroAttr(elementType));
  if (auto splat = other.getDefiningOp<SplatOp>()) return splat.getSrc();
  if (auto splat = other.getDefiningOp<tensor::SplatOp>()) return splat.getInput();
  if (auto cst = other.getDefiningOp<arith::ConstantOp>())
    if (auto dense = dyn_cast<DenseElementsAttr>(cst.getValue()); dense && dense.isSplat())
      return arith::ConstantOp::create(builder, loc, elementType, dense.getSplatValue<TypedAttr>());
  load.emitError("masked load with non-splat 'other' is unsupported");
  return {};
}

Value createBufferWindow(OpBuilder& builder, Location loc, Value buf, ArrayRef<int64_t> shape, ArrayRef<Value> dimSizes)
{
  SmallVector<OpFoldResult> offsets;
  SmallVector<OpFoldResult> sizes;
  SmallVector<OpFoldResult> strides;
  for (auto [d, dimSize] : llvm::enumerate(shape)) {
    offsets.push_back(builder.getIndexAttr(0));
    sizes.push_back(dimSizes[d] ? OpFoldResult(dimSizes[d]) : OpFoldResult(builder.getIndexAttr(dimSize)));
    strides.push_back(builder.getIndexAttr(1));
  }
  return memref::SubViewOp::create(builder, loc, buf, offsets, sizes, strides);
}
}

}
