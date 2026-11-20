#include "privateuse/Conversion/MemrefToPumem/Passes.h"

#include "llvm/ADT/BitVector.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Func/Transforms/FuncConversions.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/DataLayoutInterfaces.h"
#include "mlir/Transforms/DialectConversion.h"
#include "privateuse/Conversion/Common/I1ToI8.h"
#include "privateuse/Conversion/Common/MemRefParts.h"
#include "privateuse/Dialect/PUMem/IR/PUMemDialect.h"
#include "privateuse/Utils/Constants.h"
#include "triton/Dialect/Triton/IR/Dialect.h"

#define GEN_PASS_DEF_MEMREFTOPUMEM
#include "privateuse/Conversion/MemrefToPumem/Passes.h.inc"

using namespace mlir;
using namespace mlir::pumem;

namespace {

bool containsMemRefType(Type type)
{
  return isa<BaseMemRefType>(type);
}

bool isOwnedAllocation(Value pointer)
{
  Operation* definingOp = pointer.getDefiningOp();
  if (auto cast = dyn_cast_or_null<PointCastOp>(definingOp)) definingOp = cast.getPtr().getDefiningOp();
  return definingOp && isa<AllocOp>(definingOp);
}

bool hasMemRefTypes(Operation* op)
{
  if (llvm::any_of(op->getOperandTypes(), containsMemRefType) || llvm::any_of(op->getResultTypes(), containsMemRefType))
    return true;
  for (Region& region : op->getRegions())
    for (Block& block : region)
      for (BlockArgument argument : block.getArguments())
        if (containsMemRefType(argument.getType())) return true;
  if (auto function = dyn_cast<FunctionOpInterface>(op)) {
    auto type = dyn_cast<FunctionType>(function.getFunctionType());
    if (!type) return true;
    if (llvm::any_of(type.getInputs(), containsMemRefType) || llvm::any_of(type.getResults(), containsMemRefType))
      return true;
  }
  return false;
}

SmallVector<Value> flattenValues(ArrayRef<ValueRange> values)
{
  SmallVector<Value> flat;
  for (ValueRange value : values) flat.append(value.begin(), value.end());
  return flat;
}

SmallVector<Value>
materializeMixedIndexList(OpBuilder& builder, Location loc, ArrayRef<OpFoldResult> mixed, ArrayRef<ValueRange> dynamics)
{
  SmallVector<Value> values;
  unsigned dynamicIdx = 0;
  for (OpFoldResult ofr : mixed) {
    if (Attribute attr = dyn_cast<Attribute>(ofr)) {
      values.push_back(constantIndex(builder, loc, cast<IntegerAttr>(attr).getInt()));
    } else {
      assert(dynamicIdx < dynamics.size() && "missing dynamic operand");
      ValueRange range = dynamics[dynamicIdx++];
      assert(range.size() == 1 && "index operands convert one-to-one");
      values.push_back(range.front());
    }
  }
  assert(dynamicIdx == dynamics.size() && "unused dynamic operand");
  return values;
}

SmallVector<Attribute> expandAttrArray(ArrayAttr attrs, TypeRange types, const TypeConverter& converter)
{
  SmallVector<Attribute> expanded;
  if (!attrs) return expanded;
  for (auto [attr, type] : llvm::zip_equal(attrs.getValue(), types)) {
    SmallVector<Type> converted;
    if (failed(converter.convertTypes(type, converted))) return {};
    expanded.append(converted.size(), attr);
  }
  return expanded;
}

LogicalResult getConvertedCounts(TypeRange types, const TypeConverter& converter, SmallVectorImpl<size_t>& counts)
{
  for (Type type : types) {
    SmallVector<Type> converted;
    if (failed(converter.convertTypes(type, converted))) return failure();
    counts.push_back(converted.size());
  }
  return success();
}

LogicalResult checkPreconditions(ModuleOp module)
{
  WalkResult result = module.walk([&](Operation* op) -> WalkResult {
    auto check = [&](Type type) -> LogicalResult {
      if (isa<UnrankedMemRefType>(type)) return op->emitError("unranked memref types are not supported");
      auto ranked = dyn_cast<MemRefType>(type);
      if (!ranked) return success();
      if (failed(getMemRefSpace(ranked))) return op->emitError("memref memory space must be an integer attribute");
      SmallVector<int64_t> strides;
      int64_t offset = 0;
      if (failed(ranked.getStridesAndOffset(strides, offset)))
        return op->emitError("memref layout is not strided; run -expand-strided-metadata first");
      return success();
    };
    for (Type type : op->getOperandTypes())
      if (failed(check(type))) return WalkResult::interrupt();
    for (Type type : op->getResultTypes())
      if (failed(check(type))) return WalkResult::interrupt();
    for (Region& region : op->getRegions())
      for (Block& block : region)
        for (BlockArgument argument : block.getArguments())
          if (failed(check(argument.getType()))) return WalkResult::interrupt();
    if (auto function = dyn_cast<FunctionOpInterface>(op)) {
      if (auto type = dyn_cast<FunctionType>(function.getFunctionType())) {
        for (Type input : type.getInputs())
          if (failed(check(input))) return WalkResult::interrupt();
        for (Type output : type.getResults())
          if (failed(check(output))) return WalkResult::interrupt();
      }
    }
    return WalkResult::advance();
  });
  return failure(result.wasInterrupted());
}

class MemRefTupleTypeConverter : public TypeConverter
{
public:
  MemRefTupleTypeConverter()
  {
    addConversion([](Type type) { return type; });
    addConversion([](MemRefType type, SmallVectorImpl<Type>& results) -> LogicalResult {
      FailureOr<SmallVector<Type>> types = MemRefParts::typesFor(type, type.getContext());
      if (failed(types)) return failure();
      results.append(types->begin(), types->end());
      return success();
    });
  }
};

class AllocOpLowering : public OpConversionPattern<memref::AllocOp>
{
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(memref::AllocOp op, OneToNOpAdaptor adaptor, ConversionPatternRewriter& rewriter) const override
  {
    Location loc    = op.getLoc();
    MemRefType type = op.getType();

    SmallVector<int64_t> staticStrides;
    int64_t staticOffset = 0;
    if (failed(type.getStridesAndOffset(staticStrides, staticOffset)) || staticOffset == ShapedType::kDynamic)
      return op.emitError("cannot lower an allocation whose layout is not strided or whose offset is dynamic");

    if (ShapedType::isDynamicShape(type.getShape()))
      return op.emitError("cannot lower an allocation with a dynamic shape");

    SmallVector<Value> sizes;
    for (int64_t dim : type.getShape()) sizes.push_back(constantIndex(rewriter, loc, dim));

    SmallVector<Value> strides;
    for (int64_t stride : staticStrides) {
      if (stride == ShapedType::kDynamic) return op.emitError("cannot lower an allocation with dynamic strides");
      strides.push_back(constantIndex(rewriter, loc, stride));
    }

    FailureOr<int64_t> elementBytes = getElementByteSize(type.getElementType(), op);
    if (failed(elementBytes)) return op.emitError("cannot determine the byte size of the element type");
    FailureOr<int64_t> space = getMemRefSpace(type);
    if (failed(space)) return op.emitError("cannot lower an allocation with a non-integer memory space");

    if (llvm::any_of(type.getShape(), [](int64_t dim) { return dim == 0; }))
      return op.emitError("cannot lower an allocation with a zero-sized dimension");

    Value elements = constantIndex(rewriter, loc, 1 + staticOffset);
    for (auto [size, stride] : llvm::zip_equal(sizes, strides)) {
      Value extent = arith::SubIOp::create(rewriter, loc, size, constantIndex(rewriter, loc, 1));
      elements = arith::AddIOp::create(rewriter, loc, elements, arith::MulIOp::create(rewriter, loc, extent, stride));
    }
    elements = arith::MulIOp::create(rewriter, loc, elements, constantIndex(rewriter, loc, *elementBytes));

    int64_t alignment = op.getAlignment().value_or(triton::kDefaultAlignment);
    auto voidPtrType  = PtrType::get(rewriter.getContext(), VoidType::get(rewriter.getContext()), *space);
    auto alloc        = AllocOp::create(rewriter, loc, voidPtrType, elements, constantIndex(rewriter, loc, alignment));
    auto ptrType      = PtrType::get(rewriter.getContext(), triton::i1ToI8(type.getElementType()), *space);
    Value base        = PointCastOp::create(rewriter, loc, ptrType, alloc.getResult());

    MemRefParts parts{base, constantIndex(rewriter, loc, staticOffset), std::move(sizes), std::move(strides)};
    SmallVector<Value> tuple = parts.pack();
    rewriter.replaceOpWithMultiple(op, {tuple});
    return success();
  }
};

class DmaSlotOpLowering : public OpConversionPattern<DmaSlotOp>
{
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(DmaSlotOp op, OneToNOpAdaptor adaptor, ConversionPatternRewriter& rewriter) const override
  {
    auto type = dyn_cast<MemRefType>(op.getBuffer().getType());
    if (!type) return op.emitError("expected a memref buffer");

    FailureOr<MemRefParts> parts = MemRefParts::unpack(adaptor.getBuffer(), type, op);
    if (failed(parts)) return failure();
    FailureOr<Value> ptr = parts->elementPtr(rewriter, op.getLoc());
    if (failed(ptr)) return failure();
    auto slot = DmaSlotOp::create(rewriter, op.getLoc(), op.getHandle().getType(), op.getMeta().getType(), *ptr);
    rewriter.replaceOp(op, slot.getResults());
    return success();
  }
};

class DeallocOpLowering : public OpConversionPattern<memref::DeallocOp>
{
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(memref::DeallocOp op, OneToNOpAdaptor adaptor, ConversionPatternRewriter& rewriter) const override
  {
    auto type                      = cast<MemRefType>(op.getMemref().getType());
    FailureOr<MemRefParts> partsOr = MemRefParts::unpack(adaptor.getMemref(), type, op);
    if (failed(partsOr)) return failure();
    MemRefParts parts = *partsOr;
    ;
    if (!isOwnedAllocation(parts.base))
      return op.emitError("cannot free a memref that does not come straight out of a pumem.alloc");
    std::optional<int64_t> offset = getConstantIntValue(parts.offset);
    if (!offset || *offset != 0) return op.emitError("cannot free a memref whose offset is not statically zero");
    Value base = parts.base;
    if (auto cast = base.getDefiningOp<PointCastOp>()) base = cast.getPtr();
    DeallocOp::create(rewriter, op.getLoc(), base);
    rewriter.eraseOp(op);
    return success();
  }
};

class LoadOpLowering : public OpConversionPattern<memref::LoadOp>
{
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(memref::LoadOp op, OneToNOpAdaptor adaptor, ConversionPatternRewriter& rewriter) const override
  {
    auto type                      = cast<MemRefType>(op.getMemref().getType());
    FailureOr<MemRefParts> partsOr = MemRefParts::unpack(adaptor.getMemref(), type, op);
    if (failed(partsOr)) return failure();
    MemRefParts parts = *partsOr;
    ;
    SmallVector<Value> indices;
    for (ValueRange index : adaptor.getIndices()) indices.push_back(index.front());
    FailureOr<Value> address = parts.elementPtr(rewriter, op.getLoc(), indices);
    if (failed(address)) return failure();
    rewriter.replaceOpWithNewOp<LoadOp>(op, op.getType(), *address);
    return success();
  }
};

class StoreOpLowering : public OpConversionPattern<memref::StoreOp>
{
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(memref::StoreOp op, OneToNOpAdaptor adaptor, ConversionPatternRewriter& rewriter) const override
  {
    auto type                      = cast<MemRefType>(op.getMemref().getType());
    FailureOr<MemRefParts> partsOr = MemRefParts::unpack(adaptor.getMemref(), type, op);
    if (failed(partsOr)) return failure();
    MemRefParts parts = *partsOr;
    ;
    SmallVector<Value> indices;
    for (ValueRange index : adaptor.getIndices()) indices.push_back(index.front());
    FailureOr<Value> address = parts.elementPtr(rewriter, op.getLoc(), indices);
    if (failed(address)) return failure();
    StoreOp::create(rewriter, op.getLoc(), adaptor.getValue().front(), *address);
    rewriter.eraseOp(op);
    return success();
  }
};

class CopyOpLowering : public OpConversionPattern<memref::CopyOp>
{
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(memref::CopyOp op, OneToNOpAdaptor adaptor, ConversionPatternRewriter& rewriter) const override
  {
    Location loc                    = op.getLoc();
    auto sourceType                 = cast<MemRefType>(op.getSource().getType());
    auto targetType                 = cast<MemRefType>(op.getTarget().getType());
    FailureOr<MemRefParts> sourceOr = MemRefParts::unpack(adaptor.getSource(), sourceType, op);
    if (failed(sourceOr)) return failure();
    MemRefParts source = *sourceOr;
    ;
    FailureOr<MemRefParts> targetOr = MemRefParts::unpack(adaptor.getTarget(), targetType, op);
    if (failed(targetOr)) return failure();
    MemRefParts target = *targetOr;
    ;

    FailureOr<int64_t> elementBytes = getElementByteSize(sourceType.getElementType(), op);
    if (failed(elementBytes)) return op.emitError("cannot determine the byte size of the element type");

    FailureOr<Value> targetPtr = target.elementPtr(rewriter, loc);
    FailureOr<Value> sourcePtr = source.elementPtr(rewriter, loc);
    if (failed(targetPtr) || failed(sourcePtr)) return failure();
    CopyOp::create(
      rewriter, loc, *targetPtr, *sourcePtr, source.sizes, target.strides, source.strides,
      constantIndex(rewriter, loc, *elementBytes)
    );
    rewriter.eraseOp(op);
    return success();
  }
};

class TileGatherOpLowering : public OpConversionPattern<TileGatherOp>
{
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(TileGatherOp op, OneToNOpAdaptor adaptor, ConversionPatternRewriter& rewriter) const override
  {
    Location loc                    = op.getLoc();
    auto sourceType                 = cast<MemRefType>(op.getSource().getType());
    auto indicesType                = cast<MemRefType>(op.getIndices().getType());
    auto destinationType            = cast<MemRefType>(op.getDestination().getType());
    FailureOr<MemRefParts> sourceOr = MemRefParts::unpack(adaptor.getSource(), sourceType, op);
    if (failed(sourceOr)) return failure();
    MemRefParts source = *sourceOr;
    ;
    FailureOr<MemRefParts> indicesOr = MemRefParts::unpack(adaptor.getIndices(), indicesType, op);
    if (failed(indicesOr)) return failure();
    MemRefParts indices = *indicesOr;
    ;
    FailureOr<MemRefParts> destinationOr = MemRefParts::unpack(adaptor.getDestination(), destinationType, op);
    if (failed(destinationOr)) return failure();
    MemRefParts destination = *destinationOr;

    FailureOr<Value> maskPtr;
    SmallVector<Value> maskStrides;
    if (op.getMask()) {
      FailureOr<MemRefParts> maskPartsOr =
        MemRefParts::unpack(adaptor.getMask(), cast<MemRefType>(op.getMask().getType()), op);
      if (failed(maskPartsOr)) return failure();
      MemRefParts maskParts = *maskPartsOr;
      ;
      maskPtr     = maskParts.elementPtr(rewriter, loc);
      maskStrides = std::move(maskParts.strides);
      if (failed(maskPtr)) return failure();
    }
    Value mask  = succeeded(maskPtr) ? *maskPtr : Value();
    Value other = op.getOther() ? adaptor.getOther().front() : Value();

    FailureOr<Value> sourcePtr      = source.elementPtr(rewriter, loc);
    FailureOr<Value> indicesPtr     = indices.elementPtr(rewriter, loc);
    FailureOr<Value> destinationPtr = destination.elementPtr(rewriter, loc);
    if (failed(sourcePtr) || failed(indicesPtr) || failed(destinationPtr)) return failure();

    GatherOp::create(
      rewriter, loc, *sourcePtr, source.strides.front(), *indicesPtr, *destinationPtr, destination.sizes,
      indices.strides, destination.strides, mask, maskStrides, other
    );
    rewriter.eraseOp(op);
    return success();
  }
};

class TileScatterOpLowering : public OpConversionPattern<TileScatterOp>
{
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(TileScatterOp op, OneToNOpAdaptor adaptor, ConversionPatternRewriter& rewriter) const override
  {
    Location loc                    = op.getLoc();
    auto sourceType                 = cast<MemRefType>(op.getSource().getType());
    auto indicesType                = cast<MemRefType>(op.getIndices().getType());
    auto destinationType            = cast<MemRefType>(op.getDestination().getType());
    FailureOr<MemRefParts> sourceOr = MemRefParts::unpack(adaptor.getSource(), sourceType, op);
    if (failed(sourceOr)) return failure();
    MemRefParts source = *sourceOr;
    ;
    FailureOr<MemRefParts> indicesOr = MemRefParts::unpack(adaptor.getIndices(), indicesType, op);
    if (failed(indicesOr)) return failure();
    MemRefParts indices = *indicesOr;
    ;
    FailureOr<MemRefParts> destinationOr = MemRefParts::unpack(adaptor.getDestination(), destinationType, op);
    if (failed(destinationOr)) return failure();
    MemRefParts destination = *destinationOr;

    FailureOr<Value> maskPtr;
    SmallVector<Value> maskStrides;
    if (op.getMask()) {
      FailureOr<MemRefParts> maskPartsOr =
        MemRefParts::unpack(adaptor.getMask(), cast<MemRefType>(op.getMask().getType()), op);
      if (failed(maskPartsOr)) return failure();
      MemRefParts maskParts = *maskPartsOr;
      ;
      maskPtr     = maskParts.elementPtr(rewriter, loc);
      maskStrides = std::move(maskParts.strides);
      if (failed(maskPtr)) return failure();
    }
    Value mask = succeeded(maskPtr) ? *maskPtr : Value();

    FailureOr<Value> sourcePtr      = source.elementPtr(rewriter, loc);
    FailureOr<Value> indicesPtr     = indices.elementPtr(rewriter, loc);
    FailureOr<Value> destinationPtr = destination.elementPtr(rewriter, loc);
    if (failed(sourcePtr) || failed(indicesPtr) || failed(destinationPtr)) return failure();

    ScatterOp::create(
      rewriter, loc, *sourcePtr, *indicesPtr, *destinationPtr, destination.strides.front(), source.sizes,
      source.strides, indices.strides, mask, maskStrides
    );
    rewriter.eraseOp(op);
    return success();
  }
};

class RuntimeCallOpLowering : public OpConversionPattern<RuntimeCallOp>
{
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(RuntimeCallOp op, OneToNOpAdaptor adaptor, ConversionPatternRewriter& rewriter) const override
  {
    bool outlined = isa_and_nonnull<func::FuncOp>(SymbolTable::lookupNearestSymbolFrom(op, op.getCalleeAttr()));

    SmallVector<Value> operands;
    for (auto [type, mapping] : llvm::zip_equal(op->getOperandTypes(), adaptor.getOperands())) {
      if (auto memrefType = dyn_cast<MemRefType>(type)) {
        if (outlined) {
          operands.append(mapping.begin(), mapping.end());
          continue;
        }
        FailureOr<MemRefParts> parts = MemRefParts::unpack(mapping, memrefType, op);
        if (failed(parts)) return failure();
        FailureOr<Value> ptr = parts->elementPtr(rewriter, op.getLoc());
        if (failed(ptr)) return failure();
        operands.push_back(*ptr);
        continue;
      }
      operands.append(mapping.begin(), mapping.end());
    }

    auto templateArgs = op.getTemplateArgs() ? *op.getTemplateArgs() : ArrayAttr{};
    RuntimeCallOp::create(rewriter, op.getLoc(), TypeRange{}, op.getCalleeAttr(), templateArgs, operands);
    rewriter.eraseOp(op);
    return success();
  }
};

class CastOpLowering : public OpConversionPattern<memref::CastOp>
{
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(memref::CastOp op, OneToNOpAdaptor adaptor, ConversionPatternRewriter& rewriter) const override
  {
    auto sourceType                = cast<MemRefType>(op.getSource().getType());
    FailureOr<MemRefParts> partsOr = MemRefParts::unpack(adaptor.getSource(), sourceType, op);
    if (failed(partsOr)) return failure();
    MemRefParts parts = *partsOr;
    ;

    FailureOr<int64_t> sourceSpace = getMemRefSpace(sourceType);
    auto targetType                = cast<MemRefType>(op.getType());
    FailureOr<int64_t> targetSpace = getMemRefSpace(targetType);
    if (failed(sourceSpace) || failed(targetSpace) || *sourceSpace != *targetSpace)
      return op.emitError("cannot lower a memref.cast that changes the memory space");

    SmallVector<Value> tuple = parts.pack();
    rewriter.replaceOpWithMultiple(op, {tuple});
    return success();
  }
};

class CastOpReconcile : public OpConversionPattern<UnrealizedConversionCastOp>
{
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
    UnrealizedConversionCastOp op, OneToNOpAdaptor adaptor, ConversionPatternRewriter& rewriter
  ) const override
  {
    if (op.getInputs().size() != 1 || op.getOutputs().size() != 1) return failure();

    SmallVector<Type> results;
    if (failed(getTypeConverter()->convertType(op.getOutputs().front().getType(), results))) return failure();

    ValueRange converted = adaptor.getOperands().front();
    if (converted.size() != results.size()) return failure();
    for (auto [value, type] : llvm::zip_equal(converted, results))
      if (value.getType() != type) return failure();

    rewriter.replaceOpWithMultiple(op, {converted});
    return success();
  }
};

class DimOpLowering : public OpConversionPattern<memref::DimOp>
{
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(memref::DimOp op, OneToNOpAdaptor adaptor, ConversionPatternRewriter& rewriter) const override
  {
    auto constant = getConstantIntValue(adaptor.getIndex().front());
    if (!constant) {
      return op.emitError("cannot lower memref.dim with a dynamic dimension index");
    }

    auto type                      = cast<MemRefType>(op.getSource().getType());
    FailureOr<MemRefParts> partsOr = MemRefParts::unpack(adaptor.getSource(), type, op);
    if (failed(partsOr)) return failure();
    MemRefParts parts = *partsOr;
    ;
    int64_t index = *constant;
    if (index >= type.getRank()) return op.emitError("dimension index is out of range");
    rewriter.replaceOp(op, parts.sizes[index]);
    return success();
  }
};

class RankOpLowering : public OpConversionPattern<memref::RankOp>
{
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(memref::RankOp op, OneToNOpAdaptor adaptor, ConversionPatternRewriter& rewriter) const override
  {
    auto type = cast<MemRefType>(op.getMemref().getType());
    rewriter.replaceOpWithNewOp<arith::ConstantIndexOp>(op, type.getRank());
    return success();
  }
};

class ExtractStridedMetadataOpLowering : public OpConversionPattern<memref::ExtractStridedMetadataOp>
{
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
    memref::ExtractStridedMetadataOp op, OneToNOpAdaptor adaptor, ConversionPatternRewriter& rewriter
  ) const override
  {
    Location loc                   = op.getLoc();
    auto type                      = cast<MemRefType>(op.getSource().getType());
    FailureOr<MemRefParts> partsOr = MemRefParts::unpack(adaptor.getSource(), type, op);
    if (failed(partsOr)) return failure();
    MemRefParts parts = *partsOr;
    ;

    SmallVector<Value> flat{parts.base, constantIndex(rewriter, loc, 0)};
    flat.push_back(parts.offset);
    flat.append(parts.sizes.begin(), parts.sizes.end());
    flat.append(parts.strides.begin(), parts.strides.end());

    SmallVector<ValueRange> replacements;
    replacements.push_back(ValueRange(flat).take_front(2));
    for (unsigned i = 2, e = flat.size(); i < e; ++i) replacements.push_back(ValueRange(flat).slice(i, 1));
    rewriter.replaceOpWithMultiple(op, replacements);
    return success();
  }
};

class ReinterpretCastOpLowering : public OpConversionPattern<memref::ReinterpretCastOp>
{
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
    memref::ReinterpretCastOp op, OneToNOpAdaptor adaptor, ConversionPatternRewriter& rewriter
  ) const override
  {
    Location loc                    = op.getLoc();
    auto sourceType                 = cast<MemRefType>(op.getSource().getType());
    auto targetType                 = cast<MemRefType>(op.getType());
    FailureOr<MemRefParts> sourceOr = MemRefParts::unpack(adaptor.getSource(), sourceType, op);
    if (failed(sourceOr)) return failure();
    MemRefParts source = *sourceOr;
    ;

    FailureOr<int64_t> sourceSpace = getMemRefSpace(sourceType);
    FailureOr<int64_t> targetSpace = getMemRefSpace(targetType);
    if (failed(sourceSpace) || failed(targetSpace) || *sourceSpace != *targetSpace)
      return op.emitError("cannot lower a memref.reinterpret_cast that changes the memory space");

    SmallVector<Value> offsets = materializeMixedIndexList(rewriter, loc, op.getMixedOffsets(), adaptor.getOffsets());
    SmallVector<Value> sizes   = materializeMixedIndexList(rewriter, loc, op.getMixedSizes(), adaptor.getSizes());
    SmallVector<Value> strides = materializeMixedIndexList(rewriter, loc, op.getMixedStrides(), adaptor.getStrides());

    Value offset = offsets.front();
    SmallVector<Value> tuple{source.base, offset};
    tuple.append(sizes.begin(), sizes.end());
    tuple.append(strides.begin(), strides.end());
    rewriter.replaceOpWithMultiple(op, {tuple});
    return success();
  }
};

template <typename OpT> class ForOpLowering : public OpConversionPattern<OpT>
{
public:
  using OpConversionPattern<OpT>::OpConversionPattern;
  using OneToNOpAdaptor = typename OpConversionPattern<OpT>::OneToNOpAdaptor;

  LogicalResult matchAndRewrite(OpT op, OneToNOpAdaptor adaptor, ConversionPatternRewriter& rewriter) const override
  {
    SmallVector<size_t> resultCounts;
    if (failed(getConvertedCounts(op.getResultTypes(), *this->getTypeConverter(), resultCounts)))
      return rewriter.notifyMatchFailure(op, "cannot convert loop result types");

    if (failed(rewriter.convertRegionTypes(&op.getRegion(), *this->getTypeConverter())))
      return rewriter.notifyMatchFailure(op, "cannot convert loop region types");

    SmallVector<Value> initArgs = flattenValues(adaptor.getInitArgs());
    auto newOp                  = OpT::create(
      rewriter, op.getLoc(), adaptor.getLowerBound().front(), adaptor.getUpperBound().front(),
      adaptor.getStep().front(), initArgs
    );
    newOp->setAttrs(op->getAttrs());
    rewriter.eraseBlock(newOp.getBody());
    rewriter.inlineRegionBefore(op.getRegion(), newOp.getRegion(), newOp.getRegion().end());

    SmallVector<ValueRange> replacements;
    unsigned offset = 0;
    for (size_t count : resultCounts) {
      replacements.push_back(newOp.getResults().slice(offset, count));
      offset += count;
    }
    rewriter.replaceOpWithMultiple(op, replacements);
    return success();
  }
};

template <typename OpT> class IfOpLowering : public OpConversionPattern<OpT>
{
public:
  using OpConversionPattern<OpT>::OpConversionPattern;
  using OneToNOpAdaptor = typename OpConversionPattern<OpT>::OneToNOpAdaptor;

  LogicalResult matchAndRewrite(OpT op, OneToNOpAdaptor adaptor, ConversionPatternRewriter& rewriter) const override
  {
    SmallVector<Type> resultTypes;
    SmallVector<size_t> resultCounts;
    for (Type type : op.getResultTypes()) {
      SmallVector<Type> converted;
      if (failed(this->getTypeConverter()->convertTypes(type, converted)))
        return rewriter.notifyMatchFailure(op, "cannot convert if result types");
      resultTypes.append(converted.begin(), converted.end());
      resultCounts.push_back(converted.size());
    }

    bool withElseRegion = !op.getElseRegion().empty();
    auto newOp = OpT::create(rewriter, op.getLoc(), resultTypes, adaptor.getCondition().front(), withElseRegion);
    newOp->setAttrs(op->getAttrs());
    rewriter.eraseBlock(&newOp.getThenRegion().front());
    if (withElseRegion) rewriter.eraseBlock(&newOp.getElseRegion().front());
    rewriter.inlineRegionBefore(op.getThenRegion(), newOp.getThenRegion(), newOp.getThenRegion().end());
    if (withElseRegion)
      rewriter.inlineRegionBefore(op.getElseRegion(), newOp.getElseRegion(), newOp.getElseRegion().end());

    SmallVector<ValueRange> replacements;
    unsigned offset = 0;
    for (size_t count : resultCounts) {
      replacements.push_back(newOp.getResults().slice(offset, count));
      offset += count;
    }
    rewriter.replaceOpWithMultiple(op, replacements);
    return success();
  }
};

template <typename OpT> class YieldOpLowering : public OpConversionPattern<OpT>
{
public:
  using OpConversionPattern<OpT>::OpConversionPattern;
  using OneToNOpAdaptor = typename OpConversionPattern<OpT>::OneToNOpAdaptor;

  LogicalResult matchAndRewrite(OpT op, OneToNOpAdaptor adaptor, ConversionPatternRewriter& rewriter) const override
  {
    SmallVector<Value> operands = flattenValues(adaptor.getResults());
    rewriter.replaceOpWithNewOp<OpT>(op, operands);
    return success();
  }
};

class FuncOpLowering : public OpConversionPattern<triton::FuncOp>
{
public:
  using OpConversionPattern::OpConversionPattern;

  static bool unsetAttrs(NamedAttribute attr)
  {
    constexpr std::array<StringRef, 5> unsetNames{
      "sym_name", "function_type", "sym_visibility", "arg_attrs", "res_attrs"
    };
    StringRef name = attr.getName().getValue();
    for (auto unsetName : unsetNames)
      if (name == unsetName) return true;
    return false;
  }

  LogicalResult
  matchAndRewrite(triton::FuncOp op, OneToNOpAdaptor adaptor, ConversionPatternRewriter& rewriter) const override
  {
    auto oldType = cast<FunctionType>(op.getFunctionType());

    TypeConverter::SignatureConversion signature(oldType.getNumInputs());
    if (failed(getTypeConverter()->convertSignatureArgs(oldType.getInputs(), signature)))
      return rewriter.notifyMatchFailure(op, "cannot convert function argument types");

    SmallVector<Type> newResults;
    if (failed(getTypeConverter()->convertTypes(oldType.getResults(), newResults)))
      return rewriter.notifyMatchFailure(op, "cannot convert function result types");

    SmallVector<Attribute> argAttrs = expandAttrArray(op.getArgAttrsAttr(), oldType.getInputs(), *getTypeConverter());
    SmallVector<Attribute> resAttrs = expandAttrArray(op.getResAttrsAttr(), oldType.getResults(), *getTypeConverter());

    auto newType = FunctionType::get(getContext(), signature.getConvertedTypes(), newResults);
    auto newOp   = func::FuncOp::create(rewriter, op.getLoc(), op.getName(), newType);
    newOp.setVisibility(op.getVisibility());

    for (NamedAttribute attr : op->getAttrs()) {
      if (unsetAttrs(attr)) continue;
      newOp->setAttr(attr.getName(), attr.getValue());
    }
    if (op.getArgAttrsAttr()) newOp.setArgAttrsAttr(rewriter.getArrayAttr(argAttrs));
    if (op.getResAttrsAttr()) newOp.setResAttrsAttr(rewriter.getArrayAttr(resAttrs));

    if (!op.getBody().empty()) {
      rewriter.inlineRegionBefore(op.getBody(), newOp.getBody(), newOp.getBody().end());
      rewriter.applySignatureConversion(&newOp.getBody().front(), signature, getTypeConverter());
    }
    rewriter.eraseOp(op);
    return success();
  }
};

class ReturnOpLowering : public OpConversionPattern<triton::ReturnOp>
{
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(triton::ReturnOp op, OneToNOpAdaptor adaptor, ConversionPatternRewriter& rewriter) const override
  {
    SmallVector<Value> operands = flattenValues(adaptor.getSrcs());
    rewriter.replaceOpWithNewOp<func::ReturnOp>(op, operands);
    return success();
  }
};

enum class EntryArgKind {
  Keep,
  Constant,
  DropIfUnused,
};

struct EntryArg
{
  EntryArgKind kind;
  int64_t value;
};

using EntryConstants = llvm::DenseMap<StringAttr, SmallVector<EntryArg>>;

EntryArg entryConstant(int64_t value)
{
  if (value == ShapedType::kDynamic) return {EntryArgKind::DropIfUnused, 0};
  return {EntryArgKind::Constant, value};
}

EntryConstants collectEntryConstants(ModuleOp module)
{
  EntryConstants constants;
  module.walk([&](triton::FuncOp func) {
    if (!func.isPublic() || func.isExternal()) return;
    std::optional<SymbolTable::UseRange> uses = SymbolTable::getSymbolUses(func, module);
    if (!uses || !uses->empty()) return;

    SmallVector<EntryArg> values;
    for (Type input : cast<FunctionType>(func.getFunctionType()).getInputs()) {
      auto type = dyn_cast<MemRefType>(input);
      SmallVector<int64_t> strides;
      int64_t offset = ShapedType::kDynamic;
      if (!type || failed(type.getStridesAndOffset(strides, offset))) {
        values.push_back({EntryArgKind::Keep, 0});
        continue;
      }
      values.push_back({EntryArgKind::Keep, 0});
      values.push_back(entryConstant(offset));
      values.append(type.getRank(), {EntryArgKind::DropIfUnused, 0});
      for (int64_t stride : strides) values.push_back(entryConstant(stride));
    }
    constants[func.getNameAttr()] = std::move(values);
  });
  return constants;
}

LogicalResult foldEntryConstants(ModuleOp module, const EntryConstants& constants)
{
  WalkResult result = module.walk([&](func::FuncOp func) -> WalkResult {
    auto it = constants.find(func.getNameAttr());
    if (it == constants.end()) return WalkResult::advance();
    if (it->second.size() != func.getNumArguments())
      return func->emitError("kernel signature does not match the entry memref descriptors");

    Block& body = func.getBody().front();
    OpBuilder builder(&body, body.begin());
    llvm::BitVector dead(func.getNumArguments());
    for (auto [index, entry] : llvm::enumerate(it->second)) {
      Value argument = func.getArgument(index);
      switch (entry.kind) {
        case EntryArgKind::Keep: break;
        case EntryArgKind::Constant:
          argument.replaceAllUsesWith(constantIndex(builder, func.getLoc(), entry.value));
          dead.set(index);
          break;
        case EntryArgKind::DropIfUnused:
          if (argument.use_empty()) dead.set(index);
          break;
      }
    }
    if (dead.none()) return WalkResult::advance();
    if (failed(func.eraseArguments(dead))) return func->emitError("failed to erase descriptor arguments");
    return WalkResult::advance();
  });
  return failure(result.wasInterrupted());
}

struct MemrefToPumemPass : public ::impl::MemrefToPumemBase<MemrefToPumemPass>
{
  using MemrefToPumemBase::MemrefToPumemBase;

  void runOnOperation() override
  {
    ModuleOp module = getOperation();
    if (failed(checkPreconditions(module))) return signalPassFailure();

    EntryConstants entryConstants = collectEntryConstants(module);
    MemRefTupleTypeConverter converter;
    ConversionTarget target(getContext());
    target.markUnknownOpDynamicallyLegal([](Operation* op) { return !hasMemRefTypes(op); });
    target.addIllegalOp<triton::FuncOp, triton::CallOp, triton::ReturnOp>();

    RewritePatternSet patterns(&getContext());
    patterns.add<AllocOpLowering>(converter, &getContext());
    patterns.add<DmaSlotOpLowering>(converter, &getContext());
    patterns.add<DeallocOpLowering>(converter, &getContext());
    patterns.add<LoadOpLowering>(converter, &getContext());
    patterns.add<StoreOpLowering>(converter, &getContext());
    patterns.add<CopyOpLowering>(converter, &getContext());
    patterns.add<TileGatherOpLowering>(converter, &getContext());
    patterns.add<TileScatterOpLowering>(converter, &getContext());
    patterns.add<RuntimeCallOpLowering>(converter, &getContext());
    patterns.add<CastOpLowering>(converter, &getContext());
    patterns.add<CastOpReconcile>(converter, &getContext());
    patterns.add<DimOpLowering>(converter, &getContext());
    patterns.add<RankOpLowering>(converter, &getContext());
    patterns.add<ExtractStridedMetadataOpLowering>(converter, &getContext());
    patterns.add<ReinterpretCastOpLowering>(converter, &getContext());
    patterns.add<ForOpLowering<scf::ForOp>, ForOpLowering<pumem::ForOp>>(converter, &getContext());
    patterns.add<IfOpLowering<scf::IfOp>, IfOpLowering<pumem::IfOp>>(converter, &getContext());
    patterns.add<YieldOpLowering<scf::YieldOp>, YieldOpLowering<pumem::YieldOp>>(converter, &getContext());
    patterns.add<FuncOpLowering>(converter, &getContext());
    patterns.add<ReturnOpLowering>(converter, &getContext());
    populateFunctionOpInterfaceTypeConversionPattern<func::FuncOp>(patterns, converter);
    populateCallOpTypeConversionPattern(patterns, converter);
    populateReturnOpTypeConversionPattern(patterns, converter);

    if (failed(applyPartialConversion(module, target, std::move(patterns)))) return signalPassFailure();
    if (failed(foldEntryConstants(module, entryConstants))) signalPassFailure();
  }
};

} // namespace

std::unique_ptr<OperationPass<ModuleOp>> mlir::triton::createMemrefToPumemPass()
{
  return std::make_unique<MemrefToPumemPass>();
}
