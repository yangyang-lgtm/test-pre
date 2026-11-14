#include "privateuse/Dialect/PUMem/IR/PUMemDialect.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/MathExtras.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpImplementation.h"

#define GET_DIALECT_DEFS
#include "privateuse/Dialect/PUMem/IR/PUMemDialect.cpp.inc"

#define GET_TYPEDEF_CLASSES
#include "privateuse/Dialect/PUMem/IR/PUMemTypes.cpp.inc"

#define GET_OP_CLASSES
#include "privateuse/Dialect/PUMem/IR/PUMemOps.cpp.inc"

using namespace mlir;

void mlir::pumem::PUMemDialect::initialize()
{
  addTypes<
#define GET_TYPEDEF_LIST
#include "privateuse/Dialect/PUMem/IR/PUMemTypes.cpp.inc"
    >();
  addOperations<
#define GET_OP_LIST
#include "privateuse/Dialect/PUMem/IR/PUMemOps.cpp.inc"
    >();
}

LogicalResult mlir::pumem::AllocOp::verify()
{
  int64_t alignment = getAlignment();
  if (alignment <= 0 || !llvm::isPowerOf2_64(alignment))
    return emitOpError("alignment must be a positive power of two");
  return success();
}

OpFoldResult mlir::pumem::AddPtrOp::fold(FoldAdaptor adaptor)
{
  if (auto offset = dyn_cast_or_null<IntegerAttr>(adaptor.getOffset()))
    if (offset.getInt() == 0) return getPtr();
  return {};
}

LogicalResult mlir::pumem::LoadOp::verify()
{
  if (getPtr().getType().getElemType() != getValue().getType())
    return emitOpError("result type must match the pointer element type");
  return success();
}

LogicalResult mlir::pumem::StoreOp::verify()
{
  if (getPtr().getType().getElemType() != getValue().getType())
    return emitOpError("value type must match the pointer element type");
  return success();
}

namespace {

ParseResult parseIndexList(OpAsmParser& parser, StringRef name, SmallVectorImpl<OpAsmParser::UnresolvedOperand>& list)
{
  if (parser.parseKeyword(name) || parser.parseEqual() || parser.parseLParen()) return failure();
  if (failed(parser.parseOptionalRParen())) {
    do {
      OpAsmParser::UnresolvedOperand operand;
      if (parser.parseOperand(operand)) return failure();
      list.push_back(operand);
    } while (succeeded(parser.parseOptionalComma()));
    if (parser.parseRParen()) return failure();
  }
  return success();
}

void printIndexList(OpAsmPrinter& printer, StringRef name, OperandRange list)
{
  printer << name << " = (";
  llvm::interleaveComma(list, printer);
  printer << ")";
}

} // namespace

ParseResult mlir::pumem::CopyOp::parse(OpAsmParser& parser, OperationState& result)
{
  Builder& builder = parser.getBuilder();
  OpAsmParser::UnresolvedOperand dst;
  OpAsmParser::UnresolvedOperand src;
  OpAsmParser::UnresolvedOperand byteSize;
  SmallVector<OpAsmParser::UnresolvedOperand> sizes;
  SmallVector<OpAsmParser::UnresolvedOperand> dstStrides;
  SmallVector<OpAsmParser::UnresolvedOperand> srcStrides;

  if (parser.parseOperand(dst) || parser.parseComma() || parser.parseOperand(src)) return failure();
  if (parser.parseComma() || parseIndexList(parser, "sizes", sizes)) return failure();
  if (parser.parseComma() || parseIndexList(parser, "dst_strides", dstStrides)) return failure();
  if (parser.parseComma() || parseIndexList(parser, "src_strides", srcStrides)) return failure();
  if (parser.parseComma() || parser.parseKeyword("byte_size") || parser.parseOperand(byteSize)) return failure();
  if (parser.parseOptionalAttrDict(result.attributes)) return failure();

  Type dstType;
  Type srcType;
  if (parser.parseColonType(dstType) || parser.parseComma() || parser.parseType(srcType)) return failure();

  SmallVector<OpAsmParser::UnresolvedOperand> operands{dst, src};
  SmallVector<Type> types{dstType, srcType};
  Type indexType = builder.getIndexType();
  for (ArrayRef<OpAsmParser::UnresolvedOperand> list : {sizes, dstStrides, srcStrides}) {
    operands.append(list.begin(), list.end());
    types.append(list.size(), indexType);
  }
  operands.push_back(byteSize);
  types.push_back(indexType);

  if (parser.resolveOperands(operands, types, parser.getCurrentLocation(), result.operands)) return failure();

  result.getOrAddProperties<CopyOp::Properties>().setOperandSegmentSizes(
    {1, 1, static_cast<int32_t>(sizes.size()), static_cast<int32_t>(dstStrides.size()),
     static_cast<int32_t>(srcStrides.size()), 1}
  );
  return success();
}

void mlir::pumem::CopyOp::print(OpAsmPrinter& printer)
{
  printer << ' ' << getDst() << ", " << getSrc() << ", sizes = (";
  llvm::interleaveComma(getSizes(), printer);
  printer << "), dst_strides = (";
  llvm::interleaveComma(getDstStrides(), printer);
  printer << "), src_strides = (";
  llvm::interleaveComma(getSrcStrides(), printer);
  printer << "), byte_size " << getByteSize();
  printer.printOptionalAttrDict((*this)->getAttrs(), {"operandSegmentSizes"});
  printer << " : " << getDst().getType() << ", " << getSrc().getType();
}

LogicalResult mlir::pumem::CopyOp::verify()
{
  if (getSizes().size() != getDstStrides().size() || getSizes().size() != getSrcStrides().size())
    return emitOpError("sizes, dst_strides and src_strides must have the same length");
  if (getDst().getType().getElemType() != getSrc().getType().getElemType())
    return emitOpError("source and destination element types must match");
  return success();
}

LogicalResult mlir::pumem::TileGatherOp::verify()
{
  auto source      = dyn_cast<MemRefType>(getSource().getType());
  auto indices     = dyn_cast<MemRefType>(getIndices().getType());
  auto destination = dyn_cast<MemRefType>(getDestination().getType());
  if (!source || !indices || !destination) return emitOpError("requires ranked memref operands");
  if (source.getRank() != 1) return emitOpError("source must be a rank-1 memref");
  if (!indices.getElementType().isIntOrIndex()) return emitOpError("indices must have integer element type");
  if (indices.getShape() != destination.getShape()) return emitOpError("indices and destination shapes must match");
  if (source.getElementType() != destination.getElementType())
    return emitOpError("source and destination element types must match");
  if (auto mask = getMask()) {
    auto maskType = dyn_cast<MemRefType>(mask.getType());
    if (!maskType) return emitOpError("mask must be a ranked memref");
    if (!maskType.getElementType().isInteger(1)) return emitOpError("mask must have i1 element type");
    if (maskType.getShape() != indices.getShape()) return emitOpError("mask and indices shapes must match");
  }
  if (auto other = getOther()) {
    if (!getMask()) return emitOpError("other requires a mask");
    if (other.getType() != source.getElementType()) return emitOpError("other must match the source element type");
  }
  return success();
}

LogicalResult mlir::pumem::TileScatterOp::verify()
{
  auto source      = dyn_cast<MemRefType>(getSource().getType());
  auto indices     = dyn_cast<MemRefType>(getIndices().getType());
  auto destination = dyn_cast<MemRefType>(getDestination().getType());
  if (!source || !indices || !destination) return emitOpError("requires ranked memref operands");
  if (destination.getRank() != 1) return emitOpError("destination must be a rank-1 memref");
  if (!indices.getElementType().isIntOrIndex()) return emitOpError("indices must have integer element type");
  if (indices.getShape() != source.getShape()) return emitOpError("indices and source shapes must match");
  if (source.getElementType() != destination.getElementType())
    return emitOpError("source and destination element types must match");
  if (auto mask = getMask()) {
    auto maskType = dyn_cast<MemRefType>(mask.getType());
    if (!maskType) return emitOpError("mask must be a ranked memref");
    if (!maskType.getElementType().isInteger(1)) return emitOpError("mask must have i1 element type");
    if (maskType.getShape() != source.getShape()) return emitOpError("mask and source shapes must match");
  }
  return success();
}

ParseResult mlir::pumem::GatherOp::parse(OpAsmParser& parser, OperationState& result)
{
  Builder& builder = parser.getBuilder();
  OpAsmParser::UnresolvedOperand source;
  OpAsmParser::UnresolvedOperand sourceStride;
  OpAsmParser::UnresolvedOperand indices;
  OpAsmParser::UnresolvedOperand destination;
  OpAsmParser::UnresolvedOperand mask;
  OpAsmParser::UnresolvedOperand other;
  SmallVector<OpAsmParser::UnresolvedOperand> sizes;
  SmallVector<OpAsmParser::UnresolvedOperand> indicesStrides;
  SmallVector<OpAsmParser::UnresolvedOperand> destinationStrides;
  SmallVector<OpAsmParser::UnresolvedOperand> maskStrides;
  bool hasMask  = false;
  bool hasOther = false;

  if (parser.parseOperand(source) || parser.parseComma() || parser.parseOperand(sourceStride) || parser.parseComma() ||
      parser.parseOperand(indices) || parser.parseComma() || parser.parseOperand(destination))
    return failure();
  if (parser.parseComma() || parseIndexList(parser, "sizes", sizes)) return failure();
  if (parser.parseComma() || parseIndexList(parser, "indices_strides", indicesStrides)) return failure();
  if (parser.parseComma() || parseIndexList(parser, "destination_strides", destinationStrides)) return failure();

  if (succeeded(parser.parseOptionalComma())) {
    if (succeeded(parser.parseOptionalKeyword("mask"))) {
      hasMask = true;
      if (parser.parseEqual() || parser.parseOperand(mask) || parser.parseComma() ||
          parseIndexList(parser, "mask_strides", maskStrides))
        return failure();
      if (succeeded(parser.parseOptionalComma())) {
        if (parser.parseKeyword("other") || parser.parseOperand(other)) return failure();
        hasOther = true;
      }
    } else {
      if (parser.parseKeyword("other") || parser.parseOperand(other)) return failure();
      hasOther = true;
    }
  }
  if (parser.parseOptionalAttrDict(result.attributes)) return failure();

  Type sourceType;
  Type indicesType;
  Type destinationType;
  Type maskType;
  Type otherType;
  if (parser.parseColonType(sourceType) || parser.parseComma() || parser.parseType(indicesType) ||
      parser.parseComma() || parser.parseType(destinationType))
    return failure();
  if (hasMask && (parser.parseComma() || parser.parseType(maskType))) return failure();
  if (hasOther && (parser.parseComma() || parser.parseType(otherType))) return failure();

  Type indexType = builder.getIndexType();
  SmallVector<OpAsmParser::UnresolvedOperand> operands{source, sourceStride, indices, destination};
  SmallVector<Type> types{sourceType, indexType, indicesType, destinationType};
  for (ArrayRef<OpAsmParser::UnresolvedOperand> list : {sizes, indicesStrides, destinationStrides}) {
    operands.append(list.begin(), list.end());
    types.append(list.size(), indexType);
  }
  if (hasMask) {
    operands.push_back(mask);
    types.push_back(maskType);
    operands.append(maskStrides.begin(), maskStrides.end());
    types.append(maskStrides.size(), indexType);
  }
  if (hasOther) {
    operands.push_back(other);
    types.push_back(otherType);
  }

  if (parser.resolveOperands(operands, types, parser.getCurrentLocation(), result.operands)) return failure();

  result.getOrAddProperties<GatherOp::Properties>().setOperandSegmentSizes(
    {1, 1, 1, 1, static_cast<int32_t>(sizes.size()), static_cast<int32_t>(indicesStrides.size()),
     static_cast<int32_t>(destinationStrides.size()), hasMask ? 1 : 0, static_cast<int32_t>(maskStrides.size()),
     hasOther ? 1 : 0}
  );
  return success();
}

void mlir::pumem::GatherOp::print(OpAsmPrinter& printer)
{
  printer << ' ' << getSource() << ", " << getSourceStride();
  printer << ", " << getIndices() << ", " << getDestination() << ", ";
  printIndexList(printer, "sizes", getSizes());
  printer << ", ";
  printIndexList(printer, "indices_strides", getIndicesStrides());
  printer << ", ";
  printIndexList(printer, "destination_strides", getDestinationStrides());
  if (auto mask = getMask()) {
    printer << ", mask = " << mask << ", ";
    printIndexList(printer, "mask_strides", getMaskStrides());
  }
  if (auto other = getOther()) printer << ", other " << other;
  printer.printOptionalAttrDict((*this)->getAttrs(), {"operandSegmentSizes"});
  printer << " : " << getSource().getType() << ", " << getIndices().getType() << ", " << getDestination().getType();
  if (getMask()) printer << ", " << getMask().getType();
  if (getOther()) printer << ", " << getOther().getType();
}

LogicalResult mlir::pumem::GatherOp::verify()
{
  if (getSizes().size() != getIndicesStrides().size() || getSizes().size() != getDestinationStrides().size())
    return emitOpError("sizes, indices_strides and destination_strides must have the same length");
  if (!getIndices().getType().getElemType().isIntOrIndex())
    return emitOpError("indices must have integer element type");
  if (getSource().getType().getElemType() != getDestination().getType().getElemType())
    return emitOpError("source and destination element types must match");
  if (auto mask = getMask()) {
    if (!cast<PtrType>(mask.getType()).getElemType().isInteger(1)) return emitOpError("mask must have i1 element type");
    if (getMaskStrides().size() != getSizes().size())
      return emitOpError("mask_strides must have the same length as sizes");
  } else if (!getMaskStrides().empty()) {
    return emitOpError("mask_strides require a mask");
  }
  if (auto other = getOther()) {
    if (!getMask()) return emitOpError("other requires a mask");
    if (other.getType() != getSource().getType().getElemType())
      return emitOpError("other must match the source element type");
  }
  return success();
}

ParseResult mlir::pumem::ScatterOp::parse(OpAsmParser& parser, OperationState& result)
{
  Builder& builder = parser.getBuilder();
  OpAsmParser::UnresolvedOperand source;
  OpAsmParser::UnresolvedOperand indices;
  OpAsmParser::UnresolvedOperand destination;
  OpAsmParser::UnresolvedOperand destinationStride;
  OpAsmParser::UnresolvedOperand mask;
  SmallVector<OpAsmParser::UnresolvedOperand> sizes;
  SmallVector<OpAsmParser::UnresolvedOperand> sourceStrides;
  SmallVector<OpAsmParser::UnresolvedOperand> indicesStrides;
  SmallVector<OpAsmParser::UnresolvedOperand> maskStrides;
  bool hasMask = false;

  if (parser.parseOperand(source) || parser.parseComma() || parser.parseOperand(indices) || parser.parseComma() ||
      parser.parseOperand(destination) || parser.parseComma() || parser.parseOperand(destinationStride))
    return failure();
  if (parser.parseComma() || parseIndexList(parser, "sizes", sizes)) return failure();
  if (parser.parseComma() || parseIndexList(parser, "source_strides", sourceStrides)) return failure();
  if (parser.parseComma() || parseIndexList(parser, "indices_strides", indicesStrides)) return failure();

  if (succeeded(parser.parseOptionalComma()) && succeeded(parser.parseOptionalKeyword("mask"))) {
    hasMask = true;
    if (parser.parseEqual() || parser.parseOperand(mask) || parser.parseComma() ||
        parseIndexList(parser, "mask_strides", maskStrides))
      return failure();
  }
  if (parser.parseOptionalAttrDict(result.attributes)) return failure();

  Type sourceType;
  Type indicesType;
  Type destinationType;
  Type maskType;
  if (parser.parseColonType(sourceType) || parser.parseComma() || parser.parseType(indicesType) ||
      parser.parseComma() || parser.parseType(destinationType))
    return failure();
  if (hasMask && (parser.parseComma() || parser.parseType(maskType))) return failure();

  Type indexType = builder.getIndexType();
  SmallVector<OpAsmParser::UnresolvedOperand> operands{source, indices, destination, destinationStride};
  SmallVector<Type> types{sourceType, indicesType, destinationType, indexType};
  for (ArrayRef<OpAsmParser::UnresolvedOperand> list : {sizes, sourceStrides, indicesStrides}) {
    operands.append(list.begin(), list.end());
    types.append(list.size(), indexType);
  }
  if (hasMask) {
    operands.push_back(mask);
    types.push_back(maskType);
    operands.append(maskStrides.begin(), maskStrides.end());
    types.append(maskStrides.size(), indexType);
  }

  if (parser.resolveOperands(operands, types, parser.getCurrentLocation(), result.operands)) return failure();

  result.getOrAddProperties<ScatterOp::Properties>().setOperandSegmentSizes(
    {1, 1, 1, 1, static_cast<int32_t>(sizes.size()), static_cast<int32_t>(sourceStrides.size()),
     static_cast<int32_t>(indicesStrides.size()), hasMask ? 1 : 0, static_cast<int32_t>(maskStrides.size())}
  );
  return success();
}

void mlir::pumem::ScatterOp::print(OpAsmPrinter& printer)
{
  printer << ' ' << getSource() << ", " << getIndices() << ", " << getDestination() << ", " << getDestinationStride()
          << ", ";
  printIndexList(printer, "sizes", getSizes());
  printer << ", ";
  printIndexList(printer, "source_strides", getSourceStrides());
  printer << ", ";
  printIndexList(printer, "indices_strides", getIndicesStrides());
  if (auto mask = getMask()) {
    printer << ", mask = " << mask << ", ";
    printIndexList(printer, "mask_strides", getMaskStrides());
  }
  printer.printOptionalAttrDict((*this)->getAttrs(), {"operandSegmentSizes"});
  printer << " : " << getSource().getType() << ", " << getIndices().getType() << ", " << getDestination().getType();
  if (getMask()) printer << ", " << getMask().getType();
}

LogicalResult mlir::pumem::ScatterOp::verify()
{
  if (getSizes().size() != getSourceStrides().size() || getSizes().size() != getIndicesStrides().size())
    return emitOpError("sizes, source_strides and indices_strides must have the same length");
  if (!getIndices().getType().getElemType().isIntOrIndex())
    return emitOpError("indices must have integer element type");
  if (getSource().getType().getElemType() != getDestination().getType().getElemType())
    return emitOpError("source and destination element types must match");
  if (auto mask = getMask()) {
    if (!cast<PtrType>(mask.getType()).getElemType().isInteger(1)) return emitOpError("mask must have i1 element type");
    if (getMaskStrides().size() != getSizes().size())
      return emitOpError("mask_strides must have the same length as sizes");
  } else if (!getMaskStrides().empty()) {
    return emitOpError("mask_strides require a mask");
  }
  return success();
}
