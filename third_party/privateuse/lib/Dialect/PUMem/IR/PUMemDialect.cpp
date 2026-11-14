#include "privateuse/Dialect/PUMem/IR/PUMemDialect.h"

#include "llvm/ADT/APSInt.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/MathExtras.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/LoopLikeInterface.h"
#include "privateuse/Utils/Constants.h"

#define GET_DIALECT_DEFS
#include "privateuse/Dialect/PUMem/IR/PUMemDialect.cpp.inc"

#define GET_TYPEDEF_CLASSES
#include "privateuse/Dialect/PUMem/IR/PUMemTypes.cpp.inc"

#define GET_OP_CLASSES
#include "privateuse/Dialect/PUMem/IR/PUMemOps.cpp.inc"

using namespace mlir;
using namespace mlir::pumem;

// Mask buffers are i1 in the buffer world but get promoted to i8 alongside
// the rest of the i1 values before lowering (see the i1 -> i8 conversion).
static bool isMaskElementType(Type type)
{
  return type.isInteger(1) || type.isInteger(8);
}

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
  if (!isa<VoidType>(getPtr().getType().getElemType()))
    return emitOpError("result must be a void pointer (!pumem.ptr<void, ...>), got ") << getPtr().getType();
  return success();
}

OpFoldResult mlir::pumem::PointCastOp::fold(FoldAdaptor adaptor)
{
  if (getPtr().getType() == getResult().getType()) return getPtr();
  if (auto inner = getPtr().getDefiningOp<PointCastOp>()) {
    if (inner.getPtr().getType() == getResult().getType()) return inner.getPtr();
  }
  return {};
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
  result.attributes.erase("dir");

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
  bool dstDram    = cast<PtrType>(result.operands[0].getType()).getSpace() == triton::kDramSpace;
  bool srcDram    = cast<PtrType>(result.operands[1].getType()).getSpace() == triton::kDramSpace;
  const char* dir = dstDram ? (srcDram ? "dram_to_dram" : "sram_to_dram") : (srcDram ? "dram_to_sram" : "sram_to_sram");
  result.addAttribute("dir", builder.getStringAttr(dir));
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
  StringRef dir = getDir();
  if (dir != "dram_to_sram" && dir != "sram_to_dram" && dir != "sram_to_sram" && dir != "dram_to_dram")
    return emitOpError() << "dir must be one of \"dram_to_sram\", \"sram_to_dram\", \"sram_to_sram\", "
                            "\"dram_to_dram\", got \""
                         << dir << "\"";
  bool dstDram = getDst().getType().getSpace() == triton::kDramSpace;
  bool srcDram = getSrc().getType().getSpace() == triton::kDramSpace;
  StringRef expect =
    dstDram ? (srcDram ? "dram_to_dram" : "sram_to_dram") : (srcDram ? "dram_to_sram" : "sram_to_sram");
  if (dir != expect)
    return emitOpError() << "dir \"" << dir << "\" does not match the operand spaces (expected \"" << expect << "\")";
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
    if (!isMaskElementType(maskType.getElementType())) return emitOpError("mask must have i1 or i8 element type");
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
    if (!isMaskElementType(maskType.getElementType())) return emitOpError("mask must have i1 or i8 element type");
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
    if (!isMaskElementType(cast<PtrType>(mask.getType()).getElemType()))
      return emitOpError("mask must have i1 or i8 element type");
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
    if (!isMaskElementType(cast<PtrType>(mask.getType()).getElemType()))
      return emitOpError("mask must have i1 or i8 element type");
    if (getMaskStrides().size() != getSizes().size())
      return emitOpError("mask_strides must have the same length as sizes");
  } else if (!getMaskStrides().empty()) {
    return emitOpError("mask_strides require a mask");
  }
  return success();
}

static void printInitializationList(
  OpAsmPrinter& p, Block::BlockArgListType blocksArgs, ValueRange initializers, StringRef prefix = ""
)
{
  if (!(blocksArgs.size() == initializers.size()))
    llvm::report_fatal_error("expected same length of arguments and initializers");
  if (initializers.empty()) return;

  p << prefix << '(';
  llvm::interleaveComma(llvm::zip(blocksArgs, initializers), p, [&](auto it) {
    p << std::get<0>(it) << " = " << std::get<1>(it);
  });
  p << ")";
}

void ForOp::build(
  OpBuilder& builder, OperationState& result, Value lowerBound, Value upperBound, Value step, ValueRange initArgs,
  BodyBuilderFn bodyBuilder
)
{
  OpBuilder::InsertionGuard guard(builder);

  result.addOperands({lowerBound, upperBound, step});
  result.addOperands(initArgs);
  for (Value v : initArgs) result.addTypes(v.getType());

  Type t             = lowerBound.getType();
  Region* bodyRegion = result.addRegion();
  Block* bodyBlock   = builder.createBlock(bodyRegion);
  bodyBlock->addArgument(t, result.location);
  for (Value v : initArgs) bodyBlock->addArgument(v.getType(), v.getLoc());

  if (initArgs.empty() && !bodyBuilder) {
    ForOp::ensureTerminator(*bodyRegion, builder, result.location);
  } else if (bodyBuilder) {
    OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToStart(bodyBlock);
    bodyBuilder(builder, result.location, bodyBlock->getArgument(0), bodyBlock->getArguments().drop_front());
  }
}

LogicalResult ForOp::verify()
{
  if (getBody()->getNumArguments() < getNumInductionVars())
    return emitOpError("expected body to have at least ")
           << getNumInductionVars() << " argument(s) for the induction variable, but got "
           << getBody()->getNumArguments();

  if (getInitArgs().size() != getNumResults())
    return emitOpError("mismatch in number of loop-carried values and defined values");

  return success();
}

LogicalResult ForOp::verifyRegions()
{
  if (getInductionVar().getType() != getLowerBound().getType())
    return emitOpError("expected induction variable to be same type as bounds and step");

  if (getNumRegionIterArgs() != getNumResults())
    return emitOpError("mismatch in number of basic block args and defined values");

  auto initArgs  = getInitArgs();
  auto iterArgs  = getRegionIterArgs();
  auto opResults = getResults();
  unsigned i     = 0;
  for (auto e : llvm::zip(initArgs, iterArgs, opResults)) {
    if (std::get<0>(e).getType() != std::get<2>(e).getType())
      return emitOpError() << "types mismatch between " << i << "th iter operand and defined value";
    if (std::get<1>(e).getType() != std::get<2>(e).getType())
      return emitOpError() << "types mismatch between " << i << "th iter region arg and defined value";
    ++i;
  }
  return success();
}

static std::optional<llvm::APSInt> computeUbMinusLbFor(Value lb, Value ub, bool isSigned)
{
  IntegerAttr lbAttr, ubAttr;
  if (matchPattern(lb, m_Constant(&lbAttr)) && matchPattern(ub, m_Constant(&ubAttr))) {
    llvm::APInt diff = ubAttr.getValue() - lbAttr.getValue();
    return llvm::APSInt(diff, !isSigned);
  }
  return std::nullopt;
}

SmallVector<Region*> ForOp::getLoopRegions()
{
  return {&getBodyRegion()};
}

std::optional<SmallVector<Value>> ForOp::getLoopInductionVars()
{
  return SmallVector<Value>{getInductionVar()};
}

std::optional<SmallVector<OpFoldResult>> ForOp::getLoopLowerBounds()
{
  return SmallVector<OpFoldResult>{OpFoldResult(getLowerBound())};
}

std::optional<SmallVector<OpFoldResult>> ForOp::getLoopUpperBounds()
{
  return SmallVector<OpFoldResult>{OpFoldResult(getUpperBound())};
}

std::optional<SmallVector<OpFoldResult>> ForOp::getLoopSteps()
{
  return SmallVector<OpFoldResult>{OpFoldResult(getStep())};
}

bool ForOp::isValidInductionVarType(Type type)
{
  return type.isIndex() || type.isSignlessInteger();
}

LogicalResult ForOp::setLoopLowerBounds(ArrayRef<OpFoldResult> bounds)
{
  if (bounds.size() != 1) return failure();
  if (auto val = dyn_cast<Value>(bounds[0])) {
    setLowerBound(val);
    return success();
  }
  return failure();
}

LogicalResult ForOp::setLoopUpperBounds(ArrayRef<OpFoldResult> bounds)
{
  if (bounds.size() != 1) return failure();
  if (auto val = dyn_cast<Value>(bounds[0])) {
    setUpperBound(val);
    return success();
  }
  return failure();
}

LogicalResult ForOp::setLoopSteps(ArrayRef<OpFoldResult> steps)
{
  if (steps.size() != 1) return failure();
  if (auto val = dyn_cast<Value>(steps[0])) {
    setStep(val);
    return success();
  }
  return failure();
}

MutableArrayRef<OpOperand> ForOp::getInitsMutable()
{
  return getInitArgsMutable();
}

std::optional<MutableArrayRef<OpOperand>> ForOp::getYieldedValuesMutable()
{
  return cast<YieldOp>(getBody()->getTerminator()).getResultsMutable();
}

std::optional<ResultRange> ForOp::getLoopResults()
{
  return getResults();
}

std::optional<APInt> ForOp::getStaticTripCount()
{
  return constantTripCount(
    OpFoldResult(getLowerBound()), OpFoldResult(getUpperBound()), OpFoldResult(getStep()), true, computeUbMinusLbFor
  );
}

FailureOr<LoopLikeOpInterface> ForOp::replaceWithAdditionalYields(
  RewriterBase& rewriter, ValueRange newInitOperands, bool replaceInitOperandUsesInLoop,
  const NewYieldValuesFn& newYieldValuesFn
)
{
  OpBuilder::InsertionGuard g(rewriter);
  rewriter.setInsertionPoint(getOperation());
  SmallVector<Value> inits = llvm::to_vector(getInitArgs());
  inits.append(newInitOperands.begin(), newInitOperands.end());
  ForOp newLoop = ForOp::create(
    rewriter, getLoc(), getLowerBound(), getUpperBound(), getStep(), inits,
    [](OpBuilder&, Location, Value, ValueRange) {}
  );

  YieldOp yieldOp                     = cast<YieldOp>(getBody()->getTerminator());
  ArrayRef<BlockArgument> newIterArgs = newLoop.getBody()->getArguments().take_back(newInitOperands.size());
  {
    OpBuilder::InsertionGuard g(rewriter);
    rewriter.setInsertionPoint(yieldOp);
    SmallVector<Value> newYieldedValues = newYieldValuesFn(rewriter, getLoc(), newIterArgs);
    rewriter.modifyOpInPlace(yieldOp, [&]() { yieldOp.getResultsMutable().append(newYieldedValues); });
  }

  rewriter.mergeBlocks(
    getBody(), newLoop.getBody(), newLoop.getBody()->getArguments().take_front(getBody()->getNumArguments())
  );

  if (replaceInitOperandUsesInLoop) {
    for (auto it : llvm::zip(newInitOperands, newIterArgs)) {
      rewriter.replaceUsesWithIf(std::get<0>(it), std::get<1>(it), [&](OpOperand& use) {
        Operation* user = use.getOwner();
        return newLoop->isProperAncestor(user);
      });
    }
  }

  rewriter.replaceOp(getOperation(), newLoop->getResults().take_front(getNumResults()));
  return cast<LoopLikeOpInterface>(newLoop.getOperation());
}

ParseResult ForOp::parse(OpAsmParser& parser, OperationState& result)
{
  auto& builder = parser.getBuilder();
  Type type;

  OpAsmParser::Argument inductionVariable;
  OpAsmParser::UnresolvedOperand lb, ub, step;

  if (parser.parseOperand(inductionVariable.ssaName) || parser.parseEqual() || parser.parseOperand(lb) ||
      parser.parseKeyword("to") || parser.parseOperand(ub) || parser.parseKeyword("step") || parser.parseOperand(step))
    return failure();

  SmallVector<OpAsmParser::Argument, 4> regionArgs;
  SmallVector<OpAsmParser::UnresolvedOperand, 4> operands;
  regionArgs.push_back(inductionVariable);

  bool hasIterArgs = succeeded(parser.parseOptionalKeyword("iter_args"));
  if (hasIterArgs) {
    if (parser.parseAssignmentList(regionArgs, operands) || parser.parseArrowTypeList(result.types)) return failure();
  }

  if (regionArgs.size() != result.types.size() + 1)
    return parser.emitError(parser.getNameLoc(), "mismatch in number of loop-carried values and defined values");

  if (parser.parseOptionalColon())
    type = builder.getIndexType();
  else if (parser.parseType(type))
    return failure();

  regionArgs.front().type = type;
  for (auto [iterArg, type] : llvm::zip_equal(llvm::drop_begin(regionArgs), result.types)) iterArg.type = type;

  Region* body = result.addRegion();
  if (parser.parseRegion(*body, regionArgs, false)) return failure();
  ForOp::ensureTerminator(*body, builder, result.location);

  if (parser.resolveOperand(lb, type, result.operands) || parser.resolveOperand(ub, type, result.operands) ||
      parser.resolveOperand(step, type, result.operands))
    return failure();
  if (hasIterArgs) {
    for (auto argOperandType : llvm::zip_equal(llvm::drop_begin(regionArgs), operands, result.types)) {
      Type type                        = std::get<2>(argOperandType);
      std::get<0>(argOperandType).type = type;
      if (parser.resolveOperand(std::get<1>(argOperandType), type, result.operands)) return failure();
    }
  }

  if (parser.parseOptionalAttrDict(result.attributes)) return failure();

  return success();
}

void ForOp::print(OpAsmPrinter& p)
{
  p << " " << getInductionVar() << " = " << getLowerBound() << " to " << getUpperBound() << " step " << getStep();

  printInitializationList(p, getRegionIterArgs(), getInitArgs(), " iter_args");
  if (!getInitArgs().empty()) p << " -> (" << getInitArgs().getTypes() << ')';
  p << ' ';
  p.printRegion(getRegion(), false, !getInitArgs().empty());
  p.printOptionalAttrDict((*this)->getAttrs());
}

void IfOp::build(
  OpBuilder& builder, OperationState& result, TypeRange resultTypes, Value cond, bool addThenBlock, bool addElseBlock
)
{
  if (!((!addElseBlock || addThenBlock))) llvm::report_fatal_error("must not create else block w/o then block");
  result.addTypes(resultTypes);
  result.addOperands(cond);

  OpBuilder::InsertionGuard guard(builder);
  Region* thenRegion = result.addRegion();
  if (addThenBlock) builder.createBlock(thenRegion);
  Region* elseRegion = result.addRegion();
  if (addElseBlock) builder.createBlock(elseRegion);
}

void IfOp::build(OpBuilder& builder, OperationState& result, TypeRange resultTypes, Value cond)
{
  build(builder, result, resultTypes, cond, /*withElseRegion=*/false);
}

void IfOp::build(OpBuilder& builder, OperationState& result, TypeRange resultTypes, Value cond, bool withElseRegion)
{
  build(builder, result, resultTypes, cond, /*addThenBlock=*/true, withElseRegion);
}

void IfOp::build(OpBuilder& builder, OperationState& result, Value cond, bool withElseRegion)
{
  build(builder, result, TypeRange{}, cond, withElseRegion);
}

LogicalResult IfOp::verify()
{
  if (getNumResults() != 0 && getElseRegion().empty()) return emitOpError("must have an else block if defining values");
  return success();
}

ParseResult IfOp::parse(OpAsmParser& parser, OperationState& result)
{
  result.regions.reserve(2);
  Region* thenRegion = result.addRegion();
  Region* elseRegion = result.addRegion();

  auto& builder = parser.getBuilder();
  OpAsmParser::UnresolvedOperand cond;
  Type i8Type = builder.getIntegerType(8);
  if (parser.parseOperand(cond) || parser.resolveOperand(cond, i8Type, result.operands)) return failure();
  if (parser.parseOptionalArrowTypeList(result.types)) return failure();
  if (parser.parseRegion(*thenRegion, {}, {})) return failure();
  IfOp::ensureTerminator(*thenRegion, parser.getBuilder(), result.location);

  if (!parser.parseOptionalKeyword("else")) {
    if (parser.parseRegion(*elseRegion, {}, {})) return failure();
    IfOp::ensureTerminator(*elseRegion, parser.getBuilder(), result.location);
  }

  if (parser.parseOptionalAttrDict(result.attributes)) return failure();
  return success();
}

void IfOp::print(OpAsmPrinter& p)
{
  bool printBlockTerminators = false;

  p << " " << getCondition();
  if (!getResults().empty()) {
    p << " -> (" << getResultTypes() << ")";
    printBlockTerminators = true;
  }
  p << ' ';
  p.printRegion(getThenRegion(), false, printBlockTerminators);

  auto& elseRegion = getElseRegion();
  if (!elseRegion.empty()) {
    p << " else ";
    p.printRegion(elseRegion, false, printBlockTerminators);
  }
  p.printOptionalAttrDict((*this)->getAttrs());
}
