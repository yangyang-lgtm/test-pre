#include "privateuse/Dialect/PuStream/IR/PuStreamDialect.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/TypeSwitch.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "privateuse/Utils/Constants.h"

#include <optional>

#define GET_DIALECT_DEFS
#include "privateuse/Dialect/PuStream/IR/PuStreamDialect.cpp.inc"

#define GET_TYPEDEF_CLASSES
#include "privateuse/Dialect/PuStream/IR/PuStreamTypes.cpp.inc"

#define GET_OP_CLASSES
#include "privateuse/Dialect/PuStream/IR/PuStreamOps.cpp.inc"

using namespace mlir;
using namespace mlir::pustream;

namespace {

std::optional<int64_t> memorySpace(MemRefType type)
{
  Attribute space = type.getMemorySpace();
  if (!space) return triton::kDramSpace;
  if (auto integer = dyn_cast<IntegerAttr>(space)) return integer.getInt();
  return std::nullopt;
}

bool isSram(MemRefType type)
{
  auto space = memorySpace(type);
  return space && *space == triton::kSramSpace;
}

bool isDram(MemRefType type)
{
  auto space = memorySpace(type);
  return space && *space == triton::kDramSpace;
}

std::optional<int64_t> declaredStages(Value value)
{
  Operation* def = value.getDefiningOp();
  if (!def) return std::nullopt;
  if (auto stages = def->getAttrOfType<IntegerAttr>(triton::kStagesAttr)) return stages.getInt();
  return std::nullopt;
}

std::optional<int64_t> constantIndex(Value index)
{
  Attribute value;
  if (!matchPattern(index, m_Constant(&value))) return std::nullopt;
  if (auto integer = dyn_cast<IntegerAttr>(value)) return integer.getInt();
  return std::nullopt;
}

MemRefType expectedTile(MemRefType buffer)
{
  return MemRefType::get(
    buffer.getShape().drop_front(), buffer.getElementType(), MemRefLayoutAttrInterface{}, buffer.getMemorySpace()
  );
}

Value slotBase(Value value)
{
  if (auto slot = value.getDefiningOp<SlotOp>()) return slot.getBuffer();
  return Value{};
}

} // namespace

void PuStreamDialect::initialize()
{
  addTypes<
#define GET_TYPEDEF_LIST
#include "privateuse/Dialect/PuStream/IR/PuStreamTypes.cpp.inc"
    >();
  addOperations<
#define GET_OP_LIST
#include "privateuse/Dialect/PuStream/IR/PuStreamOps.cpp.inc"
    >();
}

Type SlotType::parse(AsmParser& parser)
{
  MemRefType tile;
  if (parser.parseLess() || parser.parseType(tile) || parser.parseGreater()) return Type();
  return SlotType::get(parser.getContext(), tile);
}

void SlotType::print(AsmPrinter& printer) const
{
  printer << "!pustream.slot<" << getTile() << ">";
}

LogicalResult SlotOp::verify()
{
  auto buffer = dyn_cast<MemRefType>(getBuffer().getType());
  if (!buffer) return emitOpError("buffer must be a ranked memref");
  if (!isSram(buffer)) return emitOpError("buffer must live in SRAM (memory space 128)");

  auto tile = dyn_cast<SlotType>(getSlot().getType());
  if (!tile) return emitOpError("result must be a !pustream.slot");
  if (!isSram(tile.getTile())) return emitOpError("slot tile must live in SRAM");
  if (tile.getTile().getElementType() != buffer.getElementType())
    return emitOpError("slot tile element type must match the buffer element type");

  std::optional<int64_t> stages = declaredStages(getBuffer());
  if (!stages) {
    if (tile.getTile() != buffer) return emitOpError("a single-stage buffer's slot must have the buffer's own type");
    if (std::optional<int64_t> index = constantIndex(getIndex())) {
      if (*index != 0) return emitOpError("a single-stage buffer has only stage 0");
    } else {
      return emitOpError("a dynamic stage index requires a buffer with ") << triton::kStagesAttr;
    }
    return success();
  }

  if (buffer.getRank() < 2) return emitOpError("a staged buffer needs at least two dimensions (stage x tile)");
  if (*stages <= 0) return emitOpError() << triton::kStagesAttr << " must be positive";
  if (buffer.getShape().front() != *stages)
    return emitOpError() << triton::kStagesAttr << " (" << *stages << ") must equal dim 0 ("
                         << buffer.getShape().front() << ")";
  if (tile.getTile() != expectedTile(buffer))
    return emitOpError("slot tile type must be the buffer type without its stage dimension");
  if (std::optional<int64_t> index = constantIndex(getIndex())) {
    if (*index < 0 || *index >= *stages)
      return emitOpError("constant stage index ") << *index << " is outside [0, " << *stages << ")";
  }
  return success();
}

LogicalResult AsyncCopyOp::verify()
{
  const bool sourceIsSlot = isa<SlotType>(getSource().getType());
  const bool targetIsSlot = isa<SlotType>(getTarget().getType());
  if (sourceIsSlot == targetIsSlot)
    return emitOpError("exactly one operand must be a !pustream.slot; slot-to-slot and"
                       " memref-to-memref copies are not expressible");

  Value staged = sourceIsSlot ? getSource() : getTarget();
  Value dram   = sourceIsSlot ? getTarget() : getSource();

  auto dramType = dyn_cast<MemRefType>(dram.getType());
  if (!dramType) return emitOpError("the non-slot operand must be a ranked memref (the DRAM side)");
  if (!isDram(dramType)) return emitOpError("the non-slot operand must live in DRAM (memory space 0)");

  MemRefType tile = cast<SlotType>(staged.getType()).getTile();
  if (tile.getElementType() != dramType.getElementType())
    return emitOpError("element types must match: ") << tile.getElementType() << " vs " << dramType.getElementType();
  if (tile.getRank() != dramType.getRank())
    return emitOpError("box rank must match: ") << tile.getRank() << " vs " << dramType.getRank();

  for (auto [slotDim, dramDim] : llvm::zip_equal(tile.getShape(), dramType.getShape())) {
    if (ShapedType::isDynamic(slotDim) || ShapedType::isDynamic(dramDim)) continue;
    if (dramDim > slotDim)
      return emitOpError("DRAM box extent ") << dramDim << " exceeds the slot tile extent " << slotDim;
  }
  return success();
}

static void printBufferType(OpAsmPrinter& printer, Type type)
{
  if (auto slot = dyn_cast<SlotType>(type)) {
    printer << "!pustream.slot<";
    printer.printType(slot.getTile());
    printer << ">";
    return;
  }
  printer.printType(type);
}

ParseResult AsyncCopyOp::parse(OpAsmParser& parser, OperationState& result)
{
  OpAsmParser::UnresolvedOperand source, target;
  Type sourceType, targetType;
  if (parser.parseOperand(source) || parser.parseComma() || parser.parseOperand(target)) return failure();
  if (parser.parseOptionalAttrDict(result.attributes)) return failure();
  if (parser.parseColonType(sourceType) || parser.parseComma() || parser.parseType(targetType)) return failure();
  if (parser.resolveOperand(source, sourceType, result.operands)) return failure();
  return parser.resolveOperand(target, targetType, result.operands);
}

void AsyncCopyOp::print(OpAsmPrinter& printer)
{
  printer << " " << getSource() << ", " << getTarget();
  printer.printOptionalAttrDict((*this)->getAttrs());
  printer << " : ";
  printBufferType(printer, getSource().getType());
  printer << ", ";
  printBufferType(printer, getTarget().getType());
}

ParseResult AsyncDispatchOp::parse(OpAsmParser& parser, OperationState& result)
{
  FlatSymbolRefAttr callee;
  if (parser.parseAttribute(callee, "callee", result.attributes)) return failure();

  if (succeeded(parser.parseOptionalLess())) {
    ArrayAttr templateArgs;
    if (parser.parseAttribute(templateArgs) || parser.parseGreater()) return failure();
    result.addAttribute("templateArgs", templateArgs);
  }

  SmallVector<OpAsmParser::UnresolvedOperand> args;
  if (parser.parseLParen() || parser.parseOperandList(args) || parser.parseRParen()) return failure();
  if (parser.parseOptionalAttrDict(result.attributes)) return failure();

  SmallVector<Type> types;
  if (succeeded(parser.parseOptionalColon())) {
    if (parser.parseTypeList(types)) return failure();
  } else if (!args.empty()) {
    return parser.emitError(parser.getNameLoc(), "expected a type list for the dispatch operands");
  }
  return parser.resolveOperands(args, types, parser.getNameLoc(), result.operands);
}

void AsyncDispatchOp::print(OpAsmPrinter& printer)
{
  printer << " ";
  printer.printSymbolName(getCallee());
  if (Attribute templateArgs = getTemplateArgsAttr()) printer << "<" << templateArgs << ">";
  printer << "(";
  llvm::interleaveComma(getArgs(), printer, [&](Value arg) { printer << arg; });
  printer << ")";
  printer.printOptionalAttrDict((*this)->getAttrs(), {"callee", "templateArgs"});
  if (!getArgs().empty()) {
    printer << " : ";
    llvm::interleaveComma(getArgs(), printer, [&](Value arg) { printBufferType(printer, arg.getType()); });
  }
}

void AsyncCopyOp::getEffects(SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>& effects)
{
  effects.emplace_back(MemoryEffects::Read::get(), &getOperation()->getOpOperand(0));
  effects.emplace_back(MemoryEffects::Write::get(), &getOperation()->getOpOperand(1));
  effects.emplace_back(MemoryEffects::Write::get(), Ordering::get());
}

LogicalResult AsyncDispatchOp::verify()
{
  const int64_t numArgs = static_cast<int64_t>(getArgs().size());
  auto verifyIndex      = [&](int64_t index, StringRef list) -> LogicalResult {
    if (index < 0 || index >= numArgs) return emitOpError() << list << " index " << index << " is out of range";
    Type type = getArgs()[static_cast<size_t>(index)].getType();
    if (!isa<MemRefType, SlotType>(type))
      return emitOpError() << list << " index " << index << " does not refer to a buffer argument";
    return success();
  };
  for (int64_t index : getReads())
    if (failed(verifyIndex(index, "reads"))) return failure();
  for (int64_t index : getWrites())
    if (failed(verifyIndex(index, "writes"))) return failure();
  return success();
}

void AsyncDispatchOp::getEffects(SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>& effects)
{
  const int64_t numArgs = static_cast<int64_t>(getArgs().size());
  auto add              = [&](int64_t index, MemoryEffects::Effect* effect) {
    if (index < 0 || index >= numArgs) return;
    effects.emplace_back(effect, &getOperation()->getOpOperand(static_cast<unsigned>(index)));
  };
  for (int64_t index : getReads()) add(index, MemoryEffects::Read::get());
  for (int64_t index : getWrites()) add(index, MemoryEffects::Write::get());
  effects.emplace_back(MemoryEffects::Write::get(), Ordering::get());
}

LogicalResult WaitOp::verify()
{
  Value base = slotBase(getSlot());
  if (!base) return success();

  auto func = getOperation()->getParentOfType<FunctionOpInterface>();
  if (!func) return success();

  bool participant = false;
  func->walk([&](AsyncCopyOp copy) {
    for (Value operand : {copy.getSource(), copy.getTarget()})
      if (slotBase(operand) == base) {
        participant = true;
        return WalkResult::interrupt();
      }
    return WalkResult::advance();
  });
  if (!participant) return emitOpError("waits a slot whose buffer never participates in a copy");
  return success();
}

void WaitOp::getEffects(SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>& effects)
{
  effects.emplace_back(MemoryEffects::Read::get(), &getOperation()->getOpOperand(0));
  effects.emplace_back(MemoryEffects::Write::get(), Ordering::get());
}

void WaitDispatchOp::getEffects(SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>& effects)
{
  effects.emplace_back(MemoryEffects::Write::get(), Ordering::get());
}

LogicalResult SyncOp::verify()
{
  Operation* next = getOperation()->getNextNode();
  if (!next || !next->hasTrait<OpTrait::IsTerminator>())
    return emitOpError("must be the last statement of a kernel function");
  return success();
}

void SyncOp::getEffects(SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>& effects)
{
  effects.emplace_back(MemoryEffects::Write::get(), Ordering::get());
}
