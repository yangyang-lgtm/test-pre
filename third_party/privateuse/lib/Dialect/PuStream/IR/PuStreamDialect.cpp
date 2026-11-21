#include "privateuse/Dialect/PuStream/IR/PuStreamDialect.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/STLExtras.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Interfaces/ViewLikeInterface.h"
#include "privateuse/Utils/Constants.h"

#define GET_DIALECT_DEFS
#include "privateuse/Dialect/PuStream/IR/PuStreamDialect.cpp.inc"

#define GET_TYPEDEF_CLASSES
#include "privateuse/Dialect/PuStream/IR/PuStreamTypes.cpp.inc"

#define GET_OP_CLASSES
#include "privateuse/Dialect/PuStream/IR/PuStreamOps.cpp.inc"

using namespace mlir;
using namespace mlir::pustream;

static bool hasSpace(MemRefType type, int64_t space)
{
  Attribute attr = type.getMemorySpace();
  if (!attr) return space == triton::kDramSpace;
  auto integer = dyn_cast<IntegerAttr>(attr);
  return integer && integer.getInt() == space;
}

static bool isSram(Type type)
{
  auto memref = dyn_cast<MemRefType>(type);
  return memref && hasSpace(memref, triton::kSramSpace);
}

static bool isDram(Type type)
{
  auto memref = dyn_cast<MemRefType>(type);
  return memref && hasSpace(memref, triton::kDramSpace);
}

static bool isToken(Type type)
{
  return isa<StreamTokenType>(type);
}

static bool isDirty(Value token)
{
  Operation* producer = token.getDefiningOp();
  return producer && isa<TransferOp, CopyOp, DispatchOp>(producer);
}

static LogicalResult requireClean(Operation* op, Value token, StringRef what)
{
  if (!isDirty(token)) return success();
  return op->emitOpError() << what << " is still in flight (" << token.getDefiningOp()->getName()
                           << " wrote it); await it before reusing the allocation";
}

static Value allocationOf(Value token)
{
  while (true) {
    Operation* def = token.getDefiningOp();
    if (!def) return {};
    if (isa<AllocOp>(def)) return token;
    if (auto await = dyn_cast<AwaitOp>(def)) {
      token = await.getToken();
      continue;
    }
    if (auto awaitCompute = dyn_cast<AwaitComputeOp>(def)) {
      token = awaitCompute.getTokens()[cast<OpResult>(token).getResultNumber()];
      continue;
    }
    if (auto transfer = dyn_cast<TransferOp>(def)) {
      token = transfer.getToken();
      continue;
    }
    if (auto copy = dyn_cast<CopyOp>(def)) {
      token = copy.getTargetToken();
      continue;
    }
    if (auto dispatch = dyn_cast<DispatchOp>(def)) {
      ValueRange outs = dispatch.getOuts();
      unsigned index  = cast<OpResult>(token).getResultNumber();
      if (outs.size() != 2 * dispatch.getResults().size()) return {};
      token = outs[2 * index];
      continue;
    }
    return {};
  }
}

using Slot = std::pair<Operation*, unsigned>;

struct AllocationSet
{
  llvm::DenseSet<Slot> slots;
  bool unknown = false;

  bool empty() const { return slots.empty(); }
  void add(Operation* alloc, unsigned odd) { slots.insert({alloc, odd}); }

  void merge(const AllocationSet& other, bool flip = false)
  {
    for (const Slot& slot : other.slots) slots.insert({slot.first, flip ? 1u - slot.second : slot.second});
    unknown |= other.unknown;
  }

  bool equals(const AllocationSet& other) const
  {
    if (slots.size() != other.slots.size()) return false;
    for (const Slot& slot : slots)
      if (!other.slots.contains(slot)) return false;
    return true;
  }
};

class AllocationResolver
{
public:
  AllocationSet resolve(Value value) { return walk(value, 0); }

private:
  AllocationSet walk(Value value, unsigned depth);
  llvm::SmallPtrSet<Value, 16> active_;
};

AllocationSet AllocationResolver::walk(Value value, unsigned depth)
{
  AllocationSet set;
  if (depth > 32) {
    set.unknown = true;
    return set;
  }

  if (auto argument = dyn_cast<BlockArgument>(value)) {
    Operation* parent = argument.getOwner()->getParentOp();
    if (!parent) {
      set.unknown = true;
      return set;
    }
    unsigned index = argument.getArgNumber();
    if (auto loop = dyn_cast<scf::ForOp>(parent)) {
      unsigned iterIndex = index - loop.getNumInductionVars();
      if (iterIndex >= loop.getInitArgs().size()) {
        set.unknown = true;
        return set;
      }
      if (!active_.insert(value).second) return set;
      set.merge(walk(loop.getInitArgs()[iterIndex], depth + 1));
      auto yield = cast<scf::YieldOp>(loop.getBody()->getTerminator());
      set.merge(walk(yield.getOperand(iterIndex), depth + 1), /*flip=*/true);
      active_.erase(value);
      return set;
    }
    Operation* terminator = argument.getOwner()->getTerminator();
    if (!terminator || index >= terminator->getNumOperands()) {
      set.unknown = true;
      return set;
    }
    if (!active_.insert(value).second) return set;
    set.merge(walk(terminator->getOperand(index), depth + 1));
    active_.erase(value);
    return set;
  }

  Operation* producer = value.getDefiningOp();
  if (!producer) {
    set.unknown = true;
    return set;
  }

  if (isa<AllocOp>(producer)) {
    set.add(producer, /*odd=*/false);
    return set;
  }
  if (auto view = dyn_cast<ViewLikeOpInterface>(producer)) return walk(view.getViewSource(), depth + 1);
  if (auto await = dyn_cast<AwaitOp>(producer)) return walk(await.getToken(), depth + 1);
  if (auto awaitCompute = dyn_cast<AwaitComputeOp>(producer))
    return walk(awaitCompute.getTokens()[cast<OpResult>(value).getResultNumber()], depth + 1);
  if (auto transfer = dyn_cast<TransferOp>(producer)) return walk(transfer.getToken(), depth + 1);
  if (auto copy = dyn_cast<CopyOp>(producer)) return walk(copy.getTargetToken(), depth + 1);
  if (auto dispatch = dyn_cast<DispatchOp>(producer)) {
    ValueRange outs = dispatch.getOuts();
    unsigned index  = cast<OpResult>(value).getResultNumber();
    if (outs.size() != 2 * dispatch.getResults().size()) {
      set.unknown = true;
      return set;
    }
    return walk(outs[2 * index], depth + 1);
  }
  if (auto loop = dyn_cast<scf::ForOp>(producer)) {
    unsigned index = cast<OpResult>(value).getResultNumber();
    if (!active_.insert(value).second) return set;
    set.merge(walk(loop.getInitArgs()[index], depth + 1));
    auto yield = cast<scf::YieldOp>(loop.getBody()->getTerminator());
    set.merge(walk(yield.getOperand(index), depth + 1), /*flip=*/true);
    active_.erase(value);
    return set;
  }
  if (auto ifOp = dyn_cast<scf::IfOp>(producer)) {
    unsigned index = cast<OpResult>(value).getResultNumber();
    if (!active_.insert(value).second) return set;
    for (Region& region : ifOp->getRegions()) {
      if (region.empty()) continue;
      if (auto yield = dyn_cast<scf::YieldOp>(region.front().getTerminator()))
        if (index < yield.getNumOperands()) set.merge(walk(yield.getOperand(index), depth + 1));
    }
    active_.erase(value);
    return set;
  }
  set.unknown = true;
  return set;
}

static bool belongsTo(Value token, Value view)
{
  while (auto viewLike = view.getDefiningOp<ViewLikeOpInterface>()) view = viewLike.getViewSource();

  AllocationResolver resolver;
  AllocationSet tokens = resolver.resolve(token);
  AllocationSet views  = resolver.resolve(view);
  if (tokens.empty() || views.empty() || tokens.unknown || views.unknown) return false;
  return tokens.equals(views);
}

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

LogicalResult AllocOp::verify()
{
  if (!isToken(getToken().getType())) return emitOpError("the first result must be a !pustream.token");

  auto memref = dyn_cast<MemRefType>(getMemref().getType());
  if (!memref) return emitOpError("the second result must be a ranked memref");
  if (!isSram(memref)) return emitOpError("the allocation must live in SRAM (memory space 128)");
  if (ShapedType::isDynamicShape(memref.getShape())) return emitOpError("the allocation must have a static shape");
  return success();
}

LogicalResult TransferOp::verify()
{
  if (!isToken(getToken().getType())) return emitOpError("the first operand must be a !pustream.token");

  auto source = dyn_cast<MemRefType>(getSource().getType());
  auto target = dyn_cast<MemRefType>(getTarget().getType());
  if (!source || !target) return emitOpError("both sides must be ranked memrefs");

  bool sourceSram = isSram(source);
  bool targetSram = isSram(target);
  if (sourceSram == targetSram)
    return emitOpError("exactly one side must live in SRAM: SRAM to SRAM is copy, DRAM to DRAM is not a statement");

  MemRefType sram = sourceSram ? source : target;
  MemRefType dram = sourceSram ? target : source;
  if (!isDram(dram)) return emitOpError("the other side must live in DRAM (memory space 0)");
  if (sram.getElementType() != dram.getElementType())
    return emitOpError("element types must match: ") << sram.getElementType() << " vs " << dram.getElementType();
  if (sram.getRank() != dram.getRank())
    return emitOpError("ranks must match: ") << sram.getRank() << " vs " << dram.getRank();

  Value view = sourceSram ? getSource() : getTarget();
  if (!belongsTo(getToken(), view)) return emitOpError("the SRAM side is not a view of this token's allocation");
  if (getResult().getType() != getToken().getType()) return emitOpError("the result must be the token");
  return requireClean(*this, getToken(), "the token");
}

LogicalResult CopyOp::verify()
{
  if (!isToken(getSourceToken().getType()) || !isToken(getTargetToken().getType()))
    return emitOpError("both tokens must be !pustream.token");

  auto source = dyn_cast<MemRefType>(getSource().getType());
  auto target = dyn_cast<MemRefType>(getTarget().getType());
  if (!source || !target) return emitOpError("both sides must be ranked memrefs");
  if (!isSram(source) || !isSram(target)) return emitOpError("both sides must live in SRAM (memory space 128)");
  if (source.getElementType() != target.getElementType())
    return emitOpError("element types must match: ") << source.getElementType() << " vs " << target.getElementType();
  if (source.getRank() != target.getRank())
    return emitOpError("ranks must match: ") << source.getRank() << " vs " << target.getRank();

  if (!belongsTo(getSourceToken(), getSource()))
    return emitOpError("the source view is not a view of the source token's allocation");
  if (!belongsTo(getTargetToken(), getTarget()))
    return emitOpError("the target view is not a view of the target token's allocation");
  if (allocationOf(getSourceToken()) == allocationOf(getTargetToken()))
    return emitOpError("a copy needs two allocations; copying an allocation onto itself is not a statement");

  if (getResult().getType() != getTargetToken().getType()) return emitOpError("the result must be the target token");
  if (failed(requireClean(*this, getSourceToken(), "the source token"))) return failure();
  return requireClean(*this, getTargetToken(), "the target token");
}

LogicalResult DispatchOp::verify()
{
  ValueRange ins = getIns();

  unsigned index = 0;
  for (; index + 1 < ins.size() && isToken(ins[index].getType()); index += 2) {
    Value token = ins[index];
    auto view   = dyn_cast<MemRefType>(ins[index + 1].getType());
    if (!view) return emitOpError("a token input must be followed by the view it works on");
    if (!isSram(view)) return emitOpError("an input view must live in SRAM (memory space 128)");
    if (!belongsTo(token, ins[index + 1]))
      return emitOpError("an input view does not belong to its token's allocation");
    if (failed(requireClean(*this, token, "an input token"))) return failure();
  }
  for (; index < ins.size(); ++index)
    if (!ins[index].getType().isIntOrIndexOrFloat())
      return emitOpError("only scalars may follow the input token/view pairs");

  ValueRange outs = getOuts();
  if (outs.size() % 2 != 0) return emitOpError("every output must be a (token, view) pair");
  if (getResults().size() * 2 != outs.size()) return emitOpError("expected one result token per output pair");
  for (unsigned i = 0; i < outs.size(); i += 2) {
    if (!isToken(outs[i].getType())) return emitOpError("an output pair must start with a token");
    auto view = dyn_cast<MemRefType>(outs[i + 1].getType());
    if (!view) return emitOpError("an output pair must end with the view it writes");
    if (!isSram(view)) return emitOpError("an output view must live in SRAM (memory space 128)");
    if (!belongsTo(outs[i], outs[i + 1]))
      return emitOpError("an output view does not belong to its token's allocation");
    if (failed(requireClean(*this, outs[i], "an output token"))) return failure();
    if (getResults()[i / 2].getType() != outs[i].getType()) return emitOpError("the result must be the output token");
  }

  for (unsigned i = 0; i + 1 < ins.size() && isToken(ins[i].getType()); i += 2) {
    Value inAllocation = allocationOf(ins[i]);
    if (!inAllocation) continue;
    for (unsigned j = 0; j < outs.size(); j += 2)
      if (allocationOf(outs[j]) == inAllocation)
        return emitOpError("an input and an output share an allocation; a dispatch cannot read and write "
                           "the same buffer in place");
  }
  return success();
}

ParseResult DispatchOp::parse(OpAsmParser& parser, OperationState& result)
{
  FlatSymbolRefAttr callee;
  if (parser.parseAttribute(callee)) return failure();
  result.addAttribute("callee", callee);

  ArrayAttr templateArgs;
  if (succeeded(parser.parseOptionalLess())) {
    if (parser.parseAttribute(templateArgs) || parser.parseGreater()) return failure();
    result.addAttribute("templateArgs", templateArgs);
  }

  SmallVector<OpAsmParser::UnresolvedOperand> ins, outs;
  SmallVector<Type> inputs, results;
  if (parser.parseOperandList(ins, OpAsmParser::Delimiter::Paren)) return failure();
  if (parser.parseOperandList(outs, OpAsmParser::Delimiter::Paren)) return failure();
  if (parser.parseOptionalAttrDict(result.attributes)) return failure();

  if (parser.parseColon() || parser.parseLParen() || parser.parseTypeList(inputs) || parser.parseRParen())
    return failure();
  if (parser.parseArrow()) return failure();
  if (succeeded(parser.parseOptionalLParen())) {
    if (parser.parseTypeList(results) || parser.parseRParen()) return failure();
  } else {
    Type resultType;
    if (parser.parseType(resultType)) return failure();
    results.push_back(resultType);
  }

  if (inputs.size() != ins.size() + outs.size())
    return parser.emitError(parser.getCurrentLocation(), "expected one type per operand");
  if (outs.size() % 2 != 0)
    return parser.emitError(parser.getCurrentLocation(), "every output must be a (token, view) pair");

  if (results.size() != outs.size() / 2)
    return parser.emitError(parser.getCurrentLocation(), "expected one result token per output pair");
  for (unsigned i = 0; i < outs.size(); i += 2)
    if (results[i / 2] != inputs[ins.size() + i])
      return parser.emitError(parser.getCurrentLocation(), "expected the result to be its output token");

  ArrayRef<Type> inputTypes(inputs);
  if (parser.resolveOperands(ins, inputTypes.take_front(ins.size()), parser.getCurrentLocation(), result.operands))
    return failure();
  if (parser.resolveOperands(outs, inputTypes.drop_front(ins.size()), parser.getCurrentLocation(), result.operands))
    return failure();
  for (Type type : results) result.addTypes(type);
  result.addAttribute(
    "operandSegmentSizes",
    parser.getBuilder().getDenseI32ArrayAttr({static_cast<int32_t>(ins.size()), static_cast<int32_t>(outs.size())})
  );
  return success();
}

void DispatchOp::print(OpAsmPrinter& printer)
{
  printer << " @" << getCallee();
  if (auto templateArgs = getTemplateArgsAttr()) printer << "<" << templateArgs << ">";
  printer << "(";
  llvm::interleaveComma(getIns(), printer);
  printer << ")(";
  llvm::interleaveComma(getOuts(), printer);
  printer << ")";
  printer.printOptionalAttrDict((*this)->getAttrs(), {"callee", "templateArgs", "operandSegmentSizes"});
  SmallVector<Type> inputs(getIns().getTypes());
  inputs.append(getOuts().getTypes().begin(), getOuts().getTypes().end());
  printer << " : (";
  llvm::interleaveComma(inputs, printer);
  printer << ") -> ";
  if (getResults().size() == 1) {
    printer << getResults().front().getType();
  } else {
    printer << "(";
    llvm::interleaveComma(getResults().getTypes(), printer);
    printer << ")";
  }
}

LogicalResult AwaitOp::verify()
{
  if (!isToken(getToken().getType())) return emitOpError("the operand must be a !pustream.token");
  if (getResult().getType() != getToken().getType()) return emitOpError("the result must be the same token type");

  if (isa_and_nonnull<DispatchOp>(getToken().getDefiningOp()))
    return emitOpError("this token is written by a dispatch; the runtime has no per-buffer wait for "
                       "the compute queue, use await_compute");
  return success();
}

LogicalResult AwaitComputeOp::verify()
{
  if (getResults().size() != getTokens().size()) return emitOpError("expected one result per token");
  for (auto [result, token] : llvm::zip_equal(getResults(), getTokens())) {
    if (!isToken(token.getType())) return emitOpError("every operand must be a !pustream.token");
    if (result.getType() != token.getType()) return emitOpError("the result must be the same token type");
    if (isa_and_nonnull<TransferOp, CopyOp>(token.getDefiningOp()))
      return emitOpError("this token is written by the copy executor; await_compute does not wait it, use await");
  }
  return success();
}
