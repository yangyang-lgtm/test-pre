#include "privateuse/Dialect/Puc/IR/PucDialect.h"

#include "llvm/ADT/APSInt.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/ErrorHandling.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Interfaces/FunctionImplementation.h"
#include "mlir/Interfaces/LoopLikeInterface.h"
#include "privateuse/Dialect/PUMem/IR/PUMemDialect.h"

#define GET_DIALECT_DEFS
#include "privateuse/Dialect/Puc/IR/PucDialect.cpp.inc"

#define GET_OP_CLASSES
#include "privateuse/Dialect/Puc/IR/PucOps.cpp.inc"

using namespace mlir;
using namespace mlir::puc;

void mlir::puc::PucDialect::initialize()
{
  addOperations<
#define GET_OP_LIST
#include "privateuse/Dialect/Puc/IR/PucOps.cpp.inc"
    >();
}

Operation* mlir::puc::PucDialect::materializeConstant(OpBuilder& builder, Attribute value, Type type, Location loc)
{
  return ConstantOp::materialize(builder, value, type, loc);
}

bool ConstantOp::isBuildableWith(Attribute value, Type type)
{
  auto typedAttr = dyn_cast<TypedAttr>(value);
  if (!typedAttr || typedAttr.getType() != type) return false;
  if (auto intType = dyn_cast<IntegerType>(getElementTypeOrSelf(type))) {
    if (!intType.isSignless()) return false;
  }
  return llvm::isa<IntegerAttr, FloatAttr, ElementsAttr>(value);
}

ConstantOp ConstantOp::materialize(OpBuilder& builder, Attribute value, Type type, Location loc)
{
  if (isBuildableWith(value, type)) return ConstantOp::create(builder, loc, type, cast<TypedAttr>(value));
  return nullptr;
}

OpFoldResult ConstantOp::fold(FoldAdaptor adaptor)
{
  return getValue();
}

LogicalResult ConstantOp::verify()
{
  auto type = getType();
  if (auto intType = dyn_cast<IntegerType>(getElementTypeOrSelf(type)); intType && !intType.isSignless())
    return emitOpError("integer return type must be signless");
  if (!llvm::isa<IntegerAttr, FloatAttr, ElementsAttr>(getValue()))
    return emitOpError("value must be an integer, float, or elements attribute");
  return success();
}

LogicalResult AddPtrOp::verify()
{
  if (isa<pumem::VoidType>(cast<pumem::PtrType>(getPtr().getType()).getElemType()))
    return emitOpError("cannot do pointer arithmetic on a void pointer");
  return success();
}

OpFoldResult AddPtrOp::fold(FoldAdaptor adaptor)
{
  if (auto offset = dyn_cast_or_null<IntegerAttr>(adaptor.getOffset()))
    if (offset.getInt() == 0) return getPtr();
  return {};
}

OpFoldResult PointCastOp::fold(FoldAdaptor adaptor)
{
  if (getPtr().getType() == getResult().getType()) return getPtr();
  if (auto inner = getPtr().getDefiningOp<PointCastOp>()) {
    if (inner.getPtr().getType() == getResult().getType()) return inner.getPtr();
  }
  return {};
}

LogicalResult LoadOp::verify()
{
  auto ptrType = cast<pumem::PtrType>(getPtr().getType());
  if (isa<pumem::VoidType>(ptrType.getElemType())) return emitOpError("cannot load through a void pointer");
  if (ptrType.getElemType() != getScalar().getType())
    return emitOpError("result type must match the pointer element type");
  return success();
}

LogicalResult StoreOp::verify()
{
  auto ptrType = cast<pumem::PtrType>(getPtr().getType());
  if (isa<pumem::VoidType>(ptrType.getElemType())) return emitOpError("cannot store through a void pointer");
  if (ptrType.getElemType() != getScalar().getType())
    return emitOpError("value type must match the pointer element type");
  return success();
}

void FuncOp::build(
  OpBuilder& builder, OperationState& state, StringRef name, FunctionType type, ArrayRef<NamedAttribute> attrs,
  ArrayRef<DictionaryAttr> argAttrs
)
{
  state.addAttribute(SymbolTable::getSymbolAttrName(), builder.getStringAttr(name));
  state.addAttribute(getFunctionTypeAttrName(state.name), TypeAttr::get(type));
  state.addAttributes(attrs);
  state.addRegion();

  if (!argAttrs.empty()) {
    if (!(type.getNumInputs() == argAttrs.size())) llvm::report_fatal_error("type.getNumInputs() == argAttrs.size()");
    call_interface_impl::addArgAndResultAttrs(
      builder, state, argAttrs, {}, getArgAttrsAttrName(state.name), getResAttrsAttrName(state.name)
    );
  }
}

LogicalResult FuncOp::verifyRegions()
{
  if (isDeclaration()) return success();

  auto resultTypes = getFunctionType().getResults();
  for (Block& block : getBody()) {
    if (block.empty()) continue;
    auto returnOp = dyn_cast<RegionBranchTerminatorOpInterface>(&block.back());
    if (!returnOp) continue;
    auto operands = returnOp.getMutableSuccessorOperands(RegionSuccessor(getOperation()));
    if (operands.size() != resultTypes.size())
      return returnOp->emitOpError("has ") << operands.size() << " operands, but enclosing function (@" << getName()
                                           << ") returns " << resultTypes.size();

    for (auto [i, opType] : llvm::enumerate(llvm::zip(operands, resultTypes))) {
      auto [operand, resTy] = opType;
      if (operand.get().getType() != resTy)
        return returnOp->emitError() << "type of return operand " << i << " (" << operand.get().getType()
                                     << ") doesn't match function "
                                     << "result type (" << resTy << ") in function @" << getName();
    }
  }

  return success();
}

ParseResult FuncOp::parse(OpAsmParser& parser, OperationState& result)
{
  auto buildFuncType =
    [](Builder& builder, ArrayRef<Type> argTypes, ArrayRef<Type> results, function_interface_impl::VariadicFlag, std::string&) {
      return builder.getFunctionType(argTypes, results);
    };

  return function_interface_impl::parseFunctionOp(
    parser, result, false, getFunctionTypeAttrName(result.name), buildFuncType, getArgAttrsAttrName(result.name),
    getResAttrsAttrName(result.name)
  );
}

void FuncOp::print(OpAsmPrinter& p)
{
  function_interface_impl::printFunctionOp(
    p, *this, false, getFunctionTypeAttrName(), getArgAttrsAttrName(), getResAttrsAttrName()
  );
}

ParseResult CallOp::parse(OpAsmParser& parser, OperationState& result)
{
  FlatSymbolRefAttr callee;
  if (parser.parseCustomAttributeWithFallback(callee, parser.getBuilder().getType<NoneType>())) return failure();
  if (callee) result.getOrAddProperties<CallOp::Properties>().callee = callee;

  if (succeeded(parser.parseOptionalLess())) {
    ArrayAttr templateArgs;
    if (parser.parseCustomAttributeWithFallback(templateArgs, parser.getBuilder().getType<NoneType>()))
      return failure();
    if (templateArgs) result.getOrAddProperties<CallOp::Properties>().templateArgs = templateArgs;
    if (parser.parseGreater()) return failure();
  }

  if (parser.parseLParen()) return failure();

  SMLoc operandsLoc = parser.getCurrentLocation();
  SmallVector<OpAsmParser::UnresolvedOperand> operands;
  if (parser.parseOperandList(operands) || parser.parseRParen()) return failure();

  if (parser.parseOptionalAttrDict(result.attributes)) return failure();

  if (parser.parseColon()) return failure();

  FunctionType type;
  if (parser.parseType(type)) return failure();
  result.addTypes(type.getResults());
  return parser.resolveOperands(operands, type.getInputs(), operandsLoc, result.operands);
}

void CallOp::print(OpAsmPrinter& p)
{
  p << ' ';
  p.printAttributeWithoutType(getCalleeAttr());
  if (ArrayAttr templateArgs = getTemplateArgsAttr()) {
    p << '<';
    p.printAttributeWithoutType(templateArgs);
    p << '>';
  }
  p << '(';
  p.printOperands(getOperands());
  p << ')';
  p.printOptionalAttrDict((*this)->getAttrs(), {"callee", "templateArgs"});
  p << " : ";
  p.printFunctionalType(getOperands().getTypes(), getResultTypes());
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

ParseResult LocalOp::parse(OpAsmParser& parser, OperationState& result)
{
  OpAsmParser::UnresolvedOperand init;
  OptionalParseResult opt = parser.parseOptionalOperand(init);
  if (opt.has_value() && failed(*opt)) return failure();
  bool hasInit = opt.has_value();
  Type type;
  if (parser.parseOptionalAttrDict(result.attributes) || parser.parseColonType(type)) return failure();
  result.addTypes(type);
  if (hasInit && parser.resolveOperand(init, type, result.operands)) return failure();
  return success();
}

void LocalOp::print(OpAsmPrinter& p)
{
  p << ' ';
  if (getInit()) p << getInit() << ' ';
  p.printOptionalAttrDict((*this)->getAttrs());
  p << ": " << getResult().getType();
}
