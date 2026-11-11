#include "privateuse/Analysis/AccessAnalysis.h"

#include "privateuse/Utils/Constants.h"
#include "privateuse/Utils/Utils.h"

#include "llvm/ADT/APInt.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/ADT/Twine.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "triton/Dialect/Triton/IR/Dialect.h"

#include <algorithm>
#include <optional>
#include <utility>

namespace mlir {
namespace triton {

namespace {

LogicalResult emitFailure(Location loc, const Twine& message)
{
  emitError(loc) << message;
  return failure();
}

bool getConstInt(Value value, int64_t& out)
{
  if (auto cst = value.getDefiningOp<arith::ConstantOp>()) {
    if (auto attr = dyn_cast<IntegerAttr>(cst.getValue())) {
      out = attr.getInt();
      return true;
    }
  }
  return false;
}

bool getConstBool(Value value, bool& out)
{
  int64_t integer = 0;
  if (!getConstInt(value, integer)) return false;
  out = integer != 0;
  return true;
}

SmallVector<int64_t> shapeOf(Type type)
{
  SmallVector<int64_t> shape;
  if (auto tensor = dyn_cast<RankedTensorType>(type)) shape.assign(tensor.getShape().begin(), tensor.getShape().end());
  return shape;
}

SmallVector<AccessDim> scalarDims(ArrayRef<int64_t> shape)
{
  SmallVector<AccessDim> dims;
  dims.reserve(shape.size());
  for (int64_t size : shape) dims.push_back(AccessDim{ScalarExpr::cst(0), size, 0});
  return dims;
}

bool areDimsStrideless(const SmallVector<AccessDim>& dims)
{
  for (const AccessDim& dim : dims)
    if (dim.stride != 0 || !dim.start || !dim.start->isConst()) return false;
  return true;
}

bool dependsOnLoopIv(const ExprPtr& expr, unsigned id)
{
  if (!expr) return false;
  if (expr->kind == ScalarExpr::Kind::Sym) return expr->symKind == SymKind::LoopIv && expr->symId == id;
  return dependsOnLoopIv(expr->lhs, id) || dependsOnLoopIv(expr->rhs, id);
}

std::optional<int64_t> constantFactor(const AccessView& view)
{
  if (view.base || !areDimsStrideless(view.dims)) return std::nullopt;
  if (view.offset && !view.offset->isConst()) return std::nullopt;
  int64_t sum = view.offset ? view.offset->value : 0;
  for (const AccessDim& dim : view.dims)
    if (!checkedAdd(sum, dim.start->value, sum)) return std::nullopt;
  return sum;
}

bool isPointerLike(Type type)
{
  if (isa<triton::PointerType>(type)) return true;
  if (auto tensor = dyn_cast<RankedTensorType>(type)) return isa<triton::PointerType>(tensor.getElementType());
  return false;
}

Value integerCastSource(Value value)
{
  Operation* def = value.getDefiningOp();
  if (!def) return {};
  if (isa<arith::IndexCastOp, arith::ExtSIOp, arith::ExtUIOp, arith::TruncIOp>(def)) return def->getOperand(0);
  return {};
}

std::optional<ScalarExpr::Kind> scalarBinaryKind(Operation* op)
{
  if (isa<arith::AddIOp>(op)) return ScalarExpr::Kind::Add;
  if (isa<arith::SubIOp>(op)) return ScalarExpr::Kind::Sub;
  if (isa<arith::MulIOp>(op)) return ScalarExpr::Kind::Mul;
  if (isa<arith::DivSIOp>(op)) return ScalarExpr::Kind::DivS;
  if (isa<arith::DivUIOp>(op)) return ScalarExpr::Kind::DivU;
  if (isa<arith::RemSIOp>(op)) return ScalarExpr::Kind::RemS;
  if (isa<arith::RemUIOp>(op)) return ScalarExpr::Kind::RemU;
  if (isa<arith::ShLIOp>(op)) return ScalarExpr::Kind::Shl;
  if (isa<arith::ShRSIOp>(op)) return ScalarExpr::Kind::ShrS;
  if (isa<arith::ShRUIOp>(op)) return ScalarExpr::Kind::ShrU;
  if (isa<arith::MinSIOp>(op)) return ScalarExpr::Kind::MinS;
  if (isa<arith::MaxSIOp>(op)) return ScalarExpr::Kind::MaxS;
  if (isa<arith::MinUIOp>(op)) return ScalarExpr::Kind::MinU;
  if (isa<arith::MaxUIOp>(op)) return ScalarExpr::Kind::MaxU;
  if (isa<arith::AndIOp>(op)) return ScalarExpr::Kind::And;
  if (isa<arith::OrIOp>(op)) return ScalarExpr::Kind::Or;
  if (isa<arith::XOrIOp>(op)) return ScalarExpr::Kind::Xor;
  return std::nullopt;
}

int64_t elementByteWidth(Type pointee)
{
  if (auto integer = dyn_cast<IntegerType>(pointee)) return std::max<int64_t>(1, integer.getWidth() / 8);
  if (auto floating = dyn_cast<FloatType>(pointee)) return std::max<int64_t>(1, floating.getWidth() / 8);
  if (isa<triton::PointerType>(pointee)) return 8;
  return 0;
}

Interval addIntervals(Interval lhs, Interval rhs)
{
  Interval result;
  int64_t value = 0;
  if (lhs.hasLo && rhs.hasLo && checkedAdd(lhs.lo, rhs.lo, value)) {
    result.hasLo = true;
    result.lo    = value;
  }
  if (lhs.hasHi && rhs.hasHi && checkedAdd(lhs.hi, rhs.hi, value)) {
    result.hasHi = true;
    result.hi    = value;
  }
  return result;
}

Interval rangeOfExpr(const ExprPtr& expr, const SymbolBounds& bounds)
{
  return expr ? evaluateInterval(*expr, bounds) : Interval::point(0);
}

Interval rangeOfValue(const ExprPtr& offset, const SmallVector<AccessDim>& dims, const SymbolBounds& bounds)
{
  Interval total = rangeOfExpr(offset, bounds);
  for (const AccessDim& dim : dims) {
    total = addIntervals(total, rangeOfExpr(dim.start, bounds));
    Interval span;
    span.hasLo = true;
    span.hasHi = true;
    if (dim.stride > 0) {
      if (!checkedMul(dim.stride, dim.size - 1, span.hi)) return Interval::unknown();
    } else if (dim.stride < 0) {
      if (!checkedMul(dim.stride, dim.size - 1, span.lo)) return Interval::unknown();
    }
    total = addIntervals(total, span);
  }
  return total;
}

struct PredicateInfo
{
  bool swap;
  bool strict;
  bool isUnsigned;
};

std::optional<PredicateInfo> classifyPredicate(arith::CmpIPredicate pred)
{
  switch (pred) {
    case arith::CmpIPredicate::slt: return PredicateInfo{false, true, false};
    case arith::CmpIPredicate::sle: return PredicateInfo{false, false, false};
    case arith::CmpIPredicate::sgt: return PredicateInfo{true, true, false};
    case arith::CmpIPredicate::sge: return PredicateInfo{true, false, false};
    case arith::CmpIPredicate::ult: return PredicateInfo{false, true, true};
    case arith::CmpIPredicate::ule: return PredicateInfo{false, false, true};
    case arith::CmpIPredicate::ugt: return PredicateInfo{true, true, true};
    case arith::CmpIPredicate::uge: return PredicateInfo{true, false, true};
    default: return std::nullopt;
  }
}

std::optional<bool> foldConstantComparison(const AccessView& diff, bool strict, const SymbolBounds& bounds)
{
  ExprPtr total = diff.offset;
  for (const AccessDim& dim : diff.dims) total = ScalarExpr::add(total, dim.start);
  Interval interval = rangeOfExpr(total, bounds);
  if (strict ? (interval.hasHi && interval.hi < 0) : (interval.hasHi && interval.hi <= 0)) return true;
  if (strict ? (interval.hasLo && interval.lo >= 0) : (interval.hasLo && interval.lo > 0)) return false;
  return std::nullopt;
}

void evaluateBound(const LinearBound& bound, int64_t size, const SymbolBounds& bounds, bool& holds, bool& fails)
{
  int64_t span   = 0;
  bool spanKnown = checkedMul(bound.stride, size - 1, span);

  Interval atStart = rangeOfExpr(bound.offset, bounds);
  Interval atEnd   = Interval::unknown();
  if (spanKnown) atEnd = addIntervals(atStart, Interval::point(span));

  Interval atMin = bound.stride < 0 ? atEnd : atStart;
  Interval atMax = bound.stride < 0 ? atStart : atEnd;

  if (!bound.lower) {
    holds = bound.strict ? (atMax.hasHi && atMax.hi < 0) : (atMax.hasHi && atMax.hi <= 0);
    fails = bound.strict ? (atMin.hasLo && atMin.lo >= 0) : (atMin.hasLo && atMin.lo > 0);
  } else {
    holds = bound.strict ? (atMin.hasLo && atMin.lo > 0) : (atMin.hasLo && atMin.lo >= 0);
    fails = bound.strict ? (atMax.hasHi && atMax.hi <= 0) : (atMax.hasHi && atMax.hi < 0);
  }
}

FailureOr<AccessMask> makeScalarMask(Value value)
{
  bool truth = false;
  if (!getConstBool(value, truth)) return AccessMask::dynamicScalar();
  return AccessMask::uniform(truth);
}

std::optional<AccessInfo> deferredAccess(Operation* op)
{
  if (isa<AtomicRMWOp>(op)) return AccessInfo::makeDeferred(DeferredReason::Atomic);

  if (isa<DescriptorLoadOp, DescriptorStoreOp, DescriptorReduceOp, DescriptorGatherOp, DescriptorScatterOp>(op))
    return AccessInfo::makeDeferred(
      DeferredReason::UnsupportedDescriptor, isa<DescriptorLoadOp, DescriptorGatherOp>(op)
    );

  if (isa<GatherOp>(op)) return AccessInfo::makeDeferred(DeferredReason::DataDependentOffset, /*isLoad=*/true);

  return std::nullopt;
}

} // namespace

llvm::StringRef toString(DeferredReason reason)
{
  switch (reason) {
    case DeferredReason::None: return "none";
    case DeferredReason::NegativeStride: return "negative_stride";
    case DeferredReason::DataDependentOffset: return "data_dependent_offset";
    case DeferredReason::UnsupportedDescriptor: return "unsupported_descriptor";
    case DeferredReason::Atomic: return "atomic";
  }
  return "unknown";
}

AccessMask AccessMask::uniform(bool truth)
{
  AccessMask mask;
  mask.present  = true;
  mask.allTrue  = truth;
  mask.allFalse = !truth;
  return mask;
}

AccessMask AccessMask::dynamicScalar()
{
  AccessMask mask;
  mask.present       = true;
  mask.allTrue       = false;
  mask.scalarDynamic = true;
  return mask;
}

AccessInfo AccessInfo::makeDeferred(DeferredReason reason, bool isLoad)
{
  AccessInfo info;
  info.isLoad   = isLoad;
  info.deferred = reason;
  return info;
}

LogicalResult AccessView::normalize(Type resultType, Location loc)
{
  if (auto tensor = dyn_cast<RankedTensorType>(resultType)) {
    ArrayRef<int64_t> shape = tensor.getShape();
    if (dims.size() != shape.size()) return emitFailure(loc, "address rank does not match the result type");
    for (size_t d = 0; d < shape.size(); ++d) {
      int64_t size = shape[d];
      if (size < 0) return emitFailure(loc, "address has a dynamic dimension");
      if (dims[d].size == 0) {
        dims[d].size = size;
      } else if (dims[d].size != size) {
        return emitFailure(loc, "address shape does not match the result type");
      }
    }
    return success();
  }
  if (!dims.empty()) return emitFailure(loc, "scalar value carries tensor dims");
  return success();
}

AccessAnalyzer::AccessAnalyzer(ModuleOp module)
{
  if (auto grid = module->getAttrOfType<DenseI64ArrayAttr>(kGridAttr)) {
    for (int axis = 0; axis < 3 && axis < grid.size(); ++axis) {
      if (grid[axis] <= 0) continue;
      Interval interval;
      interval.hasLo    = true;
      interval.hasHi    = true;
      interval.lo       = 0;
      interval.hi       = grid[axis] - 1;
      bounds.pids[axis] = interval;
    }
  }
}

unsigned AccessAnalyzer::loopId(Operation* loop)
{
  auto it = loopIds.find(loop);
  if (it != loopIds.end()) return it->second;
  unsigned id   = nextLoopId++;
  loopIds[loop] = id;
  return id;
}

FailureOr<ExprPtr> AccessAnalyzer::buildScalar(Value value)
{
  if (auto cst = value.getDefiningOp<arith::ConstantOp>()) {
    if (auto attr = dyn_cast<IntegerAttr>(cst.getValue())) return ScalarExpr::cst(attr.getInt());
    return emitFailure(value.getLoc(), "expected an integer constant");
  }

  if (auto pid = value.getDefiningOp<GetProgramIdOp>()) {
    int axis = pid.getAxisAsInt();
    if (axis < 0 || axis >= 3) return emitFailure(value.getLoc(), "unsupported program id axis " + Twine(axis));
    return ScalarExpr::sym(static_cast<SymKind>(static_cast<int>(SymKind::PidX) + axis));
  }

  if (auto arg = dyn_cast<BlockArgument>(value)) {
    if (auto loop = dyn_cast<scf::ForOp>(arg.getOwner()->getParentOp()))
      if (arg == loop.getInductionVar()) return ScalarExpr::sym(SymKind::LoopIv, loopId(loop));
    return emitFailure(value.getLoc(), "unsupported block argument in address expression");
  }

  if (Value src = integerCastSource(value)) return buildScalar(src);

  Operation* def = value.getDefiningOp();
  if (!def) return emitFailure(value.getLoc(), "value has no definition");

  if (auto kind = scalarBinaryKind(def)) {
    auto lhs = buildScalar(def->getOperand(0));
    if (failed(lhs)) return failure();
    auto rhs = buildScalar(def->getOperand(1));
    if (failed(rhs)) return failure();
    return ScalarExpr::bin(*kind, std::move(*lhs), std::move(*rhs));
  }

  return emitFailure(value.getLoc(), "unsupported scalar expression '" + Twine(def->getName().getStringRef()) + "'");
}

FailureOr<AccessView> AccessAnalyzer::scale(AccessView value, int64_t factor, Location loc)
{
  if (value.base) return emitFailure(loc, "cannot scale a pointer");
  ExprPtr factorExpr = ScalarExpr::cst(factor);
  AccessView view;
  view.offset = value.offset ? ScalarExpr::mul(value.offset, factorExpr) : nullptr;
  view.dims   = std::move(value.dims);
  for (AccessDim& dim : view.dims) {
    dim.start = ScalarExpr::mul(dim.start, factorExpr);
    if (!checkedMul(dim.stride, factor, dim.stride)) return emitFailure(loc, "overflow while scaling strides");
  }
  return view;
}

FailureOr<AccessView>
AccessAnalyzer::combineViews(AccessView lhs, AccessView rhs, bool subtract, Type resultType, Location loc)
{
  if (lhs.base && rhs.base) return emitFailure(loc, "cannot combine two base pointers");
  if (!lhs.dims.empty() && !rhs.dims.empty() && lhs.dims.size() != rhs.dims.size())
    return emitFailure(loc, "rank mismatch in address arithmetic");

  AccessView view;
  view.base   = lhs.base ? lhs.base : rhs.base;
  view.offset = subtract ? ScalarExpr::sub(lhs.offset, rhs.offset) : ScalarExpr::add(lhs.offset, rhs.offset);

  size_t rank = std::max(lhs.dims.size(), rhs.dims.size());
  view.dims.resize(rank);
  for (size_t i = 0; i < rank; ++i) {
    AccessDim left  = i < lhs.dims.size() ? lhs.dims[i] : AccessDim{};
    AccessDim right = i < rhs.dims.size() ? rhs.dims[i] : AccessDim{};
    if (left.size && right.size && left.size != right.size)
      return emitFailure(loc, "shape mismatch in address arithmetic");
    view.dims[i].start = subtract ? ScalarExpr::sub(left.start, right.start) : ScalarExpr::add(left.start, right.start);
    int64_t stride     = 0;
    if (subtract ? !checkedSub(left.stride, right.stride, stride) : !checkedAdd(left.stride, right.stride, stride))
      return emitFailure(loc, "overflow while combining strides");
    view.dims[i].stride = stride;
    view.dims[i].size   = left.size ? left.size : right.size;
  }
  if (failed(view.normalize(resultType, loc))) return failure();
  return view;
}

FailureOr<AccessView> AccessAnalyzer::combineValues(Value lhs, Value rhs, bool subtract, Value result)
{
  auto left = buildValue(lhs);
  if (failed(left)) return failure();
  auto right = buildValue(rhs);
  if (failed(right)) return failure();
  return combineViews(std::move(*left), std::move(*right), subtract, result.getType(), result.getLoc());
}

FailureOr<AccessView> AccessAnalyzer::resolveIterArg(BlockArgument arg)
{
  Location loc = arg.getLoc();
  auto loop    = dyn_cast<scf::ForOp>(arg.getOwner()->getParentOp());
  if (!loop) return emitFailure(loc, "tensor block argument is not a loop iter_arg");

  unsigned argNumber = arg.getArgNumber();
  if (argNumber == 0 || argNumber > loop.getInitArgs().size())
    return emitFailure(loc, "loop induction variable used as a pointer");

  unsigned index = argNumber - 1;
  Value init     = loop.getInitArgs()[index];
  Value yielded  = cast<scf::YieldOp>(loop.getBody()->getTerminator()).getOperand(index);

  if (yielded == arg) return buildValue(init);

  Value delta;
  if (auto add = yielded.getDefiningOp<AddPtrOp>()) {
    if (add.getPtr() == arg)
      delta = add.getOffset();
    else
      return buildValue(yielded);
  } else if (auto add = yielded.getDefiningOp<arith::AddIOp>()) {
    if (add.getLhs() == arg)
      delta = add.getRhs();
    else if (add.getRhs() == arg)
      delta = add.getLhs();
    else
      return buildValue(yielded);
  } else {
    return buildValue(yielded);
  }

  auto step = buildValue(delta);
  if (failed(step)) return failure();
  if (step->base) return emitFailure(loc, "loop step carries a base pointer");

  unsigned ivId        = loopId(loop);
  bool stepDependsOnIv = dependsOnLoopIv(step->offset, ivId);
  for (const AccessDim& dim : step->dims) stepDependsOnIv |= dependsOnLoopIv(dim.start, ivId);
  if (stepDependsOnIv || !areDimsStrideless(step->dims)) {
    deferred = DeferredReason::DataDependentOffset;
    return failure();
  }

  auto initExpr = buildValue(init);
  if (failed(initExpr)) return failure();

  ExprPtr iv = ScalarExpr::sym(SymKind::LoopIv, ivId);
  AccessView view;
  view.base   = initExpr->base;
  view.offset = ScalarExpr::add(initExpr->offset, ScalarExpr::mul(iv, step->offset));

  size_t rank = std::max(initExpr->dims.size(), step->dims.size());
  view.dims.resize(rank);
  for (size_t i = 0; i < rank; ++i) {
    AccessDim base = i < initExpr->dims.size() ? initExpr->dims[i] : AccessDim{};
    AccessDim inc  = i < step->dims.size() ? step->dims[i] : AccessDim{};
    if (base.size && inc.size && base.size != inc.size)
      return emitFailure(loc, "shape mismatch across the loop-carried pointer");
    view.dims[i].start  = ScalarExpr::add(base.start, ScalarExpr::mul(iv, inc.start));
    view.dims[i].stride = base.stride + inc.stride;
    view.dims[i].size   = base.size ? base.size : inc.size;
  }

  if (failed(view.normalize(arg.getType(), loc))) return failure();
  if (!view.base) return emitFailure(loc, "loop-carried pointer has no base");
  return view;
}

FailureOr<AccessView> AccessAnalyzer::buildConstant(arith::ConstantOp op, ArrayRef<int64_t> shape)
{
  AccessView view;
  if (auto attr = dyn_cast<IntegerAttr>(op.getValue())) {
    view.offset = ScalarExpr::cst(attr.getInt());
  } else {
    auto dense = dyn_cast<DenseElementsAttr>(op.getValue());
    if (!dense.isSplat()) return emitFailure(op.getLoc(), "non-splat constant tensor");
    if (!dense.getElementType().isIntOrIndex()) return emitFailure(op.getLoc(), "non-integer constant tensor");
    view.offset = ScalarExpr::cst(dense.getSplatValue<APInt>().getSExtValue());
  }
  view.dims = scalarDims(shape);
  if (failed(view.normalize(op.getType(), op.getLoc()))) return failure();
  return view;
}

FailureOr<AccessView> AccessAnalyzer::buildSplat(SplatOp op, ArrayRef<int64_t> shape)
{
  auto src = buildValue(op.getSrc());
  if (failed(src)) return failure();
  if (!src->dims.empty()) return emitFailure(op.getLoc(), "splat of a tensor");

  AccessView view;
  view.base   = src->base;
  view.offset = src->offset;
  view.dims   = scalarDims(shape);
  if (failed(view.normalize(op.getType(), op.getLoc()))) return failure();
  return view;
}

FailureOr<AccessView> AccessAnalyzer::buildMakeRange(MakeRangeOp op, ArrayRef<int64_t> shape)
{
  int64_t begin = op.getStartAttr().getInt();
  int64_t end   = op.getEndAttr().getInt();
  AccessView view;
  view.dims.push_back(AccessDim{ScalarExpr::cst(begin), shape.empty() ? end - begin : shape.front(), 1});
  if (failed(view.normalize(op.getType(), op.getLoc()))) return failure();
  return view;
}

FailureOr<AccessView> AccessAnalyzer::buildExpandDims(ExpandDimsOp op, ArrayRef<int64_t> shape)
{
  auto src = buildValue(op.getSrc());
  if (failed(src)) return failure();
  unsigned axis = op.getAxis();
  if (axis > src->dims.size()) return emitFailure(op.getLoc(), "expand_dims axis out of range");

  AccessView view = std::move(*src);
  view.dims.insert(view.dims.begin() + axis, AccessDim{ScalarExpr::cst(0), shape.empty() ? 1 : shape[axis], 0});
  if (failed(view.normalize(op.getType(), op.getLoc()))) return failure();
  return view;
}

FailureOr<AccessView> AccessAnalyzer::buildBroadcast(BroadcastOp op, ArrayRef<int64_t> shape)
{
  auto src = buildValue(op.getSrc());
  if (failed(src)) return failure();
  if (src->dims.size() != shape.size()) return emitFailure(op.getLoc(), "rank mismatch in broadcast");

  for (size_t d = 0; d < src->dims.size(); ++d) {
    if (src->dims[d].size == shape[d]) continue;
    if (src->dims[d].size != 1) return emitFailure(op.getLoc(), "broadcast changes a non-unit dimension");
    src->dims[d] = AccessDim{src->dims[d].start, shape[d], 0};
  }
  return src;
}

FailureOr<AccessView> AccessAnalyzer::buildMultiply(arith::MulIOp op)
{
  auto lhs = buildValue(op.getLhs());
  if (failed(lhs)) return failure();
  auto rhs = buildValue(op.getRhs());
  if (failed(rhs)) return failure();

  if (auto factor = constantFactor(*lhs)) return scale(std::move(*rhs), *factor, op.getLoc());
  if (auto factor = constantFactor(*rhs)) return scale(std::move(*lhs), *factor, op.getLoc());
  return emitFailure(op.getLoc(), "non-constant multiplication");
}

FailureOr<AccessView> AccessAnalyzer::buildBlockArgument(BlockArgument arg)
{
  if (isa<triton::PointerType>(arg.getType())) {
    AccessView view;
    view.base = arg;
    return view;
  }
  if (isa<RankedTensorType>(arg.getType())) return resolveIterArg(arg);
  return buildScalarLeaf(arg);
}

FailureOr<AccessView> AccessAnalyzer::buildScalarLeaf(Value value)
{
  auto offset = buildScalar(value);
  if (failed(offset)) return failure();
  AccessView view;
  view.offset = std::move(*offset);
  return view;
}

FailureOr<AccessView> AccessAnalyzer::buildValue(Value value)
{
  if (!resolving.insert(value).second) return emitFailure(value.getLoc(), "cyclic pointer expression");
  auto guard = llvm::scope_exit([&] { resolving.erase(value); });

  Operation* def = value.getDefiningOp();
  if (!def) return buildBlockArgument(cast<BlockArgument>(value));

  SmallVector<int64_t> shape = shapeOf(value.getType());

  if (auto cst = dyn_cast<arith::ConstantOp>(def))
    if (isa<IntegerAttr, DenseElementsAttr>(cst.getValue())) return buildConstant(cst, shape);
  if (auto splat = dyn_cast<SplatOp>(def)) return buildSplat(splat, shape);
  if (auto range = dyn_cast<MakeRangeOp>(def)) return buildMakeRange(range, shape);
  if (auto expand = dyn_cast<ExpandDimsOp>(def)) return buildExpandDims(expand, shape);
  if (auto broadcast = dyn_cast<BroadcastOp>(def)) return buildBroadcast(broadcast, shape);
  if (auto add = dyn_cast<AddPtrOp>(def))
    return combineValues(add.getPtr(), add.getOffset(), /*subtract=*/false, value);
  if (auto add = dyn_cast<arith::AddIOp>(def))
    return combineValues(add.getLhs(), add.getRhs(), /*subtract=*/false, value);
  if (auto sub = dyn_cast<arith::SubIOp>(def))
    return combineValues(sub.getLhs(), sub.getRhs(), /*subtract=*/true, value);
  if (auto mul = dyn_cast<arith::MulIOp>(def)) return buildMultiply(mul);

  if (auto bitcast = dyn_cast<BitcastOp>(def))
    if (isPointerLike(bitcast.getType()) && isPointerLike(bitcast.getSrc().getType()))
      return buildValue(bitcast.getSrc());

  if (Value src = integerCastSource(value)) return buildValue(src);

  if (isa<LoadOp, DescriptorLoadOp>(def)) return emitFailure(value.getLoc(), "pointer is loaded from memory");

  if (shape.empty()) return buildScalarLeaf(value);

  return emitFailure(
    value.getLoc(), "unsupported value in address expression: '" + Twine(def->getName().getStringRef()) + "'"
  );
}

FailureOr<AccessMask> AccessAnalyzer::buildMaskImpl(Value value, size_t rank)
{
  if (rank == 0) return makeScalarMask(value);

  auto tensor = dyn_cast<RankedTensorType>(value.getType());
  if (!tensor || !tensor.getElementType().isInteger(1))
    return emitFailure(value.getLoc(), "mask is not a tensor of i1");

  Operation* def = value.getDefiningOp();
  if (!def) return emitFailure(value.getLoc(), "unsupported mask expression");
  if (auto cst = dyn_cast<arith::ConstantOp>(def)) return buildConstantMask(cst);
  if (auto splat = dyn_cast<SplatOp>(def)) return buildSplatMask(splat);
  if (auto andi = dyn_cast<arith::AndIOp>(def)) {
    auto lhs = buildMask(andi.getLhs(), rank);
    if (failed(lhs)) return failure();
    auto rhs = buildMask(andi.getRhs(), rank);
    if (failed(rhs)) return failure();

    AccessMask mask;
    mask.present = true;
    mask.dimBounds.resize(rank);
    for (size_t d = 0; d < rank; ++d) {
      mask.dimBounds[d].append(lhs->dimBounds[d]);
      mask.dimBounds[d].append(rhs->dimBounds[d]);
    }
    mask.allTrue       = lhs->allTrue && rhs->allTrue;
    mask.allFalse      = lhs->allFalse || rhs->allFalse;
    mask.scalarDynamic = lhs->scalarDynamic || rhs->scalarDynamic;
    return mask;
  }
  if (auto cmp = dyn_cast<arith::CmpIOp>(def)) return buildCompareMask(cmp, rank);
  if (auto broadcast = dyn_cast<BroadcastOp>(def)) return buildBroadcastMask(broadcast, rank);
  if (auto expand = dyn_cast<ExpandDimsOp>(def)) return buildExpandDimsMask(expand, rank);

  return emitFailure(value.getLoc(), "unsupported mask expression '" + Twine(def->getName().getStringRef()) + "'");
}

FailureOr<AccessMask> AccessAnalyzer::buildMask(Value value, size_t rank)
{
  auto mask = buildMaskImpl(value, rank);
  if (failed(mask)) return failure();
  if (mask->dimBounds.size() < rank) mask->dimBounds.resize(rank);
  return mask;
}

FailureOr<AccessMask> AccessAnalyzer::buildConstantMask(arith::ConstantOp op)
{
  auto dense = dyn_cast<DenseElementsAttr>(op.getValue());
  if (!dense || !dense.isSplat()) return emitFailure(op.getLoc(), "mask is not a splat constant");
  return AccessMask::uniform(dense.getSplatValue<APInt>().isOne());
}

FailureOr<AccessMask> AccessAnalyzer::buildSplatMask(SplatOp op)
{
  bool truth = false;
  if (!getConstBool(op.getSrc(), truth)) return emitFailure(op.getLoc(), "dynamic scalar splat mask");
  return AccessMask::uniform(truth);
}

FailureOr<AccessMask> AccessAnalyzer::buildCompareMask(arith::CmpIOp op, size_t rank)
{
  Location loc   = op.getLoc();
  auto predicate = classifyPredicate(op.getPredicate());
  if (!predicate) return emitFailure(loc, "unsupported comparison predicate");

  auto lhs = buildValue(op.getLhs());
  if (failed(lhs)) return failure();
  auto rhs = buildValue(op.getRhs());
  if (failed(rhs)) return failure();
  if (lhs->base || rhs->base) return emitFailure(loc, "comparison on pointers");

  if (predicate->isUnsigned) {
    Interval lhsRange = rangeOfValue(lhs->offset, lhs->dims, bounds);
    Interval rhsRange = rangeOfValue(rhs->offset, rhs->dims, bounds);
    if (!lhsRange.hasLo || lhsRange.lo < 0 || !rhsRange.hasLo || rhsRange.lo < 0)
      return emitFailure(loc, "unsigned comparison with possibly negative operands");
  }

  auto diff = predicate->swap
                ? combineViews(std::move(*rhs), std::move(*lhs), /*subtract=*/true, op.getResult().getType(), loc)
                : combineViews(std::move(*lhs), std::move(*rhs), /*subtract=*/true, op.getResult().getType(), loc);
  if (failed(diff)) return failure();

  SmallVector<size_t> varying;
  for (size_t d = 0; d < diff->dims.size(); ++d)
    if (diff->dims[d].stride != 0) varying.push_back(d);

  AccessMask mask;
  mask.present = true;
  if (varying.empty()) {
    auto folded = foldConstantComparison(*diff, predicate->strict, bounds);
    if (!folded) return emitFailure(loc, "comparison does not constrain an index");
    mask.allTrue  = *folded;
    mask.allFalse = !*folded;
    return mask;
  }
  if (varying.size() > 1) return emitFailure(loc, "mask couples several dimensions");

  size_t dimIndex = varying.front();
  if (diff->dims[dimIndex].stride < 0) return emitFailure(loc, "mask involves a negative stride");

  ExprPtr offset = diff->offset;
  for (size_t d = 0; d < diff->dims.size(); ++d) {
    if (d == dimIndex) continue;
    offset = ScalarExpr::add(offset, diff->dims[d].start);
  }
  offset = ScalarExpr::add(offset, diff->dims[dimIndex].start);

  mask.dimBounds.resize(rank);
  mask.dimBounds[dimIndex].push_back(
    LinearBound{offset, diff->dims[dimIndex].stride, predicate->strict, /*lower=*/false}
  );
  mask.allTrue = false;
  return mask;
}

FailureOr<AccessMask> AccessAnalyzer::buildBroadcastMask(BroadcastOp op, size_t rank)
{
  auto src = buildMask(op.getSrc(), rank);
  if (failed(src)) return failure();

  auto srcType = dyn_cast<RankedTensorType>(op.getSrc().getType());
  if (!srcType || srcType.getShape().size() != rank) return emitFailure(op.getLoc(), "rank mismatch in mask broadcast");

  auto resultType = cast<RankedTensorType>(op.getType());
  for (size_t d = 0; d < rank; ++d)
    if (srcType.getShape()[d] != resultType.getShape()[d]) src->dimBounds[d].clear();
  return src;
}

FailureOr<AccessMask> AccessAnalyzer::buildExpandDimsMask(ExpandDimsOp op, size_t rank)
{
  auto src = buildMask(op.getSrc(), rank - 1);
  if (failed(src)) return failure();
  unsigned axis = op.getAxis();
  if (axis > src->dimBounds.size()) return emitFailure(op.getLoc(), "expand_dims axis out of range in mask");

  src->dimBounds.insert(src->dimBounds.begin() + axis, SmallVector<LinearBound>{});
  return src;
}

void AccessAnalyzer::classifyMask(AccessMask& mask, ArrayRef<int64_t> sizes)
{
  if (!mask.present) {
    mask.allTrue  = true;
    mask.allFalse = false;
    return;
  }
  if (mask.scalarDynamic) {
    mask.allTrue = false;
    return;
  }

  bool allTrue  = true;
  bool anyFalse = false;
  for (size_t d = 0; d < mask.dimBounds.size(); ++d) {
    int64_t size = d < sizes.size() ? sizes[d] : 1;
    if (size <= 0) continue;
    for (const LinearBound& bound : mask.dimBounds[d]) {
      bool holds = false;
      bool fails = false;
      evaluateBound(bound, size, bounds, holds, fails);
      if (!holds) allTrue = false;
      if (fails) anyFalse = true;
    }
  }
  mask.allTrue  = allTrue;
  mask.allFalse = anyFalse;
}

FailureOr<AccessInfo> AccessAnalyzer::analyzeAccess(Value ptr, Value mask, bool isLoad, Location loc)
{
  auto view = buildValue(ptr);
  if (failed(view)) {
    if (deferred != DeferredReason::None) return AccessInfo::makeDeferred(deferred, isLoad);
    return failure();
  }
  if (!view->base || !isa<BlockArgument>(view->base)) return emitFailure(loc, "base pointer is not a kernel argument");

  auto baseType = dyn_cast<triton::PointerType>(view->base.getType());
  if (!baseType) return emitFailure(loc, "base is not a scalar pointer");

  AccessInfo info;
  info.isLoad    = isLoad;
  info.elemBytes = elementByteWidth(baseType.getPointeeType());
  info.view      = std::move(*view);
  if (failed(info.view.normalize(ptr.getType(), loc))) return failure();

  SmallVector<int64_t> sizes = shapeOf(ptr.getType());
  if (mask) {
    auto builtMask = buildMask(mask, sizes.size());
    if (failed(builtMask)) {
      if (deferred != DeferredReason::None) return AccessInfo::makeDeferred(deferred, isLoad);
      return failure();
    }
    info.mask = std::move(*builtMask);
  }
  classifyMask(info.mask, sizes);

  for (const AccessDim& dim : info.view.dims) {
    if (dim.stride < 0) {
      info.deferred = DeferredReason::NegativeStride;
      break;
    }
  }
  return info;
}

FailureOr<AccessInfo> AccessAnalyzer::analyze(Operation* op)
{
  deferred = DeferredReason::None;
  if (auto info = deferredAccess(op)) return *info;

  if (auto load = dyn_cast<LoadOp>(op))
    return analyzeAccess(load.getPtr(), load.getMask(), /*isLoad=*/true, op->getLoc());
  if (auto store = dyn_cast<StoreOp>(op))
    return analyzeAccess(store.getPtr(), store.getMask(), /*isLoad=*/false, op->getLoc());
  return emitFailure(op->getLoc(), "not a memory operation");
}

} // namespace triton
} // namespace mlir
