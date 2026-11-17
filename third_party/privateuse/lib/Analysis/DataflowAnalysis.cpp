#include "privateuse/Analysis/DataflowAnalysis.h"

#include "privateuse/Analysis/ValueRange.h"
#include "privateuse/Dialect/PUMem/IR/PUMemDialect.h"
#include "privateuse/Utils/Constants.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/MathExtras.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/Interfaces/CallInterfaces.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Interfaces/ViewLikeInterface.h"
#include "triton/Dialect/Triton/IR/Dialect.h"

#include <algorithm>
#include <limits>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <utility>

namespace mlir {
namespace triton {

llvm::StringRef toString(Space space)
{
  switch (space) {
    case Space::Dram: return "dram";
    case Space::Sram: return "sram";
    case Space::Unknown: return "unknown";
  }
  return "unknown";
}

llvm::StringRef toString(PlaceKind kind)
{
  switch (kind) {
    case PlaceKind::KernelArg: return "kernel_arg";
    case PlaceKind::SramAlloc: return "sram_alloc";
    case PlaceKind::LoopCarried: return "loop_carried";
  }
  return "unknown";
}

llvm::StringRef toString(Dir dir)
{
  switch (dir) {
    case Dir::DramToSram: return "D2S";
    case Dir::SramToDram: return "S2D";
    case Dir::SramToSram: return "S2S";
  }
  return "?";
}

llvm::StringRef toString(OpKind kind)
{
  switch (kind) {
    case OpKind::Copy: return "copy";
    case OpKind::Gather: return "gather";
    case OpKind::Scatter: return "scatter";
  }
  return "?";
}

llvm::StringRef toString(Role role)
{
  switch (role) {
    case Role::Read: return "read";
    case Role::Write: return "write";
    case Role::ScalarRead: return "scalar_read";
    case Role::ScalarWrite: return "scalar_write";
    case Role::Atomic: return "atomic";
  }
  return "?";
}

llvm::StringRef toString(Branch branch)
{
  switch (branch) {
    case Branch::None: return "none";
    case Branch::Init: return "init";
    case Branch::Yield: return "yield";
  }
  return "none";
}

llvm::StringRef toString(Guard guard)
{
  switch (guard) {
    case Guard::None: return "none";
    case Guard::FirstIter: return "first";
    case Guard::BodyIter: return "body";
  }
  return "none";
}

llvm::StringRef toString(DataflowDeferredReason reason)
{
  switch (reason) {
    case DataflowDeferredReason::None: return "none";
    case DataflowDeferredReason::UnsupportedControlFlow: return "unsupported_control_flow";
    case DataflowDeferredReason::UnmodeledRegion: return "unmodeled_region";
    case DataflowDeferredReason::MultiplePidLoops: return "multiple_pid_loops";
    case DataflowDeferredReason::UnsupportedMemorySpace: return "unsupported_memory_space";
    case DataflowDeferredReason::SramAllocOutsidePid: return "sram_alloc_outside_pid";
    case DataflowDeferredReason::PidCarriesMemref: return "pid_carries_memref";
    case DataflowDeferredReason::NestedCarrier: return "nested_carrier";
    case DataflowDeferredReason::SymbolicTrip: return "symbolic_trip";
    case DataflowDeferredReason::UnresolvedMemref: return "unresolved_memref";
    case DataflowDeferredReason::TransferBetweenCarriers: return "transfer_between_carriers";
    case DataflowDeferredReason::DramToDramTransfer: return "dram_to_dram_transfer";
    case DataflowDeferredReason::MemrefDealloc: return "memref_dealloc";
    case DataflowDeferredReason::MemrefEscape: return "memref_escape";
    case DataflowDeferredReason::ScalarWriteToSram: return "scalar_write_to_sram";
    case DataflowDeferredReason::LoadDepthMismatch: return "load_depth_mismatch";
    case DataflowDeferredReason::StoreDepthMismatch: return "store_depth_mismatch";
    case DataflowDeferredReason::MultipleLoadProducers: return "multiple_load_producers";
    case DataflowDeferredReason::MultipleStoreConsumers: return "multiple_store_consumers";
    case DataflowDeferredReason::StationWithoutProducer: return "station_without_producer";
    case DataflowDeferredReason::Count: break;
  }
  return "unknown";
}

namespace {

using PhaseResult = FailureOr<DataflowDeferredReason>;

constexpr int64_t kInfinity = std::numeric_limits<int64_t>::max();

struct Site
{
  unsigned pathId    = 0;
  int64_t blockOrder = 0;
};

struct RootRef
{
  unsigned place    = 0;
  int64_t carrierId = -1;

  bool isCarrier() const { return carrierId >= 0; }
  bool operator==(const RootRef& other) const { return place == other.place && carrierId == other.carrierId; }
};

struct RootResolution
{
  std::optional<RootRef> root;
  DataflowDeferredReason deferred = DataflowDeferredReason::None;
};

bool isFailureOrDeferred(const PhaseResult& result)
{
  return failed(result) || *result != DataflowDeferredReason::None;
}

bool isWriteRole(Role role)
{
  return role == Role::Write || role == Role::ScalarWrite || role == Role::Atomic;
}

bool isReadRole(Role role)
{
  return role == Role::Read || role == Role::ScalarRead || role == Role::Atomic;
}

int roleRank(Role role)
{
  switch (role) {
    case Role::Read:
    case Role::ScalarRead: return 0;
    case Role::Atomic: return 1;
    case Role::Write:
    case Role::ScalarWrite: return 2;
  }
  return 3;
}

Space spaceOf(MemRefType type)
{
  Attribute space = type.getMemorySpace();
  if (!space) return Space::Dram;
  auto integer = dyn_cast<IntegerAttr>(space);
  if (!integer) return Space::Unknown;
  int64_t value = integer.getInt();
  if (value == kDramSpace) return Space::Dram;
  if (value == kSramSpace) return Space::Sram;
  return Space::Unknown;
}

bool hasDynamicDim(Type type)
{
  if (auto memref = dyn_cast<MemRefType>(type))
    for (int64_t dim : memref.getShape())
      if (ShapedType::isDynamic(dim)) return true;
  return false;
}

std::optional<int64_t> getConstInt(Value value)
{
  if (!value) return std::nullopt;
  if (auto cst = value.getDefiningOp<arith::ConstantOp>()) {
    if (auto attr = dyn_cast<IntegerAttr>(cst.getValue())) return attr.getInt();
  }
  if (auto cst = value.getDefiningOp<arith::ConstantIndexOp>()) return cst.value();
  if (auto cst = value.getDefiningOp<arith::ConstantIntOp>()) return cst.value();
  return std::nullopt;
}

bool isScalarIntegerLike(Type type)
{
  return type.isIndex() || isa<IntegerType>(type);
}

bool dependsOnPhysicalId(Value value, DenseSet<Value>& seen)
{
  if (!value) return false;
  if (!seen.insert(value).second) return false;
  if (value.getDefiningOp<pumem::GetPhysicalIdOp>()) return true;
  Operation* def = value.getDefiningOp();
  if (!def) return false;
  for (Value operand : def->getOperands())
    if (isScalarIntegerLike(operand.getType()) && dependsOnPhysicalId(operand, seen)) return true;
  return false;
}

bool dependsOnPhysicalId(Value value)
{
  DenseSet<Value> seen;
  return dependsOnPhysicalId(value, seen);
}

bool isPidLoop(scf::ForOp loop)
{
  return dependsOnPhysicalId(loop.getLowerBound()) || dependsOnPhysicalId(loop.getUpperBound()) ||
         dependsOnPhysicalId(loop.getStep());
}

bool hasDynamicViewComponent(Value value, DenseSet<Value>& seen)
{
  if (!value || !seen.insert(value).second) return false;
  Operation* def = value.getDefiningOp();
  if (!def) return false;

  auto hasDynamic = [](ArrayRef<OpFoldResult> values) {
    for (OpFoldResult value : values) {
      auto operand = dyn_cast<Value>(value);
      if (operand && !getConstInt(operand)) return true;
    }
    return false;
  };

  if (auto sized = dyn_cast<OffsetSizeAndStrideOpInterface>(def)) {
    if (hasDynamic(sized.getMixedOffsets()) || hasDynamic(sized.getMixedSizes()) || hasDynamic(sized.getMixedStrides()))
      return true;
  }
  if (auto view = dyn_cast<ViewLikeOpInterface>(def)) {
    Value source = view.getViewSource();
    if (source && isa<MemRefType>(source.getType())) return hasDynamicViewComponent(source, seen);
  }
  return false;
}

bool hasDynamicViewComponent(Value value)
{
  DenseSet<Value> seen;
  return hasDynamicViewComponent(value, seen);
}

bool hasDynamicWindow(Value value)
{
  if (!value) return false;
  if (hasDynamicDim(value.getType())) return true;
  return hasDynamicViewComponent(value);
}

SmallVector<const PathInfo*> pathNodes(const FunctionInfo& function, unsigned pathId)
{
  SmallVector<const PathInfo*> reversed;
  for (unsigned current = pathId; current != 0;) {
    const PathInfo& node = function.paths[current - 1];
    reversed.push_back(&node);
    current = node.parentPathId;
  }
  return SmallVector<const PathInfo*>(reversed.rbegin(), reversed.rend());
}

SmallVector<unsigned> pathLoopIds(const FunctionInfo& function, unsigned pathId)
{
  SmallVector<unsigned> loops;
  for (const PathInfo* node : pathNodes(function, pathId))
    if (node->kind == PathKind::Loop) loops.push_back(node->loopId);
  return loops;
}

SmallVector<int64_t> pathPositions(const FunctionInfo& function, unsigned pathId)
{
  SmallVector<int64_t> positions;
  for (const PathInfo* node : pathNodes(function, pathId)) positions.push_back(node->blockPosInParent);
  return positions;
}

int pathKindRank(PathKind kind)
{
  switch (kind) {
    case PathKind::IfThen: return 0;
    case PathKind::IfElse: return 1;
    case PathKind::Loop: return 2;
  }
  return 3;
}

} // namespace

struct DataflowAnalyzer::Impl
{
  FunctionOpInterface func;
  FunctionInfo function;
  DenseMap<Operation*, unsigned> loopIndices;
  DenseMap<Operation*, Site> sites;
  DenseMap<Operation*, int64_t> opOrdinals;
  DenseMap<Value, unsigned> argPlaces;
  DenseMap<Operation*, unsigned> allocPlaces;
  DenseMap<BlockArgument, int64_t> carrierArgs;
  DenseMap<Value, RootRef> rootCache;
  DenseSet<Value> unresolvedValues;
  DenseMap<Value, unsigned> ivLoopIds;
  SymbolBounds loopBounds;
  int64_t nextOrdinal  = 0;
  unsigned nextLoopSym = 0;
  unsigned nextIfId    = 0;

  struct CarrierEndpoint
  {
    unsigned place;
    int64_t carrierId;
    Branch branch;
    Guard guard;
  };

  static constexpr unsigned kPhysicalIdSym = std::numeric_limits<unsigned>::max();

  explicit Impl(FunctionOpInterface func) : func(func)
  {
    function.op = func;
    Interval physicalId;
    physicalId.hasLo                   = true;
    physicalId.lo                      = 0;
    loopBounds.loopIvs[kPhysicalIdSym] = physicalId;
  }

  LogicalResult emitFailure(Operation* op, const Twine& message)
  {
    emitError(op ? op->getLoc() : func->getLoc()) << message;
    return failure();
  }

  static PhaseResult defer(DataflowDeferredReason reason) { return PhaseResult(reason); }

  bool isReadAccess(const Access& access) const { return isReadRole(access.role); }
  bool isWriteAccess(const Access& access) const { return isWriteRole(access.role); }

  unsigned addPath(
    FunctionInfo& function, Operation* op, PathKind kind, unsigned parentPathId, int64_t blockPosInParent,
    unsigned loopId = 0, unsigned ifId = 0
  )
  {
    PathInfo info;
    info.op               = op;
    info.kind             = kind;
    info.id               = function.paths.size();
    info.parentPathId     = parentPathId;
    info.blockPosInParent = blockPosInParent;
    info.loopId           = loopId;
    info.ifId             = ifId;
    function.paths.push_back(info);
    return info.id + 1;
  }

  PhaseResult walkBlock(FunctionInfo& function, Block* block, int64_t parentId, unsigned pathId)
  {
    int64_t position = 0;
    for (Operation& op : *block) {
      sites[&op]      = Site{pathId, position};
      opOrdinals[&op] = nextOrdinal++;

      if (auto loop = dyn_cast<scf::ForOp>(&op)) {
        unsigned id           = function.loops.size();
        LoopInfo& info        = function.loops.emplace_back();
        info.op               = loop;
        info.parentId         = parentId;
        info.parentPathId     = pathId;
        info.id               = id;
        info.pathId           = addPath(function, loop, PathKind::Loop, pathId, position, id);
        info.blockPosInParent = position;
        info.isPid            = isPidLoop(loop);

        ExprPtr lowerExpr = buildScalar(loop.getLowerBound());
        ExprPtr upperExpr = buildScalar(loop.getUpperBound());
        ExprPtr stepExpr  = buildScalar(loop.getStep());
        unsigned symId    = nextLoopSym++;
        if (lowerExpr && upperExpr && stepExpr) {
          Interval lower = evaluateInterval(*lowerExpr, loopBounds);
          Interval upper = evaluateInterval(*upperExpr, loopBounds);
          Interval step  = evaluateInterval(*stepExpr, loopBounds);
          if (lower.hasLo && upper.hasHi && step.hasLo && step.lo > 0) {
            int64_t distance = 0, adjusted = 0;
            if (upper.hi <= lower.lo) {
              info.trip = 0;
            } else if (!llvm::SubOverflow(upper.hi, lower.lo, distance) &&
                       !llvm::AddOverflow(distance, step.lo - 1, adjusted)) {
              info.trip = adjusted / step.lo;
            }
          }
          if (lower.hasLo && upper.hasHi && upper.hi != std::numeric_limits<int64_t>::min()) {
            int64_t last = upper.hi - 1;
            if (last >= lower.lo) loopBounds.loopIvs[symId] = Interval{true, true, lower.lo, last};
          }
        }
        ivLoopIds[loop.getInductionVar()] = symId;

        loopIndices[&op]   = id;
        PhaseResult nested = walkBlock(function, loop.getBody(), id, info.pathId);
        if (isFailureOrDeferred(nested)) return nested;
      }
      if (auto ifOp = dyn_cast<scf::IfOp>(&op)) {
        unsigned ifId = nextIfId++;
        if (!ifOp.getThenRegion().empty()) {
          unsigned thenPathId = addPath(function, ifOp, PathKind::IfThen, pathId, position, 0, ifId);
          PhaseResult nested  = walkBlock(function, &ifOp.getThenRegion().front(), parentId, thenPathId);
          if (isFailureOrDeferred(nested)) return nested;
        }
        if (!ifOp.getElseRegion().empty()) {
          unsigned elsePathId = addPath(function, ifOp, PathKind::IfElse, pathId, position, 0, ifId);
          PhaseResult nested  = walkBlock(function, &ifOp.getElseRegion().front(), parentId, elsePathId);
          if (isFailureOrDeferred(nested)) return nested;
        }
      }
      ++position;
    }
    return DataflowDeferredReason::None;
  }

  ExprPtr buildScalar(Value value)
  {
    if (!value) return nullptr;
    if (auto cst = getConstInt(value)) return ScalarExpr::cst(*cst);
    if (value.getDefiningOp<pumem::GetPhysicalIdOp>()) return ScalarExpr::sym(SymKind::LoopIv, kPhysicalIdSym);
    if (auto pid = value.getDefiningOp<GetProgramIdOp>()) {
      int axis = pid.getAxisAsInt();
      if (axis < 0 || axis >= kNumAxes) return nullptr;
      return ScalarExpr::sym(static_cast<SymKind>(static_cast<int>(SymKind::PidX) + axis));
    }
    if (auto arg = dyn_cast<BlockArgument>(value)) {
      auto loopIt = ivLoopIds.find(arg);
      if (loopIt != ivLoopIds.end()) return ScalarExpr::sym(SymKind::LoopIv, loopIt->second);
      return nullptr;
    }
    Operation* def = value.getDefiningOp();
    if (!def) return nullptr;
    if (isa<arith::IndexCastOp, arith::ExtSIOp, arith::ExtUIOp, arith::TruncIOp>(def))
      return buildScalar(def->getOperand(0));
    auto kind = scalarExprKind(def);
    if (!kind || def->getNumOperands() != 2) return nullptr;
    ExprPtr lhs = buildScalar(def->getOperand(0));
    ExprPtr rhs = buildScalar(def->getOperand(1));
    if (!lhs || !rhs) return nullptr;
    return ScalarExpr::bin(*kind, lhs, rhs);
  }

  PhaseResult buildLoopTree()
  {
    unsigned loopCount = 0;
    func->walk([&](scf::ForOp) { ++loopCount; });
    function.loops.reserve(loopCount);
    for (Block& block : func.getFunctionBody()) {
      PhaseResult nested = walkBlock(function, &block, -1, 0);
      if (isFailureOrDeferred(nested)) return nested;
    }

    unsigned pidCount = 0;
    for (const LoopInfo& loop : function.loops)
      if (loop.isPid) ++pidCount;
    if (pidCount > 1) return defer(DataflowDeferredReason::MultiplePidLoops);
    return DataflowDeferredReason::None;
  }

  PhaseResult buildPlaces()
  {
    for (BlockArgument arg : func.getArguments()) {
      auto memref = dyn_cast<MemRefType>(arg.getType());
      if (!memref) continue;
      Space space = spaceOf(memref);
      if (space == Space::Unknown) return defer(DataflowDeferredReason::UnsupportedMemorySpace);

      Place place;
      place.id         = function.places.size();
      place.kind       = PlaceKind::KernelArg;
      place.space      = space;
      place.anchor     = arg;
      place.defPathId  = 0;
      place.allocDepth = 0;
      argPlaces[arg]   = place.id;
      function.places.push_back(std::move(place));
    }

    PhaseResult outcome  = DataflowDeferredReason::None;
    WalkResult allocWalk = func->walk([&](Operation* op) -> WalkResult {
      if (!isa<memref::AllocOp, memref::AllocaOp>(op)) return WalkResult::advance();
      auto siteIt = sites.find(op);
      if (siteIt == sites.end()) return WalkResult::advance();
      Value result = op->getResult(0);
      auto memref  = cast<MemRefType>(result.getType());
      Space space  = spaceOf(memref);
      if (space == Space::Unknown) {
        outcome = defer(DataflowDeferredReason::UnsupportedMemorySpace);
        return WalkResult::interrupt();
      }
      const Site& site = siteIt->second;
      Place place;
      place.id         = function.places.size();
      place.kind       = PlaceKind::SramAlloc;
      place.space      = space;
      place.anchor     = result;
      place.defPathId  = site.pathId;
      place.allocDepth = pathLoopIds(function, site.pathId).size();
      place.allocOrder = site.blockOrder;
      if (space == Space::Sram && !pathContainsPid(function, site.pathId)) {
        outcome = defer(DataflowDeferredReason::SramAllocOutsidePid);
        return WalkResult::interrupt();
      }
      allocPlaces[op] = place.id;
      function.places.push_back(std::move(place));
      return WalkResult::advance();
    });
    if (allocWalk.wasInterrupted()) return outcome;

    return DataflowDeferredReason::None;
  }

  PhaseResult buildCarriers()
  {
    for (LoopInfo& loop : function.loops) {
      scf::ForOp forOp      = loop.op;
      Block* body           = forOp.getBody();
      Operation* terminator = body->getTerminator();
      auto yield            = dyn_cast<scf::YieldOp>(terminator);
      if (!yield) return emitFailure(forOp, "loop terminator is not scf.yield");

      ValueRange initArgs = forOp.getInitArgs();
      ValueRange iterArgs = forOp.getRegionIterArgs();
      ValueRange yields   = yield.getOperands();
      if (iterArgs.size() != initArgs.size() || yields.size() != initArgs.size())
        return emitFailure(forOp, "loop carrier arity mismatch");

      for (unsigned index = 0; index < initArgs.size(); ++index) {
        Value init            = initArgs[index];
        BlockArgument iterArg = cast<BlockArgument>(iterArgs[index]);
        Value result          = yields[index];
        if (!isa<MemRefType>(init.getType())) continue;
        if (loop.isPid) return defer(DataflowDeferredReason::PidCarriesMemref);
        RootResolution initRoot = resolveRoot(function, init);
        if (initRoot.deferred != DataflowDeferredReason::None) return initRoot.deferred;
        if (!initRoot.root) continue;

        Carrier carrier;
        carrier.arg    = iterArg;
        carrier.loopId = loop.id;
        carrier.id     = function.carriers.size();

        Place carrierPlace;
        carrierPlace.id         = function.places.size();
        carrierPlace.kind       = PlaceKind::LoopCarried;
        carrierPlace.space      = Space::Unknown;
        carrierPlace.anchor     = iterArg;
        carrierPlace.defPathId  = loop.parentPathId;
        carrierPlace.allocDepth = pathLoopIds(function, carrierPlace.defPathId).size();
        carrierPlace.allocOrder = loop.blockPosInParent;
        carrier.placeId         = carrierPlace.id;
        function.places.push_back(std::move(carrierPlace));

        carrierArgs[iterArg] = carrier.id;
        function.carriers.push_back(std::move(carrier));

        RootResolution yieldRoot = resolveRoot(function, result);
        if (yieldRoot.deferred != DataflowDeferredReason::None) return yieldRoot.deferred;
        if (!yieldRoot.root) {
          carrierArgs.erase(iterArg);
          rootCache.erase(iterArg);
          function.places.pop_back();
          function.carriers.pop_back();
          continue;
        }

        Carrier& added  = function.carriers.back();
        added.initPlace = concretePlace(function, *initRoot.root);
        bool selfYield  = yieldRoot.root->isCarrier() && yieldRoot.root->carrierId == added.id;
        bool nested     = initRoot.root->isCarrier() || (yieldRoot.root->isCarrier() && !selfYield);
        if (nested) return defer(DataflowDeferredReason::NestedCarrier);

        added.yieldPlace = selfYield ? added.initPlace : concretePlace(function, *yieldRoot.root);
        added.rotation   = added.initPlace != added.yieldPlace;
        added.inPlace    = !added.rotation;

        function.places[added.initPlace].carried = true;
        function.places[added.initPlace].carries.push_back(added.id);
        if (added.rotation) {
          function.places[added.yieldPlace].carries.push_back(added.id);
          function.places[added.yieldPlace].carried = true;
        }
      }
    }
    return DataflowDeferredReason::None;
  }

  unsigned concretePlace(const FunctionInfo& function, const RootRef& root) const
  {
    if (!root.isCarrier()) return root.place;
    return function.carriers[root.carrierId].initPlace;
  }

  SmallVector<RootRef> expandRoot(const FunctionInfo& function, const RootRef& root) const
  {
    if (!root.isCarrier()) return {RootRef{root.place, -1}};
    const Carrier& carrier = function.carriers[root.carrierId];
    return {RootRef{carrier.initPlace, carrier.id}, RootRef{carrier.yieldPlace, carrier.id}};
  }

  SmallVector<CarrierEndpoint> expandEndpoints(const FunctionInfo& function, const RootRef& root) const
  {
    if (!root.isCarrier()) return {CarrierEndpoint{root.place, -1, Branch::None, Guard::None}};
    const Carrier& carrier = function.carriers[root.carrierId];
    return {
      CarrierEndpoint{carrier.initPlace, carrier.id, Branch::Init, Guard::FirstIter},
      CarrierEndpoint{carrier.yieldPlace, carrier.id, Branch::Yield, Guard::BodyIter}
    };
  }

  RootResolution resolveRoot(FunctionInfo& function, Value value)
  {
    if (!value) return {};

    auto cached = rootCache.find(value);
    if (cached != rootCache.end()) return {std::optional<RootRef>(cached->second), DataflowDeferredReason::None};
    if (unresolvedValues.contains(value)) return {std::nullopt, DataflowDeferredReason::UnresolvedMemref};

    RootRef root{0, -1};
    bool resolved = false;

    auto resolveView = [&](Value source) -> RootResolution {
      RootResolution nested = resolveRoot(function, source);
      if (nested.deferred != DataflowDeferredReason::None) return nested;
      if (nested.root) {
        root     = *nested.root;
        resolved = true;
      }
      return {};
    };

    if (auto arg = dyn_cast<BlockArgument>(value)) {
      auto argIt = argPlaces.find(arg);
      if (argIt != argPlaces.end()) {
        root     = RootRef{argIt->second, -1};
        resolved = true;
      } else {
        auto carrierIt = carrierArgs.find(arg);
        if (carrierIt != carrierArgs.end()) {
          root     = RootRef{function.carriers[carrierIt->second].placeId, carrierIt->second};
          resolved = true;
        }
      }
    } else if (Operation* def = value.getDefiningOp()) {
      if (isa<memref::AllocOp, memref::AllocaOp>(def)) {
        auto allocIt = allocPlaces.find(def);
        if (allocIt != allocPlaces.end()) {
          root     = RootRef{allocIt->second, -1};
          resolved = true;
        }
      } else if (dyn_cast<scf::ForOp>(def)) {
        auto loopIt = loopIndices.find(def);
        if (loopIt != loopIndices.end()) {
          unsigned resultIndex = cast<OpResult>(value).getResultNumber();
          const LoopInfo& loop = function.loops[loopIt->second];
          scf::ForOp loopOp    = loop.op;
          Value iterArg        = loopOp.getRegionIterArgs()[resultIndex];
          for (const Carrier& carrier : function.carriers) {
            if (carrier.loopId != loop.id || carrier.arg != iterArg) continue;
            if (loop.trip < 0) return {std::nullopt, DataflowDeferredReason::SymbolicTrip};
            unsigned place = loop.trip == 0 ? carrier.initPlace : carrier.yieldPlace;
            root           = RootRef{place, -1};
            resolved       = true;
            break;
          }
        }
      } else if (auto expand = dyn_cast<memref::ExpandShapeOp>(def)) {
        RootResolution view = resolveView(expand.getSrc());
        if (view.deferred != DataflowDeferredReason::None) return view;
      } else if (auto collapse = dyn_cast<memref::CollapseShapeOp>(def)) {
        RootResolution view = resolveView(collapse.getSrc());
        if (view.deferred != DataflowDeferredReason::None) return view;
      } else if (auto transpose = dyn_cast<memref::TransposeOp>(def)) {
        RootResolution view = resolveView(transpose.getIn());
        if (view.deferred != DataflowDeferredReason::None) return view;
      } else if (auto reshape = dyn_cast<memref::ReshapeOp>(def)) {
        RootResolution view = resolveView(reshape.getSource());
        if (view.deferred != DataflowDeferredReason::None) return view;
      } else if (auto view = dyn_cast<ViewLikeOpInterface>(def)) {
        Value source = view.getViewSource();
        if (source && isa<MemRefType>(source.getType())) {
          RootResolution resolution = resolveView(source);
          if (resolution.deferred != DataflowDeferredReason::None) return resolution;
        }
      }
    }

    if (!resolved) {
      if (unresolvedValues.insert(value).second) ++function.unresolved;
      return {std::nullopt, DataflowDeferredReason::UnresolvedMemref};
    }

    rootCache[value] = root;
    return {std::optional<RootRef>(root), DataflowDeferredReason::None};
  }

  ProgramPoint
  pointFor(const FunctionInfo& function, unsigned pathId, int64_t blockOrder, int64_t delta, unsigned shiftLoopId) const
  {
    ProgramPoint point;
    point.pathId   = pathId;
    point.blockPos = pathPositions(function, pathId);
    point.blockPos.push_back(blockOrder);
    point.delta       = delta;
    point.shiftLoopId = shiftLoopId;
    return point;
  }

  ProgramPoint loopExit(const FunctionInfo& function, unsigned loopId) const
  {
    const LoopInfo& loop = function.loops[loopId];
    ProgramPoint point;
    point.pathId   = loop.parentPathId;
    point.blockPos = pathPositions(function, loop.parentPathId);
    point.blockPos.push_back(loop.blockPosInParent);
    point.blockPos.push_back(kInfinity);
    point.isLoopExit = true;
    point.exitLoopId = loopId;
    point.delta      = 0;
    return point;
  }

  int comparePoints(const FunctionInfo& function, const ProgramPoint& lhs, const ProgramPoint& rhs) const
  {
    SmallVector<unsigned> lhsLoops = pathLoopIds(function, lhs.pathId);
    SmallVector<unsigned> rhsLoops = pathLoopIds(function, rhs.pathId);
    size_t common                  = std::min(lhsLoops.size(), rhsLoops.size());
    for (size_t index = 0; index < common; ++index) {
      if (lhsLoops[index] != rhsLoops[index]) break;
      int64_t lhsDelta = lhs.shiftLoopId == lhsLoops[index] ? lhs.delta : 0;
      int64_t rhsDelta = rhs.shiftLoopId == rhsLoops[index] ? rhs.delta : 0;
      if (lhsDelta != rhsDelta) return lhsDelta < rhsDelta ? -1 : 1;
    }

    SmallVector<const PathInfo*> lhsNodes = pathNodes(function, lhs.pathId);
    SmallVector<const PathInfo*> rhsNodes = pathNodes(function, rhs.pathId);
    size_t commonNodes                    = 0;
    while (commonNodes < lhsNodes.size() && commonNodes < rhsNodes.size()) {
      if (lhsNodes[commonNodes] != rhsNodes[commonNodes]) break;
      ++commonNodes;
    }

    if (commonNodes < lhsNodes.size() && commonNodes < rhsNodes.size()) {
      const PathInfo& lhsNode = *lhsNodes[commonNodes];
      const PathInfo& rhsNode = *rhsNodes[commonNodes];
      if (lhsNode.kind != PathKind::Loop || rhsNode.kind != PathKind::Loop) {
        if (lhsNode.blockPosInParent != rhsNode.blockPosInParent)
          return lhsNode.blockPosInParent < rhsNode.blockPosInParent ? -1 : 1;
        int lhsRank = pathKindRank(lhsNode.kind);
        int rhsRank = pathKindRank(rhsNode.kind);
        if (lhsRank != rhsRank) return lhsRank < rhsRank ? -1 : 1;
      }
    }

    size_t lhsSize = lhs.blockPos.size();
    size_t rhsSize = rhs.blockPos.size();
    size_t limit   = std::min(lhsSize, rhsSize);
    for (size_t index = 0; index < limit; ++index)
      if (lhs.blockPos[index] != rhs.blockPos[index]) return lhs.blockPos[index] < rhs.blockPos[index] ? -1 : 1;
    if (lhsSize != rhsSize) return lhsSize < rhsSize ? -1 : 1;
    return 0;
  }

  bool pathContainsPid(const FunctionInfo& function, unsigned pathId) const
  {
    for (unsigned loopId : pathLoopIds(function, pathId))
      if (function.loops[loopId].isPid) return true;
    return false;
  }

  void markCarrierWrite(FunctionInfo& function, const Access& access)
  {
    if (access.carrierId >= 0 && isWriteAccess(access)) function.carriers[access.carrierId].wroteBody = true;
  }

  void markCarrierWrite(FunctionInfo& function, const Transfer& transfer)
  {
    if (transfer.carrierAtDst && transfer.carrierId >= 0) function.carriers[transfer.carrierId].wroteBody = true;
  }

  unsigned shiftLoopFor(const FunctionInfo& function, const Access& access) const
  {
    if (access.delta == 0 || access.carrierId < 0) return 0;
    return function.carriers[access.carrierId].loopId;
  }

  unsigned shiftLoopFor(const FunctionInfo& function, const Transfer& transfer) const
  {
    if (transfer.delta == 0 || transfer.carrierId < 0) return 0;
    return function.carriers[transfer.carrierId].loopId;
  }

  PhaseResult addAccess(FunctionInfo& function, Operation* op, Value value, Role role, unsigned operandIndex)
  {
    if (!value || !isa<MemRefType>(value.getType())) return DataflowDeferredReason::None;
    auto siteIt = sites.find(op);
    if (siteIt == sites.end()) return DataflowDeferredReason::None;
    RootResolution root = resolveRoot(function, value);
    if (root.deferred != DataflowDeferredReason::None) return root.deferred;
    if (!root.root) return DataflowDeferredReason::None;
    RootRef rootRef  = *root.root;
    const Site& site = siteIt->second;

    SmallVector<CarrierEndpoint> expanded = expandEndpoints(function, rootRef);

    for (const CarrierEndpoint& item : expanded) {
      SmallVector<Role> roles;
      if (item.carrierId >= 0 && role == Role::Atomic)
        roles = {Role::Read, Role::Write};
      else
        roles.push_back(role);
      for (Role itemRole : roles) {
        Access access;
        access.op           = op;
        access.place        = item.place;
        access.role         = itemRole;
        access.pathId       = site.pathId;
        access.blockOrder   = site.blockOrder;
        access.delta        = (item.branch == Branch::Yield && isReadRole(itemRole)) ? 1 : 0;
        access.carrierId    = item.carrierId;
        access.branch       = item.branch;
        access.guard        = item.guard;
        access.operandIndex = operandIndex;

        unsigned id = function.accesses.size();
        function.accesses.push_back(access);
        if (isWriteAccess(access))
          function.places[access.place].producers.push_back(Place::EventRef{Place::EventRef::AccessK, id});
        if (isReadAccess(access))
          function.places[access.place].consumers.push_back(Place::EventRef{Place::EventRef::AccessK, id});
        markCarrierWrite(function, access);
      }
    }
    return DataflowDeferredReason::None;
  }

  PhaseResult addTransfer(
    FunctionInfo& function, Operation* op, Value source, Value target, OpKind kind, bool masked, bool dynamic,
    bool indirect, ArrayRef<Value> indirectValues
  )
  {
    auto siteIt = sites.find(op);
    if (siteIt == sites.end()) return DataflowDeferredReason::None;

    RootResolution sourceRoot = resolveRoot(function, source);
    if (sourceRoot.deferred != DataflowDeferredReason::None) return sourceRoot.deferred;

    RootResolution targetRoot = resolveRoot(function, target);
    if (targetRoot.deferred != DataflowDeferredReason::None) return targetRoot.deferred;
    if (!sourceRoot.root || !targetRoot.root) return DataflowDeferredReason::None;

    RootRef sourceRef = *sourceRoot.root;
    RootRef targetRef = *targetRoot.root;
    if (sourceRef.isCarrier() && targetRef.isCarrier()) return defer(DataflowDeferredReason::TransferBetweenCarriers);

    SmallVector<CarrierEndpoint> sources = expandEndpoints(function, sourceRef);
    SmallVector<CarrierEndpoint> targets = expandEndpoints(function, targetRef);

    const Site& site = siteIt->second;
    for (const CarrierEndpoint& src : sources) {
      for (const CarrierEndpoint& dst : targets) {
        Space srcSpace = function.places[src.place].space;
        Space dstSpace = function.places[dst.place].space;
        if (srcSpace == Space::Unknown || dstSpace == Space::Unknown)
          return emitFailure(op, "transfer endpoint has unknown memory space");
        Dir dir;
        if (srcSpace == Space::Dram && dstSpace == Space::Sram) {
          dir = Dir::DramToSram;
        } else if (srcSpace == Space::Sram && dstSpace == Space::Dram) {
          dir = Dir::SramToDram;
        } else if (srcSpace == Space::Sram && dstSpace == Space::Sram) {
          dir = Dir::SramToSram;
        } else {
          return defer(DataflowDeferredReason::DramToDramTransfer);
        }

        Transfer transfer;
        transfer.op           = op;
        transfer.src          = src.place;
        transfer.dst          = dst.place;
        transfer.dir          = dir;
        transfer.opKind       = kind;
        transfer.pathId       = site.pathId;
        transfer.blockOrder   = site.blockOrder;
        transfer.masked       = masked;
        transfer.dynamic      = dynamic;
        transfer.indirect     = indirect;
        transfer.delta        = (src.branch == Branch::Yield) ? 1 : 0;
        transfer.carrierId    = src.carrierId >= 0 ? src.carrierId : dst.carrierId;
        transfer.carrierAtDst = src.carrierId < 0 && dst.carrierId >= 0;
        transfer.branch       = src.carrierId >= 0 ? src.branch : dst.branch;
        transfer.guard        = src.carrierId >= 0 ? src.guard : dst.guard;

        unsigned id = function.transfers.size();
        function.transfers.push_back(transfer);
        function.places[transfer.dst].producers.push_back(Place::EventRef{Place::EventRef::TransferK, id});
        function.places[transfer.src].consumers.push_back(Place::EventRef{Place::EventRef::TransferK, id});
        if (indirect) {
          function.places[transfer.src].indirect = true;
          function.places[transfer.dst].indirect = true;
        }
        markCarrierWrite(function, transfer);
      }
    }

    for (Value value : indirectValues) {
      if (!value || !isa<MemRefType>(value.getType())) continue;
      RootResolution root = resolveRoot(function, value);
      if (root.deferred != DataflowDeferredReason::None) return root.deferred;
      if (!root.root) continue;
      if (indirect) {
        for (const RootRef& endpoint : expandRoot(function, *root.root))
          function.places[endpoint.place].indirect = true;
      }
      unsigned operandIndex = 0;
      for (Value operand : op->getOperands()) {
        if (operand == value) break;
        ++operandIndex;
      }
      PhaseResult access = addAccess(function, op, value, Role::Read, operandIndex);
      if (isFailureOrDeferred(access)) return access;
    }
    return DataflowDeferredReason::None;
  }

  PhaseResult handleTransferOp(Operation* op)
  {
    if (auto copy = dyn_cast<memref::CopyOp>(op)) {
      bool dynamic = hasDynamicWindow(copy.getSource()) || hasDynamicWindow(copy.getTarget());
      return addTransfer(function, op, copy.getSource(), copy.getTarget(), OpKind::Copy, false, dynamic, false, {});
    }
    if (auto gather = dyn_cast<pumem::TileGatherOp>(op)) {
      bool masked  = static_cast<bool>(gather.getMask());
      bool dynamic = hasDynamicDim(gather.getDestination().getType());
      SmallVector<Value> extras{gather.getIndices()};
      if (gather.getMask()) extras.push_back(gather.getMask());
      return addTransfer(
        function, op, gather.getSource(), gather.getDestination(), OpKind::Gather, masked, dynamic, true, extras
      );
    }
    if (auto scatter = dyn_cast<pumem::TileScatterOp>(op)) {
      bool masked  = static_cast<bool>(scatter.getMask());
      bool dynamic = hasDynamicDim(scatter.getSource().getType());
      SmallVector<Value> extras{scatter.getIndices()};
      if (scatter.getMask()) extras.push_back(scatter.getMask());
      return addTransfer(
        function, op, scatter.getSource(), scatter.getDestination(), OpKind::Scatter, masked, dynamic, true, extras
      );
    }
    if (isa<memref::DeallocOp>(op)) return defer(DataflowDeferredReason::MemrefDealloc);
    return DataflowDeferredReason::None;
  }

  PhaseResult collectTransfers()
  {
    PhaseResult outcome   = DataflowDeferredReason::None;
    WalkResult walkResult = func->walk([&](Operation* op) -> WalkResult {
      PhaseResult result = handleTransferOp(op);
      if (isFailureOrDeferred(result)) {
        outcome = result;
        return WalkResult::interrupt();
      }
      return WalkResult::advance();
    });
    return walkResult.wasInterrupted() ? outcome : PhaseResult(DataflowDeferredReason::None);
  }

  PhaseResult collectAccesses()
  {
    PhaseResult outcome = DataflowDeferredReason::None;
    auto recordAccess   = [&](Operation* op, Value value, Role role, unsigned operandIndex) {
      PhaseResult result = addAccess(function, op, value, role, operandIndex);
      if (!isFailureOrDeferred(result)) return true;
      outcome = result;
      return false;
    };

    auto recordMemrefEscape = [&](Type type) {
      if (isa<MemRefType>(type)) {
        outcome = defer(DataflowDeferredReason::MemrefEscape);
        return true;
      }
      return false;
    };

    WalkResult walkResult = func->walk([&](Operation* op) -> WalkResult {
      if (auto linalgOp = dyn_cast<linalg::LinalgOp>(op)) {
        unsigned index = 0;
        for (Value input : linalgOp.getDpsInputs()) {
          if (isa<MemRefType>(input.getType()) && !recordAccess(op, input, Role::Read, index))
            return WalkResult::interrupt();
          ++index;
        }
        for (Value output : linalgOp.getDpsInits()) {
          if (isa<MemRefType>(output.getType()) && !recordAccess(op, output, Role::Write, index))
            return WalkResult::interrupt();
          ++index;
        }
        return WalkResult::advance();
      }
      if (auto load = dyn_cast<memref::LoadOp>(op)) {
        if (recordMemrefEscape(load.getType())) return WalkResult::interrupt();
        if (!recordAccess(op, load.getMemRef(), Role::ScalarRead, 0)) return WalkResult::interrupt();
        return WalkResult::advance();
      }
      if (auto store = dyn_cast<memref::StoreOp>(op)) {
        if (recordMemrefEscape(store.getValueToStore().getType())) return WalkResult::interrupt();
        if (!recordAccess(op, store.getMemRef(), Role::ScalarWrite, 0)) return WalkResult::interrupt();
        return WalkResult::advance();
      }
      if (isa<memref::AtomicRMWOp>(op)) {
        if (op->getNumOperands() > 0 && !recordAccess(op, op->getOperand(0), Role::Atomic, 0))
          return WalkResult::interrupt();
        return WalkResult::advance();
      }
      return WalkResult::advance();
    });
    return walkResult.wasInterrupted() ? outcome : PhaseResult(DataflowDeferredReason::None);
  }

  PhaseResult groupFills()
  {
    std::map<std::pair<unsigned, unsigned>, std::pair<SmallVector<unsigned>, SmallVector<unsigned>>> groups;

    for (size_t index = 0; index < function.accesses.size(); ++index) {
      const Access& access = function.accesses[index];
      if (access.role != Role::Write || !isa<linalg::FillOp>(access.op)) continue;
      groups[{access.place, access.pathId}].first.push_back(index);
    }
    for (size_t index = 0; index < function.transfers.size(); ++index) {
      const Transfer& transfer = function.transfers[index];
      if (transfer.opKind != OpKind::Copy) continue;
      groups[{transfer.dst, transfer.pathId}].second.push_back(index);
    }

    int64_t nextGroup = 0;
    for (auto& [_, group] : groups) {
      if (group.first.empty() || group.second.empty()) continue;
      for (unsigned accessId : group.first) function.accesses[accessId].groupId = nextGroup;
      for (unsigned transferId : group.second) function.transfers[transferId].groupId = nextGroup;
      ++nextGroup;
    }
    return DataflowDeferredReason::None;
  }

  PhaseResult finalizePlaceAccesses()
  {
    for (Place& place : function.places) place.accesses.clear();
    for (const Access& access : function.accesses)
      if (access.place < function.places.size()) function.places[access.place].accesses.push_back(access);

    auto sortPoint = [&](const Access& access) {
      int64_t delta  = access.branch == Branch::Yield ? 1 : 0;
      unsigned shift = delta != 0 && access.carrierId >= 0 ? function.carriers[access.carrierId].loopId : 0;
      return pointFor(function, access.pathId, access.blockOrder, delta, shift);
    };

    for (Place& place : function.places) {
      llvm::sort(place.accesses, [&](const Access& lhs, const Access& rhs) {
        int order = comparePoints(function, sortPoint(lhs), sortPoint(rhs));
        if (order != 0) return order < 0;
        int lhsRank = roleRank(lhs.role);
        int rhsRank = roleRank(rhs.role);
        if (lhsRank != rhsRank) return lhsRank < rhsRank;
        if (lhs.operandIndex != rhs.operandIndex) return lhs.operandIndex < rhs.operandIndex;
        return opOrdinals.lookup(lhs.op) < opOrdinals.lookup(rhs.op);
      });
    }
    return DataflowDeferredReason::None;
  }

  ProgramPoint eventPoint(const FunctionInfo& function, const Place::EventRef& ref, bool producer) const
  {
    if (ref.kind == Place::EventRef::TransferK) {
      const Transfer& transfer = function.transfers[ref.id];
      int64_t delta            = producer ? 0 : transfer.delta;
      unsigned shift           = producer ? 0 : shiftLoopFor(function, transfer);
      return pointFor(function, transfer.pathId, transfer.blockOrder, delta, shift);
    }
    const Access& access = function.accesses[ref.id];
    int64_t delta        = producer ? 0 : access.delta;
    unsigned shift       = producer ? 0 : shiftLoopFor(function, access);
    return pointFor(function, access.pathId, access.blockOrder, delta, shift);
  }

  PhaseResult computeLifetimes()
  {
    for (Place& place : function.places) {
      if (place.kind != PlaceKind::SramAlloc) continue;

      SmallVector<ProgramPoint> producerPoints;
      SmallVector<ProgramPoint> consumerPoints;
      for (const Place::EventRef& ref : place.producers) producerPoints.push_back(eventPoint(function, ref, true));
      for (const Place::EventRef& ref : place.consumers) consumerPoints.push_back(eventPoint(function, ref, false));

      ProgramPoint allocPoint = pointFor(function, place.defPathId, place.allocOrder, 0, 0);
      if (producerPoints.empty()) {
        place.start = allocPoint;
        return defer(DataflowDeferredReason::StationWithoutProducer);
      }

      SmallVector<ProgramPoint> startCandidates = producerPoints;
      llvm::append_range(startCandidates, consumerPoints);
      place.start = *std::min_element(
        startCandidates.begin(), startCandidates.end(),
        [&](const ProgramPoint& lhs, const ProgramPoint& rhs) { return comparePoints(function, lhs, rhs) < 0; }
      );

      SmallVector<ProgramPoint> retireCandidates = producerPoints;
      llvm::append_range(retireCandidates, consumerPoints);
      for (const Carrier& carrier : function.carriers) {
        if (!carrier.inPlace) continue;
        if (carrier.initPlace != place.id && carrier.yieldPlace != place.id) continue;
        retireCandidates.push_back(loopExit(function, carrier.loopId));
      }

      place.retire = *std::max_element(
        retireCandidates.begin(), retireCandidates.end(),
        [&](const ProgramPoint& lhs, const ProgramPoint& rhs) { return comparePoints(function, lhs, rhs) < 0; }
      );

      place.dead                = consumerPoints.empty();
      ProgramPoint lastProducer = *std::max_element(
        producerPoints.begin(), producerPoints.end(),
        [&](const ProgramPoint& lhs, const ProgramPoint& rhs) { return comparePoints(function, lhs, rhs) < 0; }
      );
      bool tailWrite = true;
      if (!consumerPoints.empty()) {
        ProgramPoint lastConsumer = *std::max_element(
          consumerPoints.begin(), consumerPoints.end(),
          [&](const ProgramPoint& lhs, const ProgramPoint& rhs) { return comparePoints(function, lhs, rhs) < 0; }
        );
        tailWrite = comparePoints(function, lastProducer, lastConsumer) >= 0;
      }
      if (tailWrite) ++function.tailWrites;

      bool allCopies = !place.producers.empty() && !place.consumers.empty() && place.accesses.empty();
      if (allCopies) {
        for (const Place::EventRef& ref : place.producers)
          if (ref.kind != Place::EventRef::TransferK || function.transfers[ref.id].opKind != OpKind::Copy)
            allCopies = false;
        for (const Place::EventRef& ref : place.consumers)
          if (ref.kind != Place::EventRef::TransferK || function.transfers[ref.id].opKind != OpKind::Copy)
            allCopies = false;
      }
      place.through = allCopies;
    }
    return DataflowDeferredReason::None;
  }

  PhaseResult checkEmpirical()
  {
    DenseMap<unsigned, unsigned> loadProducers;
    DenseMap<unsigned, unsigned> storeConsumers;
    for (const Transfer& transfer : function.transfers) {
      auto loops         = pathLoopIds(function, transfer.pathId);
      unsigned pathDepth = loops.size();
      if (transfer.dir == Dir::DramToSram) {
        ++loadProducers[transfer.dst];
        if (pathDepth != function.places[transfer.dst].allocDepth)
          return defer(DataflowDeferredReason::LoadDepthMismatch);
      }
      if (transfer.dir == Dir::SramToDram) {
        ++storeConsumers[transfer.src];
        unsigned pidDepth = 0;
        for (unsigned index = 0; index < loops.size(); ++index)
          if (function.loops[loops[index]].isPid) {
            pidDepth = index + 1;
            break;
          }
        if (pidDepth != 0 && pathDepth != pidDepth) return defer(DataflowDeferredReason::StoreDepthMismatch);
      }
    }

    for (const Place& place : function.places) {
      if (place.kind != PlaceKind::SramAlloc) continue;
      auto loadIt = loadProducers.find(place.id);
      if (loadIt != loadProducers.end() && loadIt->second > 1)
        return defer(DataflowDeferredReason::MultipleLoadProducers);
      auto storeIt = storeConsumers.find(place.id);
      if (storeIt != storeConsumers.end() && storeIt->second > 2)
        return defer(DataflowDeferredReason::MultipleStoreConsumers);
    }

    auto pathContains = [&](unsigned pathId, unsigned loopId) {
      return llvm::is_contained(pathLoopIds(function, pathId), loopId);
    };
    for (const LoopInfo& loop : function.loops) {
      if (loop.trip >= 0) continue;
      bool carriesData = false;
      for (const Place& place : function.places)
        if (place.kind == PlaceKind::SramAlloc && pathContains(place.defPathId, loop.id)) {
          carriesData = true;
          break;
        }
      if (!carriesData)
        for (const Transfer& transfer : function.transfers)
          if (pathContains(transfer.pathId, loop.id)) {
            carriesData = true;
            break;
          }
      if (!carriesData)
        for (const Access& access : function.accesses)
          if (pathContains(access.pathId, loop.id)) {
            carriesData = true;
            break;
          }
      if (carriesData) return defer(DataflowDeferredReason::SymbolicTrip);
    }

    for (const Access& access : function.accesses) {
      if (access.role != Role::ScalarWrite || function.places[access.place].space != Space::Sram) continue;
      return defer(DataflowDeferredReason::ScalarWriteToSram);
    }
    return DataflowDeferredReason::None;
  }

  PhaseResult checkModeledRegions()
  {
    PhaseResult outcome   = DataflowDeferredReason::None;
    WalkResult walkResult = func->walk([&](Operation* op) -> WalkResult {
      if (sites.contains(op)) return WalkResult::advance();
      if (llvm::any_of(op->getOperandTypes(), [](Type type) { return isa<MemRefType>(type); })) {
        outcome = defer(DataflowDeferredReason::UnmodeledRegion);
        return WalkResult::interrupt();
      }
      if (llvm::any_of(op->getResultTypes(), [](Type type) { return isa<MemRefType>(type); })) {
        outcome = defer(DataflowDeferredReason::UnmodeledRegion);
        return WalkResult::interrupt();
      }
      return WalkResult::advance();
    });
    return walkResult.wasInterrupted() ? outcome : PhaseResult(DataflowDeferredReason::None);
  }

  PhaseResult checkEscapes()
  {
    PhaseResult outcome   = DataflowDeferredReason::None;
    WalkResult walkResult = func->walk([&](Operation* op) -> WalkResult {
      if (isa<scf::YieldOp>(op) && isa<scf::ForOp>(op->getParentOp())) {
        return WalkResult::advance();
      }

      if (isa<CallOpInterface>(op)) {
        if (llvm::any_of(op->getOperandTypes(), [](Type type) { return isa<MemRefType>(type); })) {
          outcome = defer(DataflowDeferredReason::MemrefEscape);
          return WalkResult::interrupt();
        }
        if (llvm::any_of(op->getResultTypes(), [](Type type) { return isa<MemRefType>(type); })) {
          outcome = defer(DataflowDeferredReason::MemrefEscape);
          return WalkResult::interrupt();
        }
      }

      if (op->hasTrait<OpTrait::ReturnLike>()) {
        if (llvm::any_of(op->getOperandTypes(), [](Type type) { return isa<MemRefType>(type); })) {
          outcome = defer(DataflowDeferredReason::MemrefEscape);
          return WalkResult::interrupt();
        }
      }
      return WalkResult::advance();
    });
    return walkResult.wasInterrupted() ? outcome : PhaseResult(DataflowDeferredReason::None);
  }

  PhaseResult checkInvariants()
  {
    PhaseResult resolveOutcome = DataflowDeferredReason::None;
    WalkResult resolveWalk     = func->walk([&](Operation* op) -> WalkResult {
      for (Value operand : op->getOperands()) {
        if (!isa<MemRefType>(operand.getType())) continue;
        RootResolution resolution = resolveRoot(function, operand);
        if (resolution.deferred != DataflowDeferredReason::None) {
          resolveOutcome = PhaseResult(resolution.deferred);
          return WalkResult::interrupt();
        }
      }
      return WalkResult::advance();
    });
    if (resolveWalk.wasInterrupted()) return resolveOutcome;

    auto checkSpace = [&](Type type) -> DataflowDeferredReason {
      auto memref = dyn_cast<MemRefType>(type);
      if (!memref) return DataflowDeferredReason::None;
      if (spaceOf(memref) == Space::Unknown) return DataflowDeferredReason::UnsupportedMemorySpace;
      return DataflowDeferredReason::None;
    };
    PhaseResult spaceOutcome = DataflowDeferredReason::None;
    WalkResult spaceWalk     = func->walk([&](Operation* op) -> WalkResult {
      for (Type type : op->getResultTypes()) {
        DataflowDeferredReason reason = checkSpace(type);
        if (reason != DataflowDeferredReason::None) {
          spaceOutcome = PhaseResult(reason);
          return WalkResult::interrupt();
        }
      }
      for (Value operand : op->getOperands()) {
        DataflowDeferredReason reason = checkSpace(operand.getType());
        if (reason != DataflowDeferredReason::None) {
          spaceOutcome = PhaseResult(reason);
          return WalkResult::interrupt();
        }
      }
      return WalkResult::advance();
    });
    if (spaceWalk.wasInterrupted()) return spaceOutcome;
    for (BlockArgument arg : func.getArguments()) {
      DataflowDeferredReason reason = checkSpace(arg.getType());
      if (reason != DataflowDeferredReason::None) return reason;
    }

    for (const Carrier& carrier : function.carriers) {
      Operation* carrierOp = carrier.arg.getOwner()->getParentOp();
      if (carrier.initPlace >= function.places.size() || carrier.yieldPlace >= function.places.size())
        return emitFailure(carrierOp, "carrier references an invalid place");
      if (carrier.loopId >= function.loops.size()) return emitFailure(carrierOp, "carrier references an invalid loop");
      if (carrier.rotation != (carrier.initPlace != carrier.yieldPlace))
        return emitFailure(carrierOp, "carrier rotation flag is inconsistent");
      if (carrier.inPlace != (carrier.initPlace == carrier.yieldPlace))
        return emitFailure(carrierOp, "carrier inPlace flag is inconsistent");
    }

    for (LoopInfo& loop : function.loops) {
      if (!loop.isPid) continue;
      for (Value init : loop.op.getInitArgs())
        if (isa<MemRefType>(init.getType())) return defer(DataflowDeferredReason::PidCarriesMemref);
    }

    for (const Transfer& transfer : function.transfers) {
      Space srcSpace = function.places[transfer.src].space;
      Space dstSpace = function.places[transfer.dst].space;
      bool valid     = false;
      if (transfer.dir == Dir::DramToSram) valid = srcSpace == Space::Dram && dstSpace == Space::Sram;
      if (transfer.dir == Dir::SramToDram) valid = srcSpace == Space::Sram && dstSpace == Space::Dram;
      if (transfer.dir == Dir::SramToSram) valid = srcSpace == Space::Sram && dstSpace == Space::Sram;
      if (!valid) return emitFailure(transfer.op, "transfer direction does not match endpoint spaces");
    }

    auto checkBranchGuard = [&](const auto& event) -> LogicalResult {
      bool valid = (event.branch == Branch::None && event.guard == Guard::None) ||
                   (event.branch == Branch::Init && event.guard == Guard::FirstIter) ||
                   (event.branch == Branch::Yield && event.guard == Guard::BodyIter);
      if (!valid) return emitFailure(event.op, "carrier branch and guard are inconsistent");
      if ((event.carrierId >= 0) != (event.branch != Branch::None))
        return emitFailure(event.op, "carrier id and branch must be set together");
      if (event.carrierId >= static_cast<int64_t>(function.carriers.size()))
        return emitFailure(event.op, "carrier id is out of range");
      return success();
    };
    for (const Access& access : function.accesses) {
      if (failed(checkBranchGuard(access))) return failure();
      if (isWriteAccess(access) && access.delta != 0) return emitFailure(access.op, "write access has non-zero delta");
      if (access.role == Role::Read || access.role == Role::ScalarRead)
        if (access.delta != 0 && access.carrierId < 0)
          return emitFailure(access.op, "non-carrier read has non-zero delta");
    }
    for (const Transfer& transfer : function.transfers)
      if (failed(checkBranchGuard(transfer))) return failure();

    for (const Place& place : function.places) {
      if (place.kind != PlaceKind::SramAlloc) continue;
      Operation* placeOp = place.anchor.getDefiningOp();
      if (comparePoints(function, place.start, place.retire) > 0)
        return emitFailure(placeOp, "station has start after retire");
      for (const Place::EventRef& ref : place.producers) {
        ProgramPoint point = eventPoint(function, ref, true);
        if (comparePoints(function, point, place.start) < 0 || comparePoints(function, point, place.retire) > 0)
          return emitFailure(placeOp, "producer lies outside station lifetime");
      }
      for (const Place::EventRef& ref : place.consumers) {
        ProgramPoint point = eventPoint(function, ref, false);
        if (comparePoints(function, point, place.start) < 0 || comparePoints(function, point, place.retire) > 0)
          return emitFailure(placeOp, "consumer lies outside station lifetime");
      }
    }

    for (const Carrier& carrier : function.carriers) {
      if (!carrier.inPlace) continue;
      const Place& place = function.places[carrier.initPlace];
      if (comparePoints(function, place.retire, loopExit(function, carrier.loopId)) < 0)
        return emitFailure(carrier.arg.getOwner()->getParentOp(), "inPlace station retires before loop exit");
    }

    return DataflowDeferredReason::None;
  }

  FailureOr<AnalysisResult> analyze(DenseI64ArrayAttr grid)
  {
    if (!grid) return emitFailure(nullptr, "missing privateuse.grid module attribute");

    for (int axis = 0; axis < kNumAxes && axis < grid.size(); ++axis) {
      if (grid[axis] <= 0) continue;
      Interval interval;
      interval.hasLo        = true;
      interval.hasHi        = true;
      interval.lo           = 0;
      interval.hi           = grid[axis] - 1;
      loopBounds.pids[axis] = interval;
    }

    PhaseResult status = buildLoopTree();
    if (succeeded(status) && *status == DataflowDeferredReason::None) status = checkModeledRegions();
    if (succeeded(status) && *status == DataflowDeferredReason::None) status = checkEscapes();
    if (succeeded(status) && *status == DataflowDeferredReason::None) status = buildPlaces();
    if (succeeded(status) && *status == DataflowDeferredReason::None) status = buildCarriers();
    if (succeeded(status) && *status == DataflowDeferredReason::None) status = collectTransfers();
    if (succeeded(status) && *status == DataflowDeferredReason::None) status = collectAccesses();
    if (succeeded(status) && *status == DataflowDeferredReason::None) status = groupFills();
    if (succeeded(status) && *status == DataflowDeferredReason::None) status = finalizePlaceAccesses();
    if (succeeded(status) && *status == DataflowDeferredReason::None) status = computeLifetimes();
    if (succeeded(status) && *status == DataflowDeferredReason::None) status = checkEmpirical();
    if (succeeded(status) && *status == DataflowDeferredReason::None) status = checkInvariants();
    if (failed(status)) return failure();

    return AnalysisResult{std::move(function), *status};
  }
};

FailureOr<DataflowAnalyzer::AnalysisResult> DataflowAnalyzer::analyze(DenseI64ArrayAttr grid)
{
  return impl_->analyze(grid);
}

DataflowAnalyzer::DataflowAnalyzer(FunctionOpInterface func) : impl_(std::make_unique<Impl>(func)) {}

DataflowAnalyzer::~DataflowAnalyzer()                                      = default;
DataflowAnalyzer::DataflowAnalyzer(DataflowAnalyzer&&) noexcept            = default;
DataflowAnalyzer& DataflowAnalyzer::operator=(DataflowAnalyzer&&) noexcept = default;

} // namespace triton
} // namespace mlir
