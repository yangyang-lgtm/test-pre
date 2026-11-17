#include "privateuse/Analysis/DataflowAnalysis.h"
#include "privateuse/Utils/Constants.h"
#include "privateuse/Utils/DataflowAnalysis/Passes.h"

#include "llvm/Support/raw_ostream.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/BuiltinOps.h"

#include <algorithm>
#include <array>
#include <limits>
#include <string>

#define GEN_PASS_DECL_ANALYZEDATAFLOW
#define GEN_PASS_DEF_ANALYZEDATAFLOW
#include "privateuse/Utils/DataflowAnalysis/Passes.h.inc"

using namespace mlir;
using namespace mlir::triton;

namespace {

using AnalysisResult = DataflowAnalyzer::AnalysisResult;

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

std::string valueName(Value value, AsmState& asmState)
{
  std::string text;
  llvm::raw_string_ostream os(text);
  value.printAsOperand(os, asmState);
  os.flush();
  if (text.empty() && isa<BlockArgument>(value))
    text = "%arg" + std::to_string(cast<BlockArgument>(value).getArgNumber());
  return text;
}

std::string loopLabel(const LoopInfo& loop, AsmState& asmState)
{
  if (loop.isPid) return "pid";
  scf::ForOp forOp = loop.op;
  std::string name = valueName(forOp.getInductionVar(), asmState);
  if (!name.empty() && name.front() == '%') name.erase(name.begin());
  while (!name.empty() && name.back() == '_') name.pop_back();
  if (name.empty()) name = "l" + std::to_string(loop.id);
  return name;
}

Operation* operationOf(FunctionOpInterface func)
{
  return func.getOperation();
}

struct FunctionNames
{
  explicit FunctionNames(const FunctionInfo& function) : asmState(operationOf(function.op))
  {
    loopLabels.reserve(function.loops.size());
    for (const LoopInfo& loop : function.loops) loopLabels.push_back(loopLabel(loop, asmState));
  }

  std::string value(Value value) { return valueName(value, asmState); }

  AsmState asmState;
  SmallVector<std::string> loopLabels;
};

std::string pathLabel(const FunctionInfo& function, ArrayRef<std::string> loopLabels, unsigned pathId)
{
  std::string text;
  for (const PathInfo* node : pathNodes(function, pathId)) {
    if (!text.empty()) text += '.';
    if (node->kind == PathKind::Loop) {
      text += loopLabels[node->loopId];
    } else {
      text += "if" + std::to_string(node->ifId);
      text += node->kind == PathKind::IfThen ? ".then" : ".else";
    }
  }
  return text;
}

std::string formatBlockPos(ArrayRef<int64_t> blockPos)
{
  std::string text = "[";
  for (size_t index = 0; index < blockPos.size(); ++index) {
    if (index != 0) text += ',';
    if (blockPos[index] == std::numeric_limits<int64_t>::max())
      text += "inf";
    else
      text += std::to_string(blockPos[index]);
  }
  text += ']';
  return text;
}

std::string
formatAt(const FunctionInfo& function, ArrayRef<std::string> loopLabels, unsigned pathId, int64_t blockOrder)
{
  SmallVector<int64_t, 4> blockPos;
  for (const PathInfo* node : pathNodes(function, pathId)) blockPos.push_back(node->blockPosInParent);
  blockPos.push_back(blockOrder);

  std::string text = pathLabel(function, loopLabels, pathId);
  if (text.empty()) text = "root";
  text += formatBlockPos(blockPos);
  return text;
}

std::string formatDelta(int64_t delta)
{
  if (delta > 0) return "delta=+" + std::to_string(delta);
  return "delta=" + std::to_string(delta);
}

std::string formatPoint(const FunctionInfo& function, ArrayRef<std::string> loopLabels, const ProgramPoint& point)
{
  if (point.isLoopExit) return "exit(l" + std::to_string(point.exitLoopId) + ')' + formatBlockPos(point.blockPos);
  std::string text = pathLabel(function, loopLabels, point.pathId);
  if (text.empty()) text = "root";
  if (point.blockPos.empty()) {
    text += "[0]";
  } else {
    text += formatBlockPos(point.blockPos);
  }
  if (point.delta != 0) text += ' ' + formatDelta(point.delta);
  return text;
}

std::string formatFlags(const Access& access)
{
  std::string text;
  if (access.carrierId >= 0) {
    text += " carrier=c" + std::to_string(access.carrierId);
    text += " branch=" + toString(access.branch).str();
    text += " guard=" + toString(access.guard).str();
  }
  return text;
}

std::string formatFlags(const Transfer& transfer)
{
  std::string text;
  if (transfer.carrierId >= 0) {
    text += " carrier=c" + std::to_string(transfer.carrierId);
    text += " branch=" + toString(transfer.branch).str();
    text += " guard=" + toString(transfer.guard).str();
  }
  return text;
}

void printReport(ArrayRef<AnalysisResult> functions)
{
  unsigned stationCount = 0, deadCount = 0, d2s = 0, s2d = 0, s2s = 0;
  unsigned placeCount = 0, transferCount = 0, carrierCount = 0, unresolved = 0, tailWrites = 0;
  unsigned deferredCount = 0;
  std::array<unsigned, static_cast<size_t>(DataflowDeferredReason::Count)> reasons{};
  for (const AnalysisResult& analysis : functions) {
    if (analysis.isDeferred()) {
      ++deferredCount;
      ++reasons[static_cast<size_t>(analysis.deferred)];
      continue;
    }
    const FunctionInfo& function = analysis.info;
    for (const Place& place : function.places) {
      ++placeCount;
      if (place.kind != PlaceKind::SramAlloc) continue;
      ++stationCount;
      if (place.dead) ++deadCount;
    }
    for (const Transfer& transfer : function.transfers) {
      ++transferCount;
      if (transfer.dir == Dir::DramToSram) ++d2s;
      if (transfer.dir == Dir::SramToDram) ++s2d;
      if (transfer.dir == Dir::SramToSram) ++s2s;
    }
    carrierCount += function.carriers.size();
    unresolved   += function.unresolved;
    tailWrites   += function.tailWrites;
  }

  for (const AnalysisResult& analysis : functions) {
    FunctionOpInterface func = analysis.info.op;
    if (analysis.isDeferred()) {
      llvm::errs() << "[privateuse-dataflow] func @" << func.getName() << " deferred=" << toString(analysis.deferred)
                   << '\n';
      continue;
    }
    const FunctionInfo& function = analysis.info;
    llvm::errs() << "[privateuse-dataflow] func @" << func.getName() << " loops={";
    bool first = true;
    for (const LoopInfo& loop : function.loops) {
      if (!first) llvm::errs() << ',';
      first = false;
      llvm::errs() << loop.id << ':';
      if (loop.isPid)
        llvm::errs() << "pid";
      else if (loop.trip >= 0)
        llvm::errs() << "static(trip=" << loop.trip << ')';
      else
        llvm::errs() << "symbolic";
    }
    llvm::errs() << "} places=" << function.places.size() << " transfers=" << function.transfers.size()
                 << " carriers=" << function.carriers.size() << '\n';
  }

  for (const AnalysisResult& analysis : functions) {
    if (analysis.isDeferred()) continue;
    const FunctionInfo& function = analysis.info;
    FunctionNames names(function);
    for (const Place& place : function.places) {
      llvm::errs() << "[privateuse-dataflow] place p" << place.id << " kind=" << toString(place.kind)
                   << " space=" << toString(place.space) << " anchor=" << names.value(place.anchor);
      if (place.kind != PlaceKind::KernelArg)
        llvm::errs() << " path=" << formatAt(function, names.loopLabels, place.defPathId, place.allocOrder);
      if (place.kind == PlaceKind::SramAlloc)
        llvm::errs() << " producers=" << place.producers.size() << " consumers=" << place.consumers.size();
      llvm::errs() << '\n';
    }
  }

  for (const AnalysisResult& analysis : functions) {
    if (analysis.isDeferred()) continue;
    const FunctionInfo& function = analysis.info;
    FunctionNames names(function);
    for (unsigned index = 0; index < function.transfers.size(); ++index) {
      const Transfer& transfer = function.transfers[index];
      llvm::errs() << "[privateuse-dataflow] transfer t" << index << " dir=" << toString(transfer.dir) << " src=p"
                   << transfer.src << " dst=p" << transfer.dst << " opKind=" << toString(transfer.opKind)
                   << " masked=" << (transfer.masked ? 1 : 0) << " dynamic=" << (transfer.dynamic ? 1 : 0);
      if (transfer.groupId >= 0) llvm::errs() << " group=" << transfer.groupId;
      llvm::errs() << " at=" << formatAt(function, names.loopLabels, transfer.pathId, transfer.blockOrder);
      if (transfer.delta != 0) llvm::errs() << ' ' << formatDelta(transfer.delta);
      llvm::errs() << formatFlags(transfer) << '\n';
    }
  }

  for (const AnalysisResult& analysis : functions) {
    if (analysis.isDeferred()) continue;
    const FunctionInfo& function = analysis.info;
    FunctionNames names(function);
    for (const Carrier& carrier : function.carriers) {
      llvm::errs() << "[privateuse-dataflow] carrier c" << carrier.id << " loop=" << carrier.loopId
                   << " arg=" << names.value(carrier.arg) << " init=p" << carrier.initPlace << " yield=p"
                   << carrier.yieldPlace << " rotation=" << (carrier.rotation ? 1 : 0)
                   << " inPlace=" << (carrier.inPlace ? 1 : 0) << " wroteBody=" << (carrier.wroteBody ? 1 : 0) << '\n';
    }
  }

  for (const AnalysisResult& analysis : functions) {
    if (analysis.isDeferred()) continue;
    const FunctionInfo& function = analysis.info;
    FunctionNames names(function);
    for (const Place& place : function.places) {
      if (place.kind != PlaceKind::SramAlloc) continue;
      llvm::errs() << "[privateuse-dataflow] station p" << place.id
                   << " start=" << formatPoint(function, names.loopLabels, place.start)
                   << " retire=" << formatPoint(function, names.loopLabels, place.retire)
                   << " carried=" << (place.carried ? 1 : 0) << " dead=" << (place.dead ? 1 : 0)
                   << " through=" << (place.through ? 1 : 0) << " indirect=" << (place.indirect ? 1 : 0) << '\n';
      for (const Access& access : place.accesses) {
        llvm::errs() << "[privateuse-dataflow] access p" << place.id << " role=" << toString(access.role)
                     << " at=" << formatAt(function, names.loopLabels, access.pathId, access.blockOrder) << ' '
                     << formatDelta(access.delta) << formatFlags(access) << '\n';
      }
    }
  }

  if (deferredCount == 0) {
    llvm::errs() << "[privateuse-dataflow] check correctness ok\n";
    llvm::errs() << "[privateuse-dataflow] check empirical ok\n";
  }

  llvm::errs() << "[privateuse-dataflow] summary funcs=" << functions.size() << " places=" << placeCount
               << " stations=" << stationCount << " transfers={D2S=" << d2s << ",S2D=" << s2d << ",S2S=" << s2s
               << "} carriers=" << carrierCount << " dead=" << deadCount << " tailWrites=" << tailWrites
               << " unresolved=" << unresolved << " deferred=" << deferredCount;
  if (deferredCount != 0) {
    llvm::errs() << " reasons={";
    bool first = true;
    for (size_t reason = 1; reason < reasons.size(); ++reason) {
      if (!reasons[reason]) continue;
      if (!first) llvm::errs() << ',';
      first = false;
      llvm::errs() << toString(static_cast<DataflowDeferredReason>(reason)) << '=' << reasons[reason];
    }
    llvm::errs() << '}';
  }
  llvm::errs() << '\n';
}

struct AnalyzeDataflowPass : public ::impl::AnalyzeDataflowBase<AnalyzeDataflowPass>
{
  using AnalyzeDataflowBase::AnalyzeDataflowBase;

  void runOnOperation() override
  {
    ModuleOp module        = getOperation();
    DenseI64ArrayAttr grid = module->getAttrOfType<DenseI64ArrayAttr>(kGridAttr);
    if (!grid) {
      module.emitError("missing privateuse.grid module attribute");
      signalPassFailure();
      return;
    }
    SmallVector<AnalysisResult, 0> functions;
    WalkResult walkResult = module.walk([&](FunctionOpInterface func) -> WalkResult {
      if (func.isExternal()) return WalkResult::advance();
      DataflowAnalyzer analyzer(func);
      FailureOr<AnalysisResult> function = analyzer.analyze(grid);
      if (failed(function)) return WalkResult::interrupt();
      functions.push_back(std::move(*function));
      return WalkResult::advance();
    });
    if (walkResult.wasInterrupted()) {
      signalPassFailure();
      return;
    }
    printReport(functions);
  }
};

} // namespace

std::unique_ptr<OperationPass<ModuleOp>> mlir::triton::createAnalyzeDataflowPass()
{
  return std::make_unique<AnalyzeDataflowPass>();
}
