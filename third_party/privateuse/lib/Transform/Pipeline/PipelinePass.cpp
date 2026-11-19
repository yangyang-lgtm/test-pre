#include "privateuse/Transform/Pipeline/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Interfaces/ViewLikeInterface.h"
#include "privateuse/Dialect/PuStream/IR/PuStreamDialect.h"
#include "privateuse/Utils/Constants.h"

#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/STLExtras.h"

#include <optional>
#include <string>

#define GEN_PASS_DECL_PIPELINE
#include "privateuse/Transform/Pipeline/Passes.h.inc"

#define GEN_PASS_DEF_PIPELINE
#include "privateuse/Transform/Pipeline/Passes.h.inc"

using namespace mlir;
using namespace mlir::triton;

//===----------------------------------------------------------------------===//
// Tile loop pipelining.
//
// The canonical shape this pass looks for:
//
//   scf.for %i {
//     async.copy %dram, %slot(%tile)      // the one load
//     async.dispatch @body(%tile, ...)    // one or more consumers
//     async.copy %slot(%out), %dram       // optional, the one store
//   }
//
// Pipelining means: hoist the tile buffers, give them a stage dimension, prefetch
// N-1 iterations ahead and let compute(i) run against load(i+N-1)/store(i-S).
//
// A loop that is rewritten is owned entirely by this pass: it constructs the
// slots itself, emits the per-slot waits and the store barrier itself, and tags
// the loop with `pustream.pipelined` so the wait-insertion pass leaves it alone.
// That tag is what keeps the two passes from fighting: without it, the
// wait-insertion pass would drain everything at the loop terminator and there
// would be no overlap left to gain.
//
// Anything that does not match is left untouched. Bailing out is always correct
// (the loop simply stays single-buffered) and is preferred over a rewrite that
// cannot be justified -- in particular the masked-load path, where the padding
// fill writes the very buffer the prefetch would be filling.
// ==========================================================================//

namespace {

Value viewBase(Value value)
{
  while (Operation* def = value.getDefiningOp()) {
    auto view = dyn_cast<ViewLikeOpInterface>(def);
    if (!view) break;
    Value source = view.getViewSource();
    if (!source || !isa<BaseMemRefType>(source.getType())) break;
    value = source;
  }
  return value;
}

Value baseOf(Value value)
{
  if (auto slot = value.getDefiningOp<pustream::SlotOp>()) return viewBase(slot.getBuffer());
  return viewBase(value);
}

// Is `value` computable from the induction variable alone (plus values defined
// outside the loop)? Only then can the prefetch rebuild the address for
// iteration i + (N-1) * step, which is what the shifted clone needs.
bool dependsOnlyOnIV(Value value, scf::ForOp loop)
{
  Value iv = loop.getInductionVar();
  SmallVector<Value> worklist{value};
  SmallPtrSet<Value, 16> seen;

  while (!worklist.empty()) {
    Value current = worklist.pop_back_val();
    if (!seen.insert(current).second) continue;
    if (current == iv) continue;

    if (auto argument = dyn_cast<BlockArgument>(current)) {
      // A body argument other than the induction variable is a loop-carried
      // value: the prefetch would have to read a different iteration's value.
      if (argument.getOwner() == loop.getBody()) return false;
      continue;
    }

    Operation* def = current.getDefiningOp();
    if (!def) return false;
    if (!loop->isAncestor(def)) continue; // hoisted out of the loop: invariant
    if (!isMemoryEffectFree(def)) return false;
    llvm::append_range(worklist, def->getOperands());
  }
  return true;
}

// Static trip count, or nothing when it cannot be folded.
std::optional<int64_t> constantTripCount(scf::ForOp loop)
{
  auto lower = loop.getLowerBound().getDefiningOp<arith::ConstantIndexOp>();
  auto upper = loop.getUpperBound().getDefiningOp<arith::ConstantIndexOp>();
  auto step  = loop.getStep().getDefiningOp<arith::ConstantIndexOp>();
  if (!lower || !upper || !step || step.value() <= 0) return std::nullopt;
  const int64_t distance = upper.value() - lower.value();
  if (distance <= 0) return 0;
  return (distance + step.value() - 1) / step.value();
}

struct Candidate
{
  scf::ForOp loop;
  pustream::AsyncCopyOp load;
  Value tile;                              // base alloc behind the load slot
  SmallVector<pustream::AsyncDispatchOp> consumers;
  pustream::AsyncCopyOp store;             // null for a reduction loop
};

std::optional<Candidate> match(scf::ForOp loop, std::string& reason)
{
  if (loop->hasAttr(triton::kPipelinedAttr)) {
    reason = "already pipelined";
    return std::nullopt;
  }

  SmallVector<pustream::AsyncCopyOp> loads, stores;
  SmallVector<pustream::AsyncDispatchOp> dispatches;
  loop.getBody()->walk([&](Operation* op) {
    if (auto copy = dyn_cast<pustream::AsyncCopyOp>(op)) {
      if (isa<pustream::SlotType>(copy.getTarget().getType()))
        loads.push_back(copy);
      else
        stores.push_back(copy);
      return;
    }
    if (auto dispatch = dyn_cast<pustream::AsyncDispatchOp>(op)) dispatches.push_back(dispatch);
  });

  if (loads.size() != 1) {
    reason = loads.empty() ? "no load in the loop body" : "more than one load";
    return std::nullopt;
  }
  if (stores.size() > 1) {
    reason = "more than one store";
    return std::nullopt;
  }

  pustream::AsyncCopyOp load = loads.front();
  auto slot = load.getTarget().getDefiningOp<pustream::SlotOp>();
  if (!slot) {
    reason = "the load target is not a pustream.slot";
    return std::nullopt;
  }

  Value tile = viewBase(slot.getBuffer());

  SmallVector<pustream::AsyncDispatchOp> consumers;
  for (pustream::AsyncDispatchOp dispatch : dispatches) {
    bool reads = false, writes = false;
    for (int64_t index : dispatch.getReads())
      reads |= baseOf(dispatch.getArgs()[index]) == tile;
    for (int64_t index : dispatch.getWrites())
      writes |= baseOf(dispatch.getArgs()[index]) == tile;
    if (writes) {
      reason = "the body writes the loaded tile (masked fill); needs the tail-slot transform";
      return std::nullopt;
    }
    if (reads) consumers.push_back(dispatch);
  }
  if (consumers.empty()) {
    reason = "no dispatch consumes the loaded tile";
    return std::nullopt;
  }

  if (!dependsOnlyOnIV(load.getSource(), loop)) {
    reason = "the load address depends on something other than the induction variable";
    return std::nullopt;
  }

  if (std::optional<int64_t> trips = constantTripCount(loop); trips && *trips < 2) {
    reason = "trip count is below 2, there is nothing to overlap";
    return std::nullopt;
  }

  Candidate candidate;
  candidate.loop       = loop;
  candidate.load       = load;
  candidate.tile       = tile;
  candidate.consumers  = std::move(consumers);
  candidate.store      = stores.empty() ? pustream::AsyncCopyOp{} : stores.front();
  return candidate;
}

struct PipelinePass : public ::impl::PipelineBase<PipelinePass>
{
  using PipelineBase::PipelineBase;

  void runOnOperation() override
  {
    SmallVector<scf::ForOp> loops;
    getOperation().walk([&](scf::ForOp loop) { loops.push_back(loop); });

    for (scf::ForOp loop : loops) {
      std::string reason;
      std::optional<Candidate> candidate = match(loop, reason);
      if (!candidate) {
        if (verbose) loop.emitRemark() << "not pipelined: " << reason;
        continue;
      }

      // The rewrite itself is the next step; for now the pass only classifies.
      if (verbose)
        loop.emitRemark() << "pipeline candidate: " << candidate->consumers.size() << " consumer(s), "
                          << (candidate->store ? "streaming" : "reduction")
                          << ", stages=" << static_cast<int64_t>(stages.getValue());
    }
  }
};

} // namespace

std::unique_ptr<OperationPass<ModuleOp>> triton::createPipelinePass()
{
  return std::make_unique<PipelinePass>();
}
