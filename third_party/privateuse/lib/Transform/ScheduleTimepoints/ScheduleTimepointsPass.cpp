#include "privateuse/Transform/ScheduleTimepoints/Passes.h"

#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Interfaces/ViewLikeInterface.h"
#include "privateuse/Dialect/PuStream/IR/PuStreamDialect.h"
#include "privateuse/Utils/Constants.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/STLExtras.h"

#define GEN_PASS_DECL_SCHEDULETIMEPOINTS
#include "privateuse/Transform/ScheduleTimepoints/Passes.h.inc"

#define GEN_PASS_DEF_SCHEDULETIMEPOINTS
#include "privateuse/Transform/ScheduleTimepoints/Passes.h.inc"

using namespace mlir;
using namespace mlir::triton;

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

class Scheduler
{
public:
  void scheduleFunction(FunctionOpInterface func)
  {
    sawAsync = false;
    for (Block& block : func.getFunctionBody()) scheduleBlock(block, /*isFunctionBody=*/true);
    if (!sawAsync) return;

    for (Block& block : func.getFunctionBody()) {
      Operation* terminator = block.getTerminator();
      OpBuilder builder(terminator);
      pustream::SyncOp::create(builder, terminator->getLoc());
    }
  }

private:
  void clearCompute()
  {
    computePending = false;
    computeTouched.clear();
  }

  void clearAll()
  {
    pendingDma.clear();
    clearCompute();
  }

  void drainAt(Operation* where)
  {
    OpBuilder builder(where);
    Location loc = where->getLoc();
    SmallVector<Value> slots;
    for (auto& entry : pendingDma) slots.push_back(entry.second);
    for (Value slot : slots) pustream::WaitOp::create(builder, loc, slot);
    if (computePending) pustream::WaitDispatchOp::create(builder, loc);
    clearAll();
  }

  void scheduleBlock(Block& block, bool isFunctionBody)
  {
    for (Operation& op : llvm::make_early_inc_range(block)) {
      if (op.hasTrait<OpTrait::IsTerminator>()) break;
      visit(&op);
    }
    if (!isFunctionBody) drainAt(block.getTerminator());
  }

  void scheduleRegion(Region& region)
  {
    for (Block& block : region) scheduleBlock(block, /*isFunctionBody=*/false);
    clearAll();
  }

  void visit(Operation* op)
  {
    // A pipelined loop is owned by the pipeliner: it emitted its own per-slot
    // waits and its store barrier, so scanning or draining it here would
    // serialise every iteration and undo the overlap.
    if (auto forOp = dyn_cast<scf::ForOp>(op)) {
      if (forOp->hasAttr(triton::kPipelinedAttr)) {
        // The loop still leaves stores in flight, so the kernel exit drain is
        // still this pass's job even though the body is not.
        sawAsync = true;
        return;
      }
      return scheduleRegion(forOp.getRegion());
    }
    if (auto ifOp = dyn_cast<scf::IfOp>(op)) {
      // The two regions are alternatives, not a sequence: each has to start
      // from the state before the if, otherwise the branch scheduled second
      // loses the waits the first one already emitted inside itself.
      const DenseMap<Value, Value> savedPending = pendingDma;
      const SmallPtrSet<Value, 8> savedTouched   = computeTouched;
      const bool savedCompute                    = computePending;

      scheduleRegion(ifOp.getThenRegion());

      pendingDma     = savedPending;
      computeTouched = savedTouched;
      computePending = savedCompute;
      if (!ifOp.getElseRegion().empty()) scheduleRegion(ifOp.getElseRegion());

      clearAll();
      return;
    }
    if (auto copy = dyn_cast<pustream::AsyncCopyOp>(op)) return handleCopy(copy);
    if (auto dispatch = dyn_cast<pustream::AsyncDispatchOp>(op)) return handleDispatch(dispatch);
    if (auto wait = dyn_cast<pustream::WaitOp>(op)) {
      pendingDma.erase(baseOf(wait.getSlot()));
      return;
    }
    if (isa<pustream::WaitDispatchOp>(op)) return clearCompute();
    if (isa<pustream::SyncOp>(op)) return drainAt(op);
    return handleSynchronous(op);
  }

  void handleCopy(pustream::AsyncCopyOp copy)
  {
    OpBuilder builder(copy);
    Location loc = copy.getLoc();
    const bool store = isa<pustream::SlotType>(copy.getSource().getType());
    Value slot      = store ? copy.getSource() : copy.getTarget();
    Value base      = baseOf(slot);

    if (computePending && computeTouched.contains(base)) {
      pustream::WaitDispatchOp::create(builder, loc);
      clearCompute();
    }
    if (auto it = pendingDma.find(base); it != pendingDma.end()) {
      pustream::WaitOp::create(builder, loc, it->second);
      pendingDma.erase(it);
    }
    pendingDma[base] = slot;
    sawAsync         = true;
  }

  void handleDispatch(pustream::AsyncDispatchOp dispatch)
  {
    OpBuilder builder(dispatch);
    Location loc = dispatch.getLoc();

    SmallVector<Value> bases;
    for (int64_t index : dispatch.getReads()) bases.push_back(baseOf(dispatch.getArgs()[index]));
    for (int64_t index : dispatch.getWrites()) bases.push_back(baseOf(dispatch.getArgs()[index]));

    for (Value base : bases) {
      if (auto it = pendingDma.find(base); it != pendingDma.end()) {
        pustream::WaitOp::create(builder, loc, it->second);
        pendingDma.erase(it);
      }
    }

    if (computePending)
      for (Value base : bases)
        if (computeTouched.contains(base)) {
          pustream::WaitDispatchOp::create(builder, loc);
          clearCompute();
          break;
        }

    computePending = true;
    for (Value base : bases) computeTouched.insert(base);
    sawAsync = true;
  }

  void handleSynchronous(Operation* op)
  {
    auto effects = dyn_cast<MemoryEffectOpInterface>(op);
    if (!effects) return;

    SmallVector<Value> bases;
    SmallVector<SideEffects::EffectInstance<MemoryEffects::Effect>> instances;
    effects.getEffects(instances);
    for (auto& instance : instances) {
      if (!isa<MemoryEffects::Read, MemoryEffects::Write>(instance.getEffect())) continue;
      Value value = instance.getValue();
      if (!value || !isa<BaseMemRefType>(value.getType())) continue;
      bases.push_back(baseOf(value));
    }
    if (bases.empty()) return;

    OpBuilder builder(op);
    Location loc = op->getLoc();
    for (Value base : bases) {
      if (auto it = pendingDma.find(base); it != pendingDma.end()) {
        pustream::WaitOp::create(builder, loc, it->second);
        pendingDma.erase(it);
      }
    }
    if (computePending)
      for (Value base : bases)
        if (computeTouched.contains(base)) {
          pustream::WaitDispatchOp::create(builder, loc);
          clearCompute();
          break;
        }
  }

  DenseMap<Value, Value> pendingDma;
  SmallPtrSet<Value, 8> computeTouched;
  bool computePending = false;
  bool sawAsync       = false;
};

struct ScheduleTimepointsPass : public ::impl::ScheduleTimepointsBase<ScheduleTimepointsPass>
{
  using ScheduleTimepointsBase::ScheduleTimepointsBase;

  void runOnOperation() override
  {
    SmallVector<FunctionOpInterface> funcs;
    getOperation().walk([&](FunctionOpInterface func) {
      if (!func.isExternal()) funcs.push_back(func);
    });

    Scheduler scheduler;
    for (auto func : funcs) scheduler.scheduleFunction(func);
  }
};

} // namespace

std::unique_ptr<OperationPass<ModuleOp>> triton::createScheduleTimepointsPass()
{
  return std::make_unique<ScheduleTimepointsPass>();
}
