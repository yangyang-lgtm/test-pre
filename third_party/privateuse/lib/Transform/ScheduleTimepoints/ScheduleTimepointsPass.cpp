#include "privateuse/Transform/ScheduleTimepoints/Passes.h"

#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Interfaces/ViewLikeInterface.h"
#include "privateuse/Dialect/PuStream/IR/PuStreamDialect.h"

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

struct PendingWaits
{
  DenseMap<Value, Value> writers;
  DenseMap<Value, SmallVector<Value>> readers;
};

class Scheduler
{
public:
  explicit Scheduler(MLIRContext* context) : context(context) {}

  void scheduleBlock(Block& block)
  {
    PendingWaits pending;
    SmallVector<Value> tokens = runBlockImpl(block, pending);
    drain(block.getTerminator(), tokens);
  }

private:
  void drain(Operation* terminator, ArrayRef<Value> tokens)
  {
    if (tokens.empty()) return;
    OpBuilder builder(terminator);
    Value joined = tokens.size() == 1 ? tokens.front()
                                      : pustream::JoinOp::create(builder, terminator->getLoc(), timepoint(), tokens);
    pustream::AwaitOp::create(builder, terminator->getLoc(), joined);
  }

  Type timepoint() const { return pustream::TimepointType::get(context); }

  SmallVector<Value> runBlockImpl(Block& block, PendingWaits& pending)
  {
    SmallVector<Value> tokens;
    for (Operation& op : llvm::make_early_inc_range(block)) schedule(&op, pending, tokens);
    return tokens;
  }

  void schedule(Operation* op, PendingWaits& pending, SmallVector<Value>& tokens)
  {
    if (auto ifOp = dyn_cast<scf::IfOp>(op)) {
      scheduleRegion(ifOp.getThenRegion(), pending);
      scheduleRegion(ifOp.getElseRegion(), pending);
      return;
    }
    if (auto forOp = dyn_cast<scf::ForOp>(op)) {
      scheduleRegion(forOp.getRegion(), pending);
      return;
    }
    if (auto dispatch = dyn_cast<pustream::AsyncDispatchOp>(op)) {
      scheduleDispatch(dispatch, pending, tokens);
      return;
    }
    if (auto copy = dyn_cast<pustream::AsyncCopyOp>(op)) {
      scheduleAsyncCopy(copy, pending, tokens);
      return;
    }
    if (auto copy = dyn_cast<memref::CopyOp>(op)) {
      scheduleSyncCopy(copy, pending);
      return;
    }
  }

  void scheduleRegion(Region& region, PendingWaits& pending)
  {
    if (region.empty()) return;
    PendingWaits inner = pending;
    SmallVector<Value> tokens;
    for (Block& block : region) llvm::append_range(tokens, runBlockImpl(block, inner));
    for (Block& block : region) drain(block.getTerminator(), tokens);
  }

  SmallVector<Value> collectDependencies(ArrayRef<Value> reads, ArrayRef<Value> writes, const PendingWaits& pending)
  {
    SmallVector<Value> deps;
    SmallPtrSet<Value, 8> seen;
    auto add = [&](Value token) {
      if (token && seen.insert(token).second) deps.push_back(token);
    };

    for (Value base : reads) add(pending.writers.lookup(base));
    for (Value base : writes) {
      add(pending.writers.lookup(base));
      for (Value token : pending.readers.lookup(base)) add(token);
    }
    return deps;
  }

  void recordWrites(ArrayRef<Value> writes, PendingWaits& pending, Value token)
  {
    for (Value base : writes) {
      pending.readers.erase(base);
      if (token)
        pending.writers[base] = token;
      else
        pending.writers.erase(base);
    }
  }

  void recordReads(ArrayRef<Value> reads, PendingWaits& pending, Value token)
  {
    if (!token) return;
    for (Value base : reads) pending.readers[base].push_back(token);
  }

  void scheduleDispatch(pustream::AsyncDispatchOp op, PendingWaits& pending, SmallVector<Value>& tokens)
  {
    SmallVector<Value> reads;
    SmallVector<Value> writes;
    for (int64_t index : op.getReads()) reads.push_back(viewBase(op.getArgs()[index]));
    for (int64_t index : op.getWrites()) writes.push_back(viewBase(op.getArgs()[index]));

    op.getAwaitsMutable().append(collectDependencies(reads, writes, pending));
    recordWrites(writes, pending, op.getDone());
    recordReads(reads, pending, op.getDone());
    tokens.push_back(op.getDone());
  }

  void scheduleAsyncCopy(pustream::AsyncCopyOp op, PendingWaits& pending, SmallVector<Value>& tokens)
  {
    Value source = viewBase(op.getSource());
    Value target = viewBase(op.getTarget());
    op.getAwaitsMutable().append(collectDependencies({source}, {target}, pending));

    recordWrites({target}, pending, op.getDone());
    recordReads({source}, pending, op.getDone());
    tokens.push_back(op.getDone());
  }

  void scheduleSyncCopy(memref::CopyOp op, PendingWaits& pending)
  {
    Value source            = viewBase(op.getSource());
    Value target            = viewBase(op.getTarget());
    SmallVector<Value> deps = collectDependencies({source}, {target}, pending);
    drain(op, deps);
    recordWrites({target}, pending, Value{});
  }

  MLIRContext* context;
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

    Scheduler scheduler(&getContext());
    for (auto func : funcs)
      for (Block& block : func.getFunctionBody()) scheduler.scheduleBlock(block);
  }
};

} // namespace

std::unique_ptr<OperationPass<ModuleOp>> triton::createScheduleTimepointsPass()
{
  return std::make_unique<ScheduleTimepointsPass>();
}
