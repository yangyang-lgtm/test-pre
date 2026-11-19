#include "privateuse/Transform/ScheduleTimepoints/Passes.h"

#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "privateuse/Dialect/PuStream/IR/PuStreamDialect.h"
#include "privateuse/Utils/Constants.h"

#include "llvm/ADT/SmallVector.h"

#define GEN_PASS_DECL_SCHEDULETIMEPOINTS
#include "privateuse/Transform/ScheduleTimepoints/Passes.h.inc"

#define GEN_PASS_DEF_SCHEDULETIMEPOINTS
#include "privateuse/Transform/ScheduleTimepoints/Passes.h.inc"

using namespace mlir;
using namespace mlir::triton;

namespace {

bool ownedByPipeliner(Operation* op)
{
  for (Operation* parent = op->getParentOp(); parent; parent = parent->getParentOp())
    if (parent->hasAttr(triton::kPipelinedAttr)) return true;
  return false;
}

struct ScheduleTimepointsPass : public ::impl::ScheduleTimepointsBase<ScheduleTimepointsPass>
{
  using ScheduleTimepointsBase::ScheduleTimepointsBase;

  void runOnOperation() override
  {
    SmallVector<FunctionOpInterface> functions;
    getOperation().walk([&](FunctionOpInterface func) {
      if (!func.isExternal()) functions.push_back(func);
    });

    for (FunctionOpInterface func : functions) {
      SmallVector<Operation*> copies;
      SmallVector<Operation*> dispatches;
      bool hasAsync = false;
      func->walk([&](Operation* op) {
        const bool async = isa<pustream::AsyncCopyOp>(op) || isa<pustream::AsyncDispatchOp>(op);
        if (!async) return;
        hasAsync = true;
        // A pipelined loop schedules its own body, but its transfers are still
        // in flight at kernel exit, so the drain below is still needed.
        if (ownedByPipeliner(op)) return;
        if (isa<pustream::AsyncCopyOp>(op))
          copies.push_back(op);
        else
          dispatches.push_back(op);
      });
      if (!hasAsync) continue;

      for (Operation* op : copies) {
        auto copy    = cast<pustream::AsyncCopyOp>(op);
        Value staged = isa<pustream::SlotType>(copy.getTarget().getType()) ? copy.getTarget() : copy.getSource();
        OpBuilder builder(op);
        builder.setInsertionPointAfter(op);
        pustream::WaitOp::create(builder, op->getLoc(), staged);
      }

      for (Operation* op : dispatches) {
        OpBuilder builder(op);
        builder.setInsertionPointAfter(op);
        pustream::WaitDispatchOp::create(builder, op->getLoc());
      }

      Block& entry       = func.getFunctionBody().front();
      Operation* terminator = entry.getTerminator();
      OpBuilder builder(terminator);
      pustream::SyncOp::create(builder, terminator->getLoc());
    }
  }
};

} // namespace

std::unique_ptr<OperationPass<ModuleOp>> triton::createScheduleTimepointsPass()
{
  return std::make_unique<ScheduleTimepointsPass>();
}
