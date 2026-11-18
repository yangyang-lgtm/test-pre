#include "privateuse/Transform/HoistInvariantStaging/Passes.h"

#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Interfaces/DestinationStyleOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Interfaces/ViewLikeInterface.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"

#define GEN_PASS_DECL_HOISTINVARIANTSTAGING
#include "privateuse/Transform/HoistInvariantStaging/Passes.h.inc"

#define GEN_PASS_DEF_HOISTINVARIANTSTAGING
#include "privateuse/Transform/HoistInvariantStaging/Passes.h.inc"

using namespace mlir;
using namespace mlir::triton;

namespace {

struct HoistInvariantStagingPass : public ::impl::HoistInvariantStagingBase<HoistInvariantStagingPass>
{
  using HoistInvariantStagingBase::HoistInvariantStagingBase;

  void getDependentDialects(DialectRegistry& registry) const override
  {
    registry.insert<bufferization::BufferizationDialect, memref::MemRefDialect, scf::SCFDialect>();
  }

  void runOnOperation() override
  {
    SmallVector<scf::ForOp> loops;
    getOperation().walk([&](scf::ForOp loop) { loops.push_back(loop); });
    for (scf::ForOp loop : llvm::reverse(loops)) hoistLoop(loop);
  }

private:
  Value viewBase(Value value)
  {
    while (Operation* def = value.getDefiningOp()) {
      auto view = dyn_cast<ViewLikeOpInterface>(def);
      if (!view) break;
      Value source = view.getViewSource();
      if (!source || !isa<MemRefType>(source.getType())) break;
      value = source;
    }
    return value;
  }

  void hoistLoop(scf::ForOp loop)
  {
    SmallVector<memref::CopyOp> copies;
    for (Operation& op : *loop.getBody())
      if (auto copy = dyn_cast<memref::CopyOp>(&op)) copies.push_back(copy);
    for (memref::CopyOp copy : copies) hoistCopy(loop, copy);
  }

  void hoistCopy(scf::ForOp loop, memref::CopyOp copy)
  {
    Value source = copy.getSource();
    if (!loop.isDefinedOutsideOfLoop(source)) return;

    auto alloc = copy.getTarget().getDefiningOp<memref::AllocOp>();
    if (!alloc || alloc->getBlock() != copy->getBlock()) return;
    if (!alloc.getType().hasStaticShape()) return;

    SmallVector<bufferization::ToTensorOp> bridges;
    for (Operation* user : alloc.getResult().getUsers()) {
      if (user == copy.getOperation()) continue;
      auto bridge = dyn_cast<bufferization::ToTensorOp>(user);
      if (!bridge || bridge->getBlock() != copy->getBlock()) return;
      bridges.push_back(bridge);
    }

    for (bufferization::ToTensorOp bridge : bridges) {
      for (Operation* user : bridge.getResult().getUsers()) {
        if (user->getBlock() != copy->getBlock() || user->hasTrait<OpTrait::IsTerminator>()) return;
        if (auto destination = dyn_cast<DestinationStyleOpInterface>(user))
          if (llvm::is_contained(destination.getDpsInits(), bridge.getResult())) return;
      }
    }

    if (writesMayAliasSource(loop, copy, viewBase(source))) return;

    alloc->moveBefore(loop);
    copy->moveBefore(loop);
    for (bufferization::ToTensorOp bridge : bridges) bridge->moveBefore(loop);
  }

  bool writesMayAliasSource(scf::ForOp loop, memref::CopyOp copy, Value sourceBase)
  {
    return loop
      .walk([&](Operation* op) {
        if (op == copy.getOperation()) return WalkResult::advance();
        if (op->hasTrait<OpTrait::IsTerminator>() || op->hasTrait<OpTrait::HasRecursiveMemoryEffects>())
          return WalkResult::advance();

        auto interface = dyn_cast<MemoryEffectOpInterface>(op);
        if (!interface) {
          return isMemoryEffectFree(op) ? WalkResult::advance() : WalkResult::interrupt();
        }

        SmallVector<MemoryEffects::EffectInstance> writes;
        interface.getEffects<MemoryEffects::Write>(writes);
        for (const MemoryEffects::EffectInstance& write : writes) {
          Value value = write.getValue();
          if (!value || viewBase(value) == sourceBase) {
            return WalkResult::interrupt();
          }
        }
        return WalkResult::advance();
      })
      .wasInterrupted();
  }
};

} // namespace

std::unique_ptr<OperationPass<ModuleOp>> mlir::triton::createHoistInvariantStagingPass()
{
  return std::make_unique<HoistInvariantStagingPass>();
}
