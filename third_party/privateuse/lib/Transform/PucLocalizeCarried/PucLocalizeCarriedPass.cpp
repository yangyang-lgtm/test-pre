#include "privateuse/Transform/PucLocalizeCarried/PucLocalizeCarried.h"

#include "llvm/ADT/STLExtras.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "privateuse/Dialect/Puc/IR/PucDialect.h"

#define DEBUG_TYPE "puc-localize-carried"

using namespace mlir;
using namespace mlir::triton;

#define GEN_PASS_DECL_PUCLOCALIZECARRIED
#include "privateuse/Transform/PucLocalizeCarried/Passes.h.inc"

#define GEN_PASS_DEF_PUCLOCALIZECARRIED
#include "privateuse/Transform/PucLocalizeCarried/Passes.h.inc"

namespace {

void copyAttrs(Operation* from, Operation* to)
{
  NamedAttrList attrs(from->getAttrDictionary());
  attrs.erase("operandSegmentSizes");
  to->setAttrs(attrs.getDictionary(from->getContext()));
}

bool dependsOnAny(Value v, const llvm::SmallPtrSetImpl<Value>& vars, Block* block)
{
  SmallVector<Value> work{v};
  llvm::SmallPtrSet<Operation*, 8> seen;
  while (!work.empty()) {
    Value cur = work.pop_back_val();
    if (vars.contains(cur)) return true;
    Operation* def = cur.getDefiningOp();
    if (!def || def->getBlock() != block) continue;
    if (!seen.insert(def).second) continue;
    for (Value operand : def->getOperands()) work.push_back(operand);
  }
  return false;
}

void emitParallelAssigns(OpBuilder& builder, Location loc, ArrayRef<std::pair<Value, Value>> pairs, Block* block)
{
  llvm::SmallPtrSet<Value, 8> vars;
  for (auto& [var, value] : pairs) vars.insert(var);

  bool needStaging = false;
  for (auto& [var, value] : pairs)
    if (dependsOnAny(value, vars, block)) needStaging = true;

  SmallVector<Value> staged;
  if (needStaging)
    for (auto& [var, value] : pairs) staged.push_back(puc::LocalOp::create(builder, loc, value.getType(), value));

  for (auto [index, pair] : llvm::enumerate(pairs))
    puc::AssignOp::create(builder, loc, pair.first, needStaging ? staged[index] : pair.second);
}

void localizeFor(puc::ForOp loop)
{
  if (loop.getNumResults() == 0) return;

  Location loc = loop.getLoc();
  OpBuilder pre(loop);

  SmallVector<Value> vars;
  for (auto [index, result] : llvm::enumerate(loop.getResults())) {
    Value init = loop.getInitArgs()[index];
    Value arg  = loop.getRegionIterArg(index);
    Value var  = puc::LocalOp::create(pre, loc, arg.getType(), init);
    vars.push_back(var);
    arg.replaceAllUsesWith(var);
  }

  auto yield = cast<puc::YieldOp>(loop.getBody()->getTerminator());
  SmallVector<std::pair<Value, Value>> pairs;
  for (auto [index, yielded] : llvm::enumerate(yield.getOperands()))
    if (yielded != vars[index]) pairs.push_back({vars[index], yielded});

  OpBuilder body(yield);
  emitParallelAssigns(body, loc, pairs, loop.getBody());
  yield->setOperands(ValueRange{});

  for (auto [result, var] : llvm::zip(loop.getResults(), vars)) result.replaceAllUsesWith(var);

  OpBuilder builder(loop);
  auto pipelined = puc::ForOp::create(builder, loc, loop.getLowerBound(), loop.getUpperBound(), loop.getStep());
  copyAttrs(loop, pipelined);

  Block* oldBody = loop.getBody();
  Block* newBody = pipelined.getBody();
  oldBody->getArgument(0).replaceAllUsesWith(newBody->getArgument(0));
  newBody->getTerminator()->erase();
  newBody->getOperations().splice(newBody->end(), oldBody->getOperations());
  loop.erase();
}

void localizeIf(puc::IfOp op)
{
  if (op.getNumResults() == 0) return;

  Location loc = op.getLoc();
  OpBuilder pre(op);

  SmallVector<Value> vars;
  for (Value result : op.getResults()) vars.push_back(puc::LocalOp::create(pre, loc, result.getType(), Value{}));

  auto assignInto = [&](Block* block) {
    auto yield = cast<puc::YieldOp>(block->getTerminator());
    SmallVector<std::pair<Value, Value>> pairs;
    for (auto [index, yielded] : llvm::enumerate(yield.getOperands())) pairs.push_back({vars[index], yielded});

    OpBuilder body(yield);
    emitParallelAssigns(body, loc, pairs, block);
    yield->setOperands(ValueRange{});
  };

  assignInto(&op.getThenRegion().front());
  if (!op.getElseRegion().empty()) assignInto(&op.getElseRegion().front());

  for (auto [result, var] : llvm::zip(op.getResults(), vars)) result.replaceAllUsesWith(var);

  OpBuilder builder(op);
  auto newIf = puc::IfOp::create(builder, loc, op.getCondition());
  copyAttrs(op, newIf);

  for (auto [oldRegion, newRegion] : llvm::zip(op->getRegions(), newIf->getRegions())) {
    if (oldRegion.empty()) continue;
    Block* oldBlock = &oldRegion.front();
    Block* newBlock = &newRegion.front();
    if (newBlock->mightHaveTerminator()) newBlock->getTerminator()->erase();
    newBlock->getOperations().splice(newBlock->end(), oldBlock->getOperations());
  }
  op.erase();
}

void eraseDeadLocals(puc::FuncOp func)
{
  SmallVector<puc::LocalOp> locals;
  func.walk([&](puc::LocalOp local) { locals.push_back(local); });
  for (puc::LocalOp local : llvm::reverse(locals))
    if (local.getResult().use_empty()) local.erase();
}

struct PucLocalizeCarriedPass : public ::impl::PucLocalizeCarriedBase<PucLocalizeCarriedPass>
{
  using PucLocalizeCarriedBase::PucLocalizeCarriedBase;

  void getDependentDialects(DialectRegistry& registry) const override { registry.insert<puc::PucDialect>(); }

  void runOnOperation() override
  {
    ModuleOp moduleOp = getOperation();

    moduleOp.walk([&](puc::FuncOp funcOp) {
      if (funcOp.getBody().empty()) return;

      SmallVector<puc::ForOp> loops;
      funcOp.walk([&](puc::ForOp loop) { loops.push_back(loop); });
      for (puc::ForOp loop : llvm::reverse(loops)) localizeFor(loop);

      SmallVector<puc::IfOp> ifs;
      funcOp.walk([&](puc::IfOp op) { ifs.push_back(op); });
      for (puc::IfOp op : llvm::reverse(ifs)) localizeIf(op);

      eraseDeadLocals(funcOp);
    });
  }
};

} // namespace

std::unique_ptr<OperationPass<ModuleOp>> triton::createPucLocalizeCarriedPass()
{
  return std::make_unique<PucLocalizeCarriedPass>();
}
