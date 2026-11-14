#include "privateuse/Transform/PucSimplifyAssign/PucSimplifyAssign.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Matchers.h"
#include "privateuse/Dialect/Puc/IR/PucDialect.h"

#define DEBUG_TYPE "puc-simplify-assign"

using namespace mlir;
using namespace mlir::triton;

#define GEN_PASS_DECL_PUCSIMPLIFYASSIGN
#include "privateuse/Transform/PucSimplifyAssign/Passes.h.inc"

#define GEN_PASS_DEF_PUCSIMPLIFYASSIGN
#include "privateuse/Transform/PucSimplifyAssign/Passes.h.inc"

namespace {

bool assignedAnywhere(Value local)
{
  for (Operation* user : local.getUsers()) {
    auto assign = dyn_cast<puc::AssignOp>(user);
    if (assign && assign.getVar() == local) return true;
  }
  return false;
}

bool assignedBetween(Value var, Operation* from, Operation* to)
{
  if (!var) return true;
  for (Operation* user : var.getUsers()) {
    auto assign = dyn_cast<puc::AssignOp>(user);
    if (!assign || assign.getVar() != var) continue;
    if (from->isBeforeInBlock(assign) && assign->isBeforeInBlock(to)) return true;
  }
  return false;
}

bool resolveConstant(Value value, const llvm::DenseMap<Value, Attribute>& known, Attribute& out)
{
  Attribute attr;
  if (matchPattern(value, m_Constant(&attr))) {
    out = attr;
    return true;
  }
  auto it = known.find(value);
  if (it == known.end()) return false;
  out = it->second;
  return true;
}

void eraseLocalAndAssigns(puc::LocalOp local)
{
  SmallVector<puc::AssignOp> assigns;
  for (Operation* user : local.getResult().getUsers()) {
    auto assign = dyn_cast<puc::AssignOp>(user);
    if (assign && assign.getVar() == local.getResult()) assigns.push_back(assign);
  }
  for (puc::AssignOp assign : assigns) assign.erase();
  local.erase();
}

void foldConstantLocals(puc::FuncOp func)
{
  SmallVector<puc::LocalOp> locals;
  func.walk([&](puc::LocalOp local) { locals.push_back(local); });

  llvm::DenseSet<Value> blacklist;
  llvm::DenseMap<Value, Attribute> known;

  while (true) {
    known.clear();

    for (puc::LocalOp local : locals) {
      if (blacklist.contains(local.getResult()) || !local.getInit()) continue;
      Attribute attr;
      if (matchPattern(local.getInit(), m_Constant(&attr))) known[local.getResult()] = attr;
    }

    bool grew = true;
    while (grew) {
      grew = false;
      for (puc::LocalOp local : locals) {
        if (known.count(local.getResult()) || blacklist.contains(local.getResult())) continue;
        if (!local.getInit() || assignedAnywhere(local.getResult())) continue;
        Attribute attr;
        if (!resolveConstant(local.getInit(), known, attr)) continue;
        known[local.getResult()] = attr;
        grew                     = true;
      }
    }

    SmallVector<Value> doomed;
    for (auto& entry : known) {
      for (Operation* user : entry.first.getUsers()) {
        auto assign = dyn_cast<puc::AssignOp>(user);
        if (!assign || assign.getVar() != entry.first) continue;
        Attribute assigned;
        if (!resolveConstant(assign.getValue(), known, assigned) || assigned != entry.second) {
          doomed.push_back(entry.first);
          break;
        }
      }
    }
    if (doomed.empty()) break;
    for (Value value : doomed) blacklist.insert(value);
  }

  SmallVector<Value> folded;
  for (auto& entry : known) folded.push_back(entry.first);
  for (Value local : folded) {
    auto localOp = local.getDefiningOp<puc::LocalOp>();
    local.replaceAllUsesWith(localOp.getInit());
    eraseLocalAndAssigns(localOp);
  }
}

void simplifyAssigns(puc::FuncOp func)
{
  SmallVector<puc::AssignOp> assigns;
  func.walk([&](puc::AssignOp assign) { assigns.push_back(assign); });

  for (puc::AssignOp assign : assigns) {
    if (!assign->getBlock()) continue;
    if (assign.getVar() == assign.getValue()) {
      assign.erase();
      continue;
    }

    auto local = assign.getValue().getDefiningOp<puc::LocalOp>();
    if (!local || assignedAnywhere(local.getResult())) continue;
    if (assignedBetween(local.getInit(), local, assign)) continue;

    assign->setOperand(1, local.getInit());
  }
}

void eraseDeadLocals(puc::FuncOp func)
{
  SmallVector<puc::LocalOp> locals;
  func.walk([&](puc::LocalOp local) { locals.push_back(local); });
  for (puc::LocalOp local : llvm::reverse(locals))
    if (local.getResult().use_empty()) local.erase();
}

struct PucSimplifyAssignPass : public ::impl::PucSimplifyAssignBase<PucSimplifyAssignPass>
{
  using PucSimplifyAssignBase::PucSimplifyAssignBase;

  void getDependentDialects(DialectRegistry& registry) const override { registry.insert<puc::PucDialect>(); }

  void runOnOperation() override
  {
    ModuleOp moduleOp = getOperation();
    moduleOp.walk([&](puc::FuncOp funcOp) {
      if (funcOp.getBody().empty()) return;
      foldConstantLocals(funcOp);
      simplifyAssigns(funcOp);
      eraseDeadLocals(funcOp);
    });
  }
};

} // namespace

std::unique_ptr<OperationPass<ModuleOp>> triton::createPucSimplifyAssignPass()
{
  return std::make_unique<PucSimplifyAssignPass>();
}
