#include "privateuse/Transform/OutlineLinalgOps/Passes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/FunctionInterfaces.h"

#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/Twine.h"

#include <algorithm>
#include <string>

#define GEN_PASS_DECL_OUTLINELINALGOPS
#include "privateuse/Transform/OutlineLinalgOps/Passes.h.inc"

#define GEN_PASS_DEF_OUTLINELINALGOPS
#include "privateuse/Transform/OutlineLinalgOps/Passes.h.inc"

using namespace mlir;
using namespace mlir::triton;

namespace {

bool isBufferLinalgOp(Operation* op)
{
  auto linalgOp = dyn_cast<linalg::LinalgOp>(op);
  return linalgOp && linalgOp.hasPureBufferSemantics();
}

std::string buildFunctionName(Operation* op, unsigned count)
{
  std::string name  = "outlined_";
  std::string outer = op->getName().getStringRef().str();
  std::replace(outer.begin(), outer.end(), '.', '_');
  name += outer;

  llvm::SmallDenseSet<StringRef, 4> seen;
  op->walk([&](Operation* inner) {
    if (inner == op || inner->hasTrait<OpTrait::IsTerminator>()) return WalkResult::advance();
    StringRef innerName = inner->getName().stripDialect();
    if (seen.insert(innerName).second) name += ("_" + innerName).str();
    return WalkResult::advance();
  });
  return name + ("_" + Twine(count)).str();
}

SetVector<Value> collectExternalValues(Operation* op)
{
  SetVector<Value> external;
  for (Region& region : op->getRegions())
    region.walk([&](Operation* inner) {
      for (Value operand : inner->getOperands()) {
        if (auto blockArg = dyn_cast<BlockArgument>(operand)) {
          if (op->isAncestor(blockArg.getOwner()->getParentOp())) continue;
          external.insert(operand);
          continue;
        }
        if (Operation* def = operand.getDefiningOp())
          if (!op->isAncestor(def)) external.insert(operand);
      }
    });
  return external;
}

struct OutlineLinalgOpsPass : public ::impl::OutlineLinalgOpsBase<OutlineLinalgOpsPass>
{
  using OutlineLinalgOpsBase::OutlineLinalgOpsBase;

  void getDependentDialects(DialectRegistry& registry) const override
  {
    // clang-format off
    registry.insert<
        func::FuncDialect,
        linalg::LinalgDialect,
        memref::MemRefDialect
    >();
    // clang-format on
  }

  void runOnOperation() override
  {
    ModuleOp module        = getOperation();
    MLIRContext* context   = &getContext();
    unsigned outlinedCount = 0;

    SmallVector<Operation*> outlineUnits;
    module.walk([&](Operation* op) {
      if (!isBufferLinalgOp(op)) return;
      for (Operation* parent = op->getParentOp(); parent; parent = parent->getParentOp())
        if (isBufferLinalgOp(parent)) return;
      outlineUnits.push_back(op);
    });

    for (Operation* op : outlineUnits) {
      if (!op->getParentOfType<FunctionOpInterface>()) continue;

      SmallVector<Value> boundary(op->getOperands());
      for (Value value : collectExternalValues(op)) boundary.push_back(value);

      OpBuilder builder(context);
      builder.setInsertionPointToStart(module.getBody());
      auto outlined = func::FuncOp::create(
        builder, op->getLoc(), buildFunctionName(op, outlinedCount),
        builder.getFunctionType(
          llvm::map_to_vector(boundary, [](Value value) { return value.getType(); }), op->getResultTypes()
        )
      );
      SymbolTable::setSymbolVisibility(outlined, SymbolTable::Visibility::Private);

      Block* entry = outlined.addEntryBlock();
      builder.setInsertionPointToStart(entry);
      IRMapping mapping;
      for (auto [value, argument] : llvm::zip_equal(boundary, entry->getArguments())) mapping.map(value, argument);
      Operation* cloned = builder.clone(*op, mapping);

      builder.setInsertionPointToEnd(entry);
      func::ReturnOp::create(builder, op->getLoc(), cloned->getResults());

      builder.setInsertionPoint(op);
      auto call = func::CallOp::create(
        builder, op->getLoc(), FlatSymbolRefAttr::get(context, outlined.getSymName()), op->getResultTypes(), boundary
      );
      op->replaceAllUsesWith(call.getResults());
      op->erase();
      ++outlinedCount;
    }
  }
};

} // namespace

std::unique_ptr<OperationPass<ModuleOp>> triton::createOutlineLinalgOpsPass()
{
  return std::make_unique<OutlineLinalgOpsPass>();
}
