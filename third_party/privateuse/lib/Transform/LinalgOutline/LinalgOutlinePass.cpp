#include "privateuse/Transform/LinalgOutline/Passes.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/SymbolTable.h"
#include "privateuse/Dialect/PUMem/IR/PUMemDialect.h"
#include "privateuse/Utils/Constants.h"

#include <algorithm>
#include <string>

#define GEN_PASS_DECL_LINALGOUTLINE
#include "privateuse/Transform/LinalgOutline/Passes.h.inc"

#define GEN_PASS_DEF_LINALGOUTLINE
#include "privateuse/Transform/LinalgOutline/Passes.h.inc"

using namespace mlir;
using namespace mlir::triton;

namespace {

bool isComputable(linalg::LinalgOp op)
{
  if (!op.hasPureBufferSemantics() || op->getNumResults() != 0) return false;
  if (auto function = op->getParentOfType<func::FuncOp>())
    if (function->hasAttr(kOutlinedAttr)) return false;
  return true;
}

std::string buildName(linalg::LinalgOp op, unsigned index)
{
  std::string name  = "outlined_" + std::to_string(index) + "_";
  std::string inner = op->getName().getStringRef().str();
  std::replace(inner.begin(), inner.end(), '.', '_');
  return name + inner;
}

struct LinalgOutlinePass : public ::impl::LinalgOutlineBase<LinalgOutlinePass>
{
  using LinalgOutlineBase::LinalgOutlineBase;

  void getDependentDialects(DialectRegistry& registry) const override
  {
    registry.insert<func::FuncDialect, pumem::PUMemDialect>();
  }

  void runOnOperation() override
  {
    ModuleOp module = getOperation();

    SmallVector<linalg::LinalgOp> units;
    module.walk([&](linalg::LinalgOp op) {
      if (isComputable(op)) units.push_back(op);
    });

    unsigned index = 0;
    for (linalg::LinalgOp op : units) outline(op, index++);
  }

private:
  void outline(linalg::LinalgOp op, unsigned index)
  {
    ModuleOp module = getOperation();
    Location loc    = op.getLoc();

    SmallVector<Value> boundary(op->getOperands().begin(), op->getOperands().end());
    SmallVector<Type> types;
    for (Value value : boundary) types.push_back(value.getType());

    OpBuilder builder(module.getBodyRegion());
    builder.setInsertionPointToStart(module.getBody());
    auto function =
      func::FuncOp::create(builder, loc, buildName(op, index), builder.getFunctionType(types, TypeRange{}));
    SymbolTable::setSymbolVisibility(function, SymbolTable::Visibility::Private);
    function->setAttr(kOutlinedAttr, builder.getStringAttr(outlinedAttr));

    Block* entry = function.addEntryBlock();
    IRMapping mapping;
    for (auto [value, argument] : llvm::zip_equal(boundary, entry->getArguments())) mapping.map(value, argument);

    OpBuilder bodyBuilder = OpBuilder::atBlockEnd(entry);
    bodyBuilder.clone(*op.getOperation(), mapping);
    func::ReturnOp::create(bodyBuilder, loc);

    OpBuilder callBuilder(op);
    pumem::RuntimeCallOp::create(
      callBuilder, loc, TypeRange{}, FlatSymbolRefAttr::get(&getContext(), function.getSymName()), ArrayAttr{}, boundary
    );
    op->erase();
  }
};

} // namespace

std::unique_ptr<OperationPass<ModuleOp>> triton::createLinalgOutlinePass()
{
  return std::make_unique<LinalgOutlinePass>();
}

std::unique_ptr<OperationPass<ModuleOp>> triton::createLinalgOutlinePass(std::string outlinedAttr)
{
  LinalgOutlineOptions options;
  options.outlinedAttr = std::move(outlinedAttr);
  return std::make_unique<LinalgOutlinePass>(options);
}
