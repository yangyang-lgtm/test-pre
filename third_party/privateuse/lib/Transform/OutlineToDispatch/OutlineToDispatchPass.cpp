#include "privateuse/Transform/OutlineToDispatch/Passes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "privateuse/Dialect/PUMem/IR/PUMemDialect.h"
#include "privateuse/Dialect/PuStream/IR/PuStreamDialect.h"
#include "privateuse/Utils/Constants.h"

#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/Twine.h"

#include <algorithm>
#include <string>

#define GEN_PASS_DECL_OUTLINETODISPATCH
#include "privateuse/Transform/OutlineToDispatch/Passes.h.inc"

#define GEN_PASS_DEF_OUTLINETODISPATCH
#include "privateuse/Transform/OutlineToDispatch/Passes.h.inc"

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
  for (Region& region : op->getRegions()) {
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
  }
  return external;
}

bool collectEffectIndices(Operation* op, SmallVectorImpl<int64_t>& reads, SmallVectorImpl<int64_t>& writes)
{
  auto interface = dyn_cast<MemoryEffectOpInterface>(op);
  if (!interface) return false;

  SmallVector<SideEffects::EffectInstance<MemoryEffects::Effect>> effects;
  interface.getEffects(effects);

  auto record = [&](Value value, SmallVectorImpl<int64_t>& indices) {
    for (auto [index, operand] : llvm::enumerate(op->getOperands()))
      if (operand == value) indices.push_back(index);
  };

  for (const auto& effect : effects) {
    Value value = effect.getValue();
    if (!value) return false;
    if (isa<MemoryEffects::Write>(effect.getEffect()))
      record(value, writes);
    else if (isa<MemoryEffects::Read>(effect.getEffect()))
      record(value, reads);
    else
      return false;
  }
  return true;
}

struct OutlineToDispatchPass : public ::impl::OutlineToDispatchBase<OutlineToDispatchPass>
{
  using OutlineToDispatchBase::OutlineToDispatchBase;

  OutlineToDispatchPass(std::string outlinedAttrValue) { this->outlinedAttrValue = std::move(outlinedAttrValue); }

  void getDependentDialects(DialectRegistry& registry) const override
  {
    // clang-format off
    registry.insert<
      func::FuncDialect,
      linalg::LinalgDialect,
      memref::MemRefDialect,
      pumem::PUMemDialect,
      pustream::PuStreamDialect
    >();
    // clang-format on
  }

  void createOutlinedDispatch(
    Operation* op, ArrayRef<Value> boundary, ArrayRef<int64_t> reads, ArrayRef<int64_t> writes, unsigned count
  )
  {
    ModuleOp module = getOperation();
    OpBuilder builder(&getContext());
    builder.setInsertionPointToStart(module.getBody());
    auto outlined = func::FuncOp::create(
      builder, op->getLoc(), buildFunctionName(op, count),
      builder.getFunctionType(
        llvm::map_to_vector(boundary, [](Value value) { return value.getType(); }), op->getResultTypes()
      )
    );
    SymbolTable::setSymbolVisibility(outlined, SymbolTable::Visibility::Private);
    if (!outlinedAttrValue.empty())
      outlined->setAttr(triton::kOutlinedAttr, builder.getStringAttr(outlinedAttrValue));

    Block* entry = outlined.addEntryBlock();
    builder.setInsertionPointToStart(entry);
    IRMapping mapping;
    for (auto [value, argument] : llvm::zip_equal(boundary, entry->getArguments())) mapping.map(value, argument);
    Operation* cloned = builder.clone(*op, mapping);

    builder.setInsertionPointToEnd(entry);
    func::ReturnOp::create(builder, op->getLoc(), cloned->getResults());

    builder.setInsertionPoint(op);
    pustream::AsyncDispatchOp::create(
      builder, op->getLoc(), pustream::TimepointType::get(&getContext()), outlined.getSymName(),
      /*templateArgs=*/nullptr, boundary, /*awaits=*/ValueRange{}, reads, writes
    );
    op->erase();
  }

  void runOnOperation() override
  {
    SmallVector<Operation*> outlineUnits;
    getOperation().walk([&](Operation* op) {
      if (isBufferLinalgOp(op)) outlineUnits.push_back(op);
    });

    unsigned count = 0;
    for (Operation* op : outlineUnits) {
      if (!op->getParentOfType<FunctionOpInterface>()) continue;
      if (op->getNumResults() != 0) {
        op->emitError("outline-to-dispatch: expected a buffer linalg op without results");
        signalPassFailure();
        return;
      }

      SmallVector<int64_t> reads, writes;
      if (!collectEffectIndices(op, reads, writes)) {
        op->emitError("outline-to-dispatch: the outlined op does not describe its memory effects");
        signalPassFailure();
        return;
      }

      SmallVector<Value> boundary(op->getOperands());
      for (Value value : collectExternalValues(op)) {
        if (!isa<IntegerType, FloatType, IndexType>(value.getType())) {
          op->emitError("outline-to-dispatch: captured value must be a scalar, got ") << value.getType();
          signalPassFailure();
          return;
        }
        boundary.push_back(value);
      }

      createOutlinedDispatch(op, boundary, reads, writes, count++);
    }
  }
};

} // namespace

std::unique_ptr<OperationPass<ModuleOp>> triton::createOutlineToDispatchPass(std::string outlinedAttrValue)
{
  return std::make_unique<OutlineToDispatchPass>(std::move(outlinedAttrValue));
}
