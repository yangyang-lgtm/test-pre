#include "privateuse/Transform/SinkConstantCallArgs/Passes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "privateuse/Dialect/PUMem/IR/PUMemDialect.h"

#include "llvm/ADT/BitVector.h"
#include "llvm/ADT/STLExtras.h"

#include <optional>

#define GEN_PASS_DECL_SINKCONSTANTCALLARGS
#include "privateuse/Transform/SinkConstantCallArgs/Passes.h.inc"

#define GEN_PASS_DEF_SINKCONSTANTCALLARGS
#include "privateuse/Transform/SinkConstantCallArgs/Passes.h.inc"

using namespace mlir;

namespace {

// One call site of a function. Both an IR call and a runtime call pass their
// arguments as the operand list in declaration order, so the folding below does
// not care which form the caller uses.
struct CallSite
{
  Operation* op;
  ValueRange arguments;
};

std::optional<CallSite> asCallSite(Operation* op)
{
  if (auto call = dyn_cast<func::CallOp>(op)) {
    if (call.getArgAttrsAttr()) return std::nullopt;
    return CallSite{op, call.getOperands()};
  }
  if (auto call = dyn_cast<pumem::RuntimeCallOp>(op)) return CallSite{op, call.getOperands()};
  return std::nullopt;
}

Operation* uniformConstantOperand(ArrayRef<CallSite> sites, unsigned index)
{
  Operation* constant = nullptr;
  for (const CallSite& site : sites) {
    Value operand  = site.arguments[index];
    Operation* def = operand.getDefiningOp();
    if (!def || !def->hasTrait<OpTrait::ConstantLike>()) return nullptr;
    if (!constant) {
      constant = def;
      continue;
    }
    if (def->getAttrs() != constant->getAttrs() || operand.getType() != constant->getResult(0).getType())
      return nullptr;
  }
  return constant;
}

struct SinkConstantCallArgsPass : public ::impl::SinkConstantCallArgsBase<SinkConstantCallArgsPass>
{
  using SinkConstantCallArgsBase::SinkConstantCallArgsBase;

  void runOnOperation() override
  {
    ModuleOp module = getOperation();

    module.walk([&](func::FuncOp func) {
      if (func.isExternal()) return;
      if (SymbolTable::getSymbolVisibility(func) != SymbolTable::Visibility::Private) return;

      std::optional<SymbolTable::UseRange> uses = SymbolTable::getSymbolUses(func, module);
      if (!uses || uses->empty()) return;

      SmallVector<CallSite> sites;
      for (SymbolTable::SymbolUse use : *uses) {
        std::optional<CallSite> site = asCallSite(use.getUser());
        if (!site) return;
        sites.push_back(*site);
      }

      unsigned numInputs = func.getNumArguments();
      llvm::BitVector erased(numInputs);
      SmallVector<Operation*> constants(numInputs);
      for (unsigned i = 0; i < numInputs; ++i) {
        if (Operation* constant = uniformConstantOperand(sites, i)) {
          constants[i] = constant;
          erased.set(i);
        }
      }
      if (erased.none()) return;

      OpBuilder builder = OpBuilder::atBlockBegin(&func.getFunctionBody().front());
      for (unsigned i = 0; i < numInputs; ++i) {
        if (!erased.test(i)) continue;
        func.getArgument(i).replaceAllUsesWith(builder.clone(*constants[i])->getResult(0));
      }
      if (failed(func.eraseArguments(erased))) return;

      for (const CallSite& site : sites) {
        SmallVector<Value> operands;
        for (unsigned i = 0; i < numInputs; ++i)
          if (!erased.test(i)) operands.push_back(site.arguments[i]);
        site.op->setOperands(operands);
      }
    });
  }
};

} // namespace

std::unique_ptr<OperationPass<ModuleOp>> triton::createSinkConstantCallArgsPass()
{
  return std::make_unique<SinkConstantCallArgsPass>();
}
