#include "privateuse/Transform/SinkConstantCallArgs/Passes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/FunctionInterfaces.h"

#include "llvm/ADT/BitVector.h"
#include "llvm/ADT/STLExtras.h"

#include <optional>

#define GEN_PASS_DECL_SINKCONSTANTCALLARGS
#include "privateuse/Transform/SinkConstantCallArgs/Passes.h.inc"

#define GEN_PASS_DEF_SINKCONSTANTCALLARGS
#include "privateuse/Transform/SinkConstantCallArgs/Passes.h.inc"

using namespace mlir;

namespace {

Operation* uniformConstantOperand(ArrayRef<func::CallOp> calls, unsigned index)
{
  Operation* constant = nullptr;
  for (func::CallOp call : calls) {
    Value operand  = call.getOperand(index);
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

      SmallVector<func::CallOp> calls;
      for (SymbolTable::SymbolUse use : *uses) {
        auto call = dyn_cast<func::CallOp>(use.getUser());
        if (!call || call.getArgAttrsAttr()) return;
        calls.push_back(call);
      }

      unsigned numInputs = func.getNumArguments();
      llvm::BitVector erased(numInputs);
      SmallVector<Operation*> constants(numInputs);
      for (unsigned i = 0; i < numInputs; ++i) {
        if (Operation* constant = uniformConstantOperand(calls, i)) {
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

      for (func::CallOp call : calls) {
        SmallVector<Value> operands;
        for (unsigned i = 0; i < numInputs; ++i)
          if (!erased.test(i)) operands.push_back(call.getOperand(i));
        call.getOperandsMutable().assign(operands);
      }
    });
  }
};

} // namespace

std::unique_ptr<OperationPass<ModuleOp>> triton::createSinkConstantCallArgsPass()
{
  return std::make_unique<SinkConstantCallArgsPass>();
}
