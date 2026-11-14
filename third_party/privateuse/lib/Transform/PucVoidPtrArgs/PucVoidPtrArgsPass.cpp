#include "privateuse/Transform/PucVoidPtrArgs/PucVoidPtrArgs.h"

#include "llvm/ADT/STLExtras.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "privateuse/Dialect/PUMem/IR/PUMemDialect.h"
#include "privateuse/Dialect/Puc/IR/PucDialect.h"

#define DEBUG_TYPE "puc-void-ptr-args"

using namespace mlir;
using namespace mlir::triton;

#define GEN_PASS_DECL_PUCVOIDPTRARGS
#include "privateuse/Transform/PucVoidPtrArgs/Passes.h.inc"

#define GEN_PASS_DEF_PUCVOIDPTRARGS
#include "privateuse/Transform/PucVoidPtrArgs/Passes.h.inc"

namespace {

Type toVoidPtr(Type type)
{
  auto ptr = dyn_cast<pumem::PtrType>(type);
  if (!ptr || isa<pumem::VoidType>(ptr.getElemType())) return type;
  return pumem::PtrType::get(ptr.getContext(), pumem::VoidType::get(ptr.getContext()), ptr.getSpace());
}

void voidifyArgs(puc::FuncOp func)
{
  if (func.isExternal()) return;

  FunctionType oldType = func.getFunctionType();
  SmallVector<Type> inputs(oldType.getInputs());
  bool changed = false;
  for (Type& input : inputs) {
    Type erased  = toVoidPtr(input);
    changed     |= erased != input;
    input        = erased;
  }
  if (!changed) return;

  Block& entry = func.getBody().front();
  OpBuilder builder(func.getContext());
  builder.setInsertionPointToStart(&entry);
  for (auto [index, arg] : llvm::enumerate(entry.getArguments())) {
    Type oldArgType = arg.getType();
    if (oldArgType == inputs[index]) continue;
    arg.setType(inputs[index]);
    auto cast = puc::PointCastOp::create(builder, func.getLoc(), oldArgType, arg);
    arg.replaceAllUsesExcept(cast.getResult(), cast);
    builder.setInsertionPointAfter(cast);
  }
  func.setType(builder.getFunctionType(inputs, oldType.getResults()));
}

struct PucVoidPtrArgsPass : public ::impl::PucVoidPtrArgsBase<PucVoidPtrArgsPass>
{
  using PucVoidPtrArgsBase::PucVoidPtrArgsBase;

  void getDependentDialects(DialectRegistry& registry) const override { registry.insert<puc::PucDialect>(); }

  void runOnOperation() override
  {
    getOperation().walk([](puc::FuncOp func) { voidifyArgs(func); });
  }
};

} // namespace

std::unique_ptr<OperationPass<ModuleOp>> triton::createPucVoidPtrArgsPass()
{
  return std::make_unique<PucVoidPtrArgsPass>();
}
