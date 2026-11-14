#include "privateuse/Transform/PucLaunchKernel/Passes.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "privateuse/Dialect/PUMem/IR/PUMemDialect.h"
#include "privateuse/Dialect/Puc/IR/PucDialect.h"
#include "privateuse/Utils/Constants.h"

#include "llvm/ADT/StringSet.h"

#define GEN_PASS_DECL_PUCLAUNCHKERNEL
#include "privateuse/Transform/PucLaunchKernel/Passes.h.inc"

#define GEN_PASS_DEF_PUCLAUNCHKERNEL
#include "privateuse/Transform/PucLaunchKernel/Passes.h.inc"

using namespace mlir;

namespace {

bool isLauncher(puc::FuncOp func)
{
  return func.getName().ends_with("_launcher");
}

llvm::StringSet<> collectCallees(ModuleOp module)
{
  llvm::StringSet<> callees;
  module.walk([&](puc::CallOp call) { callees.insert(call.getCallee()); });
  return callees;
}

Type withTrivialSpace(Type type, MLIRContext* ctx)
{
  auto ptr = dyn_cast<pumem::PtrType>(type);
  if (!ptr) return type;
  return pumem::PtrType::get(ctx, ptr.getElemType(), triton::kTrivialSpace);
}

FunctionType launcherType(FunctionType kernelType, MLIRContext* ctx)
{
  SmallVector<Type> params;
  params.reserve(kernelType.getNumInputs());
  for (Type type : kernelType.getInputs()) params.push_back(withTrivialSpace(type, ctx));
  return FunctionType::get(ctx, params, kernelType.getResults());
}

void addLauncher(OpBuilder& builder, puc::FuncOp kernel, StringRef launcherAttr, StringRef kernelAttr)
{
  MLIRContext* ctx = kernel.getContext();

  builder.setInsertionPointAfter(kernel);
  auto launcher = puc::FuncOp::create(
    builder, kernel.getLoc(), kernel.getName().str() + "_launcher", launcherType(kernel.getFunctionType(), ctx)
  );
  launcher->setAttr(triton::kOutlinedAttr, builder.getStringAttr(launcherAttr));
  kernel->setAttr(triton::kOutlinedAttr, builder.getStringAttr(kernelAttr));

  Block* entry = launcher.addEntryBlock();
  builder.setInsertionPointToStart(entry);
  SmallVector<Value> args(entry->getArguments().begin(), entry->getArguments().end());
  puc::LaunchOp::create(builder, kernel.getLoc(), FlatSymbolRefAttr::get(ctx, kernel.getName()), args);
  puc::ReturnOp::create(builder, kernel.getLoc());
}

struct PucLaunchKernelPass : public ::impl::PucLaunchKernelBase<PucLaunchKernelPass>
{
  using PucLaunchKernelBase::PucLaunchKernelBase;

  PucLaunchKernelPass(std::string launcherAttr, std::string kernelAttr)
  {
    this->launcherAttrValue = std::move(launcherAttr);
    this->kernelAttrValue   = std::move(kernelAttr);
  }

  void getDependentDialects(DialectRegistry& registry) const override { registry.insert<puc::PucDialect>(); }

  void runOnOperation() override
  {
    ModuleOp module           = getOperation();
    llvm::StringSet<> callees = collectCallees(module);

    SmallVector<puc::FuncOp> kernels;
    for (puc::FuncOp func : module.getOps<puc::FuncOp>()) {
      if (func.isExternal()) continue;
      if (isLauncher(func)) continue;
      if (callees.contains(func.getName())) continue;
      kernels.push_back(func);
    }

    OpBuilder builder(module.getContext());
    for (puc::FuncOp kernel : kernels) addLauncher(builder, kernel, launcherAttrValue, kernelAttrValue);
  }
};

} // namespace

std::unique_ptr<OperationPass<ModuleOp>>
triton::createPucLaunchKernelPass(std::string launcherAttrValue, std::string kernelAttrValue)
{
  return std::make_unique<PucLaunchKernelPass>(std::move(launcherAttrValue), std::move(kernelAttrValue));
}
