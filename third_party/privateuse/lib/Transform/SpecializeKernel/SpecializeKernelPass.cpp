#include "privateuse/Transform/SpecializeKernel/SpecializeKernel.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "privateuse/Utils/Constants.h"
#include "triton/Dialect/Triton/IR/Dialect.h"

#define GEN_PASS_DECL_SPECIALIZEKERNEL
#include "privateuse/Transform/SpecializeKernel/Passes.h.inc"

#define GEN_PASS_DEF_SPECIALIZEKERNEL
#include "privateuse/Transform/SpecializeKernel/Passes.h.inc"

using namespace mlir;
using namespace triton;

namespace {

struct SpecializeKernelPass : public ::impl::SpecializeKernelBase<SpecializeKernelPass>
{
  using SpecializeKernelBase::SpecializeKernelBase;

  SpecializeKernelPass(std::vector<SpecializeConstant> constants, std::vector<int64_t> grid)
    : constants(std::move(constants)), grid(std::move(grid))
  {
  }

  void runOnOperation() override
  {
    ModuleOp module = getOperation();
    module->setAttr(kGridAttr, DenseI64ArrayAttr::get(&getContext(), grid));

    for (Operation& op : module.getBody()->getOperations()) {
      auto func = dyn_cast<FunctionOpInterface>(&op);
      if (!func || func.isExternal() || !func.getFunctionBody().hasOneBlock()) continue;
      if (SymbolTable::getSymbolVisibility(func.getOperation()) != SymbolTable::Visibility::Public) continue;
      if (failed(specializeArgs(func)) || failed(replaceNumPrograms(func))) {
        signalPassFailure();
        return;
      }
    }
  }

private:
  LogicalResult specializeArgs(FunctionOpInterface func)
  {
    Block& body = func.getFunctionBody().front();
    OpBuilder builder(func.getContext());
    llvm::BitVector erased(body.getNumArguments());

    for (const SpecializeConstant& constant : constants) {
      int64_t index = constant.first;
      if (index < 0 || index >= static_cast<int64_t>(body.getNumArguments()))
        return func.emitError("specialize: constant index ") << index << " is out of range";
      if (erased.test(index)) return func.emitError("specialize: duplicate constant index ") << index;

      BlockArgument arg = body.getArgument(index);
      Type type         = arg.getType();
      TypedAttr attr;
      if (auto intType = dyn_cast<IntegerType>(type)) {
        if (!std::holds_alternative<int64_t>(constant.second))
          return func.emitError("specialize: integer argument ") << index << " needs an integer";
        attr = IntegerAttr::get(intType, std::get<int64_t>(constant.second));
      } else if (isa<IndexType>(type)) {
        if (!std::holds_alternative<int64_t>(constant.second))
          return func.emitError("specialize: index argument ") << index << " needs an integer";
        attr = IntegerAttr::get(IndexType::get(&getContext()), std::get<int64_t>(constant.second));
      } else if (auto floatType = dyn_cast<FloatType>(type)) {
        if (!std::holds_alternative<double>(constant.second))
          return func.emitError("specialize: float argument ") << index << " needs a float";
        attr = FloatAttr::get(floatType, std::get<double>(constant.second));
      } else {
        return func.emitError("specialize: cannot fold argument ") << index << " of type " << type;
      }

      builder.setInsertionPointToStart(&body);
      auto cst = arith::ConstantOp::create(builder, func.getLoc(), attr);
      arg.replaceAllUsesWith(cst.getResult());
      erased.set(index);
    }

    SmallVector<Type> keptInputs;
    for (BlockArgument arg : body.getArguments()) {
      if (!erased.test(arg.getArgNumber())) keptInputs.push_back(arg.getType());
    }

    SmallVector<DictionaryAttr> keptAttrs;
    if (func.getAllArgAttrs()) func.getAllArgAttrs(keptAttrs);
    keptAttrs.resize(body.getNumArguments(), DictionaryAttr());
    SmallVector<DictionaryAttr> remaining;
    for (unsigned i = 0; i < keptAttrs.size(); ++i) {
      if (!erased.test(i)) remaining.push_back(keptAttrs[i]);
    }

    body.eraseArguments(erased);
    func.setType(builder.getFunctionType(keptInputs, func.getResultTypes()));
    func.setAllArgAttrs(remaining);
    return success();
  }

  LogicalResult replaceNumPrograms(FunctionOpInterface func)
  {
    SmallVector<triton::GetNumProgramsOp> ops;
    func.walk([&](triton::GetNumProgramsOp op) { ops.push_back(op); });

    OpBuilder builder(func.getContext());
    for (triton::GetNumProgramsOp op : ops) {
      int64_t axis = static_cast<int64_t>(op.getAxis());
      if (axis < 0 || axis >= static_cast<int64_t>(grid.size()))
        return op.emitError("specialize: program id axis out of range");

      builder.setInsertionPoint(op);
      Type type = op.getResult().getType();
      auto cst  = arith::ConstantOp::create(builder, op.getLoc(), builder.getIntegerAttr(type, grid[axis]));
      op.replaceAllUsesWith(cst.getResult());
      op.erase();
    }
    return success();
  }

  std::vector<SpecializeConstant> constants;
  std::vector<int64_t> grid;
};

} // namespace

std::unique_ptr<OperationPass<ModuleOp>>
triton::createSpecializeKernelPass(std::vector<SpecializeConstant> constants, std::vector<int64_t> grid)
{
  return std::make_unique<SpecializeKernelPass>(std::move(constants), std::move(grid));
}
