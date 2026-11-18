#include "privateuse/Transform/PidLoop/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "privateuse/Dialect/PUMem/IR/PUMemDialect.h"
#include "privateuse/Utils/Constants.h"
#include "triton/Dialect/Triton/IR/Dialect.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"

#include <array>

#define GEN_PASS_DECL_PIDLOOP
#include "privateuse/Transform/PidLoop/Passes.h.inc"

#define GEN_PASS_DEF_PIDLOOP
#include "privateuse/Transform/PidLoop/Passes.h.inc"

using namespace mlir;
using namespace mlir::triton;

namespace {

static void moveBeforeTerminator(Block* block, ArrayRef<Operation*> ops)
{
  Operation* terminator = block->getTerminator();
  for (Operation* op : ops) op->moveBefore(terminator);
}

static LogicalResult replaceProgramIds(
  OpBuilder& builder, Location loc, Value linear, ArrayRef<int64_t> grid, ArrayRef<SmallVector<GetProgramIdOp>> pidOps
)
{
  auto cst   = [&](int64_t value) -> Value { return arith::ConstantIndexOp::create(builder, loc, value); };
  auto divBy = [&](Value value, int64_t divisor) -> Value {
    return divisor == 1 ? value : arith::DivSIOp::create(builder, loc, value, cst(divisor));
  };
  auto modBy = [&](Value value, int64_t modulus) -> Value {
    return modulus == 1 ? cst(0) : arith::RemSIOp::create(builder, loc, value, cst(modulus));
  };

  std::array<Value, kNumAxes> pidValues;
  Value quotient = divBy(linear, grid[0]);
  pidValues[0]   = grid[1] == 1 && grid[2] == 1 ? linear : modBy(linear, grid[0]);
  pidValues[1]   = grid[2] == 1 ? quotient : modBy(quotient, grid[1]);
  pidValues[2]   = divBy(quotient, grid[1]);
  if (grid[1] == 1) pidValues[1] = cst(0);
  if (grid[2] == 1) pidValues[2] = cst(0);

  for (int axis = 0; axis < kNumAxes; ++axis) {
    for (GetProgramIdOp op : pidOps[axis]) {
      Value value = pidValues[axis];
      if (op.getType() != value.getType()) value = arith::IndexCastOp::create(builder, loc, op.getType(), value);
      op.replaceAllUsesWith(value);
      op.erase();
    }
  }
  return success();
}

struct PidLoopPass : public ::impl::PidLoopBase<PidLoopPass>
{
  using PidLoopBase::PidLoopBase;

  PidLoopPass(int64_t numPids) { this->numPids = numPids; }

  void getDependentDialects(DialectRegistry& registry) const override
  {
    // clang-format off
    registry.insert<
      arith::ArithDialect,
      scf::SCFDialect,
      pumem::PUMemDialect,
      triton::TritonDialect
    >();
    // clang-format on
  }

  void runOnOperation() override
  {
    ModuleOp module = getOperation();

    int64_t threads = numPids.getValue();
    if (threads <= 0) {
      module.emitError("pid-loop: num-pids must be positive, got ") << threads;
      return signalPassFailure();
    }

    auto gridAttr = module->getAttrOfType<DenseI64ArrayAttr>(kGridAttr);
    if (!gridAttr) {
      module.emitError("pid-loop: missing module attribute ") << kGridAttr;
      return signalPassFailure();
    }

    ArrayRef<int64_t> dims = gridAttr.asArrayRef();
    if (dims.size() > kNumAxes) {
      module.emitError("pid-loop: grid has more than ") << kNumAxes << " dimensions: " << gridAttr;
      return signalPassFailure();
    }

    std::array<int64_t, kNumAxes> grid = {1, 1, 1};
    for (auto [axis, dim] : llvm::enumerate(dims)) {
      if (dim <= 0) {
        module.emitError("pid-loop: grid dimensions must be positive, got ") << gridAttr;
        return signalPassFailure();
      }
      grid[axis] = dim;
    }

    for (Operation& op : module.getBody()->getOperations()) {
      auto func = dyn_cast<FunctionOpInterface>(&op);
      if (!func || func.isExternal()) continue;
      if (SymbolTable::getSymbolVisibility(func.getOperation()) != SymbolTable::Visibility::Public) continue;
      if (failed(rewriteKernel(func, grid, threads))) return signalPassFailure();
    }
  }

private:
  LogicalResult rewriteKernel(FunctionOpInterface func, ArrayRef<int64_t> grid, int64_t threads)
  {
    if (!func.getFunctionBody().hasOneBlock() || !func.getResultTypes().empty())
      return func.emitError("pid-loop: expected a single-block kernel without results");

    Block& entry = func.getFunctionBody().front();
    Location loc = func.getLoc();

    std::array<SmallVector<GetProgramIdOp>, kNumAxes> pidOps;
    func.walk([&](GetProgramIdOp op) { pidOps[op.getAxisAsInt()].push_back(op); });

    SmallVector<Operation*> bodyOps;
    for (Operation& op : entry.without_terminator()) bodyOps.push_back(&op);

    int64_t totalPids = grid[0] * grid[1] * grid[2];

    OpBuilder builder(func.getContext());
    builder.setInsertionPointToStart(&entry);
    Value tid = pumem::GetPhysicalIdOp::create(builder, loc, builder.getIndexType());

    Value linear          = tid;
    OpBuilder bodyBuilder = builder;
    if (totalPids > threads) {
      Value total = arith::ConstantIndexOp::create(builder, loc, totalPids);
      Value step  = arith::ConstantIndexOp::create(builder, loc, threads);
      auto loop   = scf::ForOp::create(builder, loc, tid, total, step);
      linear      = loop.getInductionVar();
      moveBeforeTerminator(loop.getBody(), bodyOps);
      bodyBuilder.setInsertionPointToStart(loop.getBody());
    } else if (totalPids < threads) {
      Value total = arith::ConstantIndexOp::create(builder, loc, totalPids);
      Value guard = arith::CmpIOp::create(builder, loc, arith::CmpIPredicate::slt, tid, total);
      auto ifOp   = scf::IfOp::create(builder, loc, guard, /*withElseRegion=*/false);
      moveBeforeTerminator(ifOp.thenBlock(), bodyOps);
      bodyBuilder.setInsertionPointToStart(ifOp.thenBlock());
    }

    return replaceProgramIds(bodyBuilder, loc, linear, grid, pidOps);
  }
};

} // namespace

std::unique_ptr<OperationPass<ModuleOp>> triton::createPidLoopPass(int64_t numPids)
{
  return std::make_unique<PidLoopPass>(numPids);
}
