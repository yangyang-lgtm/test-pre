#include "privateuse/Transform/PucRepeat/PucRepeat.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "privateuse/Dialect/Puc/IR/PucDialect.h"

#define GEN_PASS_DECL_PUCREPEAT
#include "privateuse/Transform/PucRepeat/Passes.h.inc"

#define GEN_PASS_DEF_PUCREPEAT
#include "privateuse/Transform/PucRepeat/Passes.h.inc"

using namespace mlir;
using namespace mlir::triton;

namespace {

void repeatLauncher(puc::FuncOp launcher, int64_t iterations)
{
  Block &entry = launcher.getBody().front();

  puc::LaunchOp launch;
  for (Operation &op : entry)
    if (auto candidate = dyn_cast<puc::LaunchOp>(op)) launch = candidate;
  if (!launch) return;

  Location loc = launcher.getLoc();
  OpBuilder builder(launcher.getContext());
  builder.setInsertionPointToStart(&entry);

  Type index  = builder.getIndexType();
  Value lower = puc::ConstantOp::create(builder, loc, index, builder.getIndexAttr(0));
  Value upper = puc::ConstantOp::create(builder, loc, index, builder.getIndexAttr(iterations));
  Value step  = puc::ConstantOp::create(builder, loc, index, builder.getIndexAttr(1));

  auto timer = puc::TimerType::get(builder.getContext(), launcher.getName(), iterations);
  puc::LocalOp::create(builder, loc, timer, Value{});

  auto loop = puc::ForOp::create(builder, loc, lower, upper, step);
  launch->moveBefore(loop.getBody()->getTerminator());
}

struct PucRepeatPass : public ::impl::PucRepeatBase<PucRepeatPass>
{
  using PucRepeatBase::PucRepeatBase;

  PucRepeatPass(int64_t iterations) { this->iterations = iterations; }

  void getDependentDialects(DialectRegistry &registry) const override { registry.insert<puc::PucDialect>(); }

  void runOnOperation() override
  {
    ModuleOp module = getOperation();
    for (puc::FuncOp func : module.getOps<puc::FuncOp>()) {
      if (func.isExternal()) continue;
      repeatLauncher(func, iterations);
    }
  }
};

} // namespace

std::unique_ptr<OperationPass<ModuleOp>> triton::createPucRepeatPass(int64_t iterations)
{
  return std::make_unique<PucRepeatPass>(iterations);
}
