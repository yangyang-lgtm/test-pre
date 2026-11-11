#ifndef PRIVATEUSE_CONVERSION_PASSES_TD
#define PRIVATEUSE_CONVERSION_PASSES_TD

include "mlir/Pass/PassBase.td"

def AnalyzeAccess : Pass<"privateuse-analyze-access", "mlir::ModuleOp"> {
  let summary = "Report canonical memory access views and masks";
  let description = [{
    Runs the access analysis over every tt.load / tt.store in the module and
    prints one report line per operation plus a module summary:

    * the canonical strided view (base pointer, scalar offset, per-dimension
      start / stride / size);
    * the mask classification (none, all-true, all-false, scalar, box);
    * the deferred reason when the access cannot be lowered yet.

    The pass never modifies the IR; it is a diagnostic and regression tool for
    the memory lowering pipeline.
  }];
  let constructor = "mlir::triton::createAnalyzeAccessPass()";
}

#endif
