#ifndef PRIVATEUSE_TRANSFORM_PIPELINE_PASSES_TD
#define PRIVATEUSE_TRANSFORM_PIPELINE_PASSES_TD

include "mlir/Pass/PassBase.td"

def Pipeline : Pass<"privateuse-pipeline", "mlir::ModuleOp"> {
  let summary = "Software-pipeline tile loops of the pustream stream";
  let description = [{
    Recognises canonical tile loops of the statement-form pustream dialect:

      scf.for %i { copy dram->slot; dispatch slot; copy slot->dram }

    and rewrites them into stage-buffered loops with an N-deep prologue and a
    predicated prefetch, so that the steady state runs compute(i) concurrently
    with load(i+N-1) and store(i-S).

    The pass owns everything inside a loop it rewrites -- it emits the per-slot
    waits, the store barrier and the prologue itself, and tags the loop with
    `pustream.pipelined`. The wait-insertion pass skips tagged loops, otherwise
    its region drain would serialise every iteration.

    Loops that do not match are left untouched and, with `verbose`, reported.
  }];

  let options = [
    Option<"verbose", "verbose", "bool", "false",
           "Report why each tile loop is or is not pipelined">,
    Option<"stages", "stages", "unsigned", "2",
           "Multi-buffer depth used for both the input and the output side">
  ];

  let dependentDialects = [
    "::mlir::arith::ArithDialect",
    "::mlir::scf::SCFDialect",
    "::mlir::pustream::PuStreamDialect"
  ];
  let constructor = "mlir::triton::createPipelinePass()";
}

#endif // PRIVATEUSE_TRANSFORM_PIPELINE_PASSES_TD
