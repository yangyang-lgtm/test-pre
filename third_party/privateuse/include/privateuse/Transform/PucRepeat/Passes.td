#ifndef PUC_REPEAT_TRANSFORM_PASSES_TD
#define PUC_REPEAT_TRANSFORM_PASSES_TD

include "mlir/Pass/PassBase.td"

def PucRepeat : Pass<"privateuse-puc-repeat", "mlir::ModuleOp"> {
  let summary = "Repeat each launcher body and report the average time per run";
  let description = [{
    A benchmark switch, not part of the normal pipeline. For every launcher it

      %before  = call @pumem_timer_read()
      puc.for %i = 0 to iterations step 1 { the launch }
      %after   = call @pumem_timer_read()
      %elapsed = call @scalar_subi(%after, %before)
      puc.timer_report "name", iterations, %elapsed

    so one invocation reports the average time of a single launch instead of a
    single measurement dominated by noise. The emitter prints it as
    pumem_timer_report("name", iterations, elapsed); the runtime divides.
  }];
  let options = [
    Option<"iterations", "iterations", "int64_t", "100",
           "How many times the launcher body runs in the measuring loop">
  ];
  let constructor = "mlir::triton::createPucRepeatPass()";
}

#endif
