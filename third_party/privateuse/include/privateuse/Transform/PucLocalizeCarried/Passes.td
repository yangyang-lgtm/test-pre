#ifndef PUC_LOCALIZE_CARRIED_TRANSFORM_PASSES_TD
#define PUC_LOCALIZE_CARRIED_TRANSFORM_PASSES_TD

include "mlir/Pass/PassBase.td"

def PucLocalizeCarried : Pass<"puc-localize-carried", "mlir::ModuleOp"> {
  let summary = "Turn puc.for/puc.if results into mutable locals and assignments";
  let description = [{
    Replaces value-carrying regions with explicit mutable locals, so that
    `puc.for` and `puc.if` no longer return values and their terminators
    carry no operands. Every carried value becomes

    ```mlir
    %v = puc.local %init : T
    puc.for ... {
      ...
      puc.assign %v, %next : T, T
    }
    ```

    which puc-translate emits as a C++ local plus direct assignments, matching
    the shape of the generated kernel. Identity assignments (a value carried
    unchanged) are dropped, and locals that end up unused are erased.

    Runs after `-privateuse-to-puc`: after this pass `puc.for` no longer
    satisfies the LoopLikeOpInterface result/yield contract, so no
    optimization pass may run afterwards.
  }];
  let constructor = "mlir::triton::createPucLocalizeCarriedPass()";
}

#endif
