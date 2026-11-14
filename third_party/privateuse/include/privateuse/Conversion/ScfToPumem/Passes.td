#ifndef PRIVATEUSE_CONVERSION_SCFTOPUMEM_PASSES
#define PRIVATEUSE_CONVERSION_SCFTOPUMEM_PASSES

include "mlir/Pass/PassBase.td"

def ScfToPumem : Pass<"scf-to-pumem", "mlir::ModuleOp"> {
  let summary = "Convert scf ops into the pumem dialect";
  let description = [{
    `scf.for` -> `pumem.for`, `scf.if` -> `pumem.if`, `scf.yield` ->
    `pumem.yield`. Loop bounds/step and the induction variable are widened to
    `index` (the pumem dialect uses `index` everywhere); `scf.if` conditions
    are widened to `i8` (the pumem boolean type).
  }];
  let constructor = "mlir::triton::createScfToPumemPass()";
}

#endif
