#ifndef PRIVATEUSE_CONVERSION_MATHTOPUMATH_PASSES
#define PRIVATEUSE_CONVERSION_MATHTOPUMATH_PASSES

include "mlir/Pass/PassBase.td"

def MathToPumath : Pass<"math-to-pumath", "mlir::ModuleOp"> {
  let summary = "Convert math ops into the pumath dialect";
  let description = [{
    1:1 op rewrite into the (cloned) pumath dialect. The fastmath attribute
    is rebuilt as the puarith class (the arith instance does not pass the
    pumath typed getters).
  }];
  let constructor = "mlir::triton::createMathToPumathPass()";
}

#endif
