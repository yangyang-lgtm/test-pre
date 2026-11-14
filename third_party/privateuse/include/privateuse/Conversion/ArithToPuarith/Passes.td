#ifndef PRIVATEUSE_CONVERSION_ARITHTOPUARITH_PASSES
#define PRIVATEUSE_CONVERSION_ARITHTOPUARITH_PASSES

include "mlir/Pass/PassBase.td"

def ArithToPuarith : Pass<"arith-to-puarith", "mlir::ModuleOp"> {
  let summary = "Convert arith ops into the puarith dialect";
  let description = [{
    1:1 op rewrite into the (cloned) puarith dialect with the i1 -> i8
    semantic change: i1-typed values (scalars, vectors and tensors) become
    i8, function signatures carrying i1 are converted too.

    Special rewrites beyond the mechanical 1:1:
    - `arith.constant` with a scalar i1 value: the attribute is rebuilt as
      i8 (`zextOrTrunc`, giving 1 instead of a sign-extended -1). All other
      constants keep their original attribute.
    - `arith.trunci` producing i1 becomes `puarith.andi %x, 1` (low bit).
    - `arith.extsi/extui` from i1 to i8 are erased (both endpoints become
      i8, so the extension is an identity).
  }];

  let constructor = "mlir::triton::createArithToPuarithPass()";
}

#endif
