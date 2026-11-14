#ifndef PRIVATEUSE_TRANSFORM_LINALGTORUNTIMECALL_PASSES_TD
#define PRIVATEUSE_TRANSFORM_LINALGTORUNTIMECALL_PASSES_TD

include "mlir/Pass/PassBase.td"

def LinalgToRuntimeCall : Pass<"privateuse-linalg-to-runtime-call", "mlir::ModuleOp"> {
  let summary = "Lower statically shaped linalg.matmul / linalg.transpose to runtime calls";
  let description = [{
    Replaces

    - `linalg.matmul ins(A, B) outs(C)` with
      `pumem.runtime_call @runtime_matmul<[elems, M, N, K]>(A, B, C)`,
    - `linalg.transpose ins(A) outs(B) permutation = [1, 0]` with
      `pumem.runtime_call @runtime_transpose<[elems, rows, cols]>(A, B)`.

    Both patterns require statically shaped rank-2 memrefs with matching
    element types. A `linalg.matmul` or `linalg.transpose` that does not meet
    the requirements is reported as an error and fails the pass.
  }];
  let constructor = "mlir::triton::createLinalgToRuntimeCallPass()";
}

#endif
