#ifndef PRIVATEUSE_TRANSFORM_LINALGTORUNTIME_PASSES_TD
#define PRIVATEUSE_TRANSFORM_LINALGTORUNTIME_PASSES_TD

include "mlir/Pass/PassBase.td"

def LinalgToRuntime : Pass<"privateuse-linalg-to-runtime", "mlir::ModuleOp"> {
  let summary = "Call the hand written runtime kernels for matmul and transpose";

  let description = [{
    A matmul or a transpose on static rank-2 buffers becomes a call to the
    runtime kernel for it, with the geometry in the template arguments:
    runtime_matmul<T, M, N, K>(A, B, C) and runtime_transpose<T, Rows, Cols>(src,
    dst). Everything else is left for the linalg lowering.
  }];
  let constructor = "mlir::triton::createLinalgToRuntimePass()";
}

#endif
