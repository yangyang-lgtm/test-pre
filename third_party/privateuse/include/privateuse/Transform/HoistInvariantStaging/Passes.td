#ifndef PRIVATEUSE_TRANSFORM_HOISTINVARIANTSTAGING_PASSES_TD
#define PRIVATEUSE_TRANSFORM_HOISTINVARIANTSTAGING_PASSES_TD

include "mlir/Pass/PassBase.td"

def HoistInvariantStaging : Pass<"privateuse-hoist-invariant-staging", "mlir::ModuleOp"> {
  let summary = "Hoist loop-invariant SRAM staging out of loops";
  let description = [{
    The structured access lowering stages every loaded block through

      %buf = memref.alloc() : memref<..., 128>
      memref.copy %view, %buf : ...
      %tensor = bufferization.to_tensor %buf restrict writable : ...

    inside the pid loop. When the source view is loop invariant the copy is
    redundant in every iteration, but neither the allocation nor the copy can
    be hoisted by standard LICM (memory effects), and hoisting after
    bufferization would let in-place reuse of the buffer change meaning.

    This pass moves the alloc, the copy and the to_tensor bridges in front of
    the loop when the copy fills a whole, statically shaped buffer that it
    alone writes, the source view is defined outside the loop, and every other
    use of the buffer is a to_tensor bridge in the same block that only feeds
    reads. The tensor then becomes loop invariant in SSA, so bufferization
    cannot reuse the buffer inside the loop.

    Runs innermost loops first, so a buffer hoisted to an inner preheader is
    still considered for the enclosing loop.
  }];
  let constructor = "mlir::triton::createHoistInvariantStagingPass()";
}

#endif
