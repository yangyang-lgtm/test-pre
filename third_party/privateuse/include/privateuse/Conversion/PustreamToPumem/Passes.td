#ifndef PRIVATEUSE_CONVERSION_PUSTREAMTOPUMEM_PASSES_TD
#define PRIVATEUSE_CONVERSION_PUSTREAMTOPUMEM_PASSES_TD

include "mlir/Pass/PassBase.td"

def PustreamToPumem : Pass<"privateuse-pustream-to-pumem", "mlir::ModuleOp"> {
  let summary = "Turn the async statements into calls into the runtime";

  let description = [{
    Each statement moves to the runtime vocabulary it stands for:

    - pustream.alloc becomes memref.alloc plus pumem.dma_slot, which mints the
      handle and the metadata slot the allocation is waited on;
    - transfer and copy become async_copy_<direction><Rank> with the box read
      from memref.extract_strided_metadata, and hand the token back;
    - await, await_compute and sync become pumem_wait, pumem_wait_dispatch and
      pumem_sync;
    - dispatch becomes a call to the callee, carrying the views and the scalars
      in operand order.

    The views stay memrefs: the pointer is only needed by the call itself, and
    -memref-to-pumem materializes one element pointer per memref operand (or the
    whole descriptor, for an outlined callee), so this pass can run before it.
  }];

  let constructor = "mlir::triton::createPustreamToPumemPass()";
}

#endif
