#ifndef PRIVATEUSE_TRANSFORM_MEMREFTOPUSTREAM_PASSES_TD
#define PRIVATEUSE_TRANSFORM_MEMREFTOPUSTREAM_PASSES_TD

include "mlir/Pass/PassBase.td"

def MemrefToPuStream : Pass<"privateuse-memref-to-pustream", "mlir::ModuleOp"> {
  let summary = "Turn buffer memory and computes into async statements";

  let description = [{
    Three rewrites, all of which need the buffer identity:

    - memref.alloc becomes pustream.alloc, which hands out the token (identity
      and readiness) and the memref (geometry), so every use keeps its memref;
    - memref.copy becomes a transfer (DRAM <-> SRAM) or a copy (SRAM <-> SRAM)
      followed by its wait;
    - a buffer-semantics linalg op becomes a dispatch. A matmul or transpose
      with static geometry names the hand-written runtime kernel and its
      template arguments; anything else is outlined into a private function
      whose parameters are the op's buffers and scalars, and that function is
      marked with pumem.outlined so the C++ emitter can print its prefix.
  }];
  let options = [
    Option<"outlinedAttr", "outlined-attr", "std::string", "\"OUTLINED\"",
           "Prefix the C++ emitter prints on outlined functions">,
  ];
  let constructor = "mlir::triton::createMemrefToPuStreamPass()";
}

#endif
