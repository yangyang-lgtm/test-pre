#ifndef PRIVATEUSE_CONVERSION_BUFFERIZE_PASSES_TD
#define PRIVATEUSE_CONVERSION_BUFFERIZE_PASSES_TD

include "mlir/Pass/PassBase.td"

def Bufferize : Pass<"privateuse-bufferize", "mlir::ModuleOp"> {
  let summary = "One-shot-bufferize with privateuse SRAM allocations";
  let description = [{
    Runs one-shot-bufferize over the module with the privateuse allocation
    policy: newly allocated buffers use the SRAM memory space and the default
    alignment, unknown (non-bufferizable) ops such as pumem are allowed to
    remain, and loops may return freshly allocated buffers so out-of-place
    loop-carried accumulators (e.g. flash-attention online softmax) bufferize
    instead of failing with "yield operand is not equivalent to the
    corresponding iter bbArg".
  }];
  let constructor = "mlir::triton::createBufferizePass()";
}

#endif
