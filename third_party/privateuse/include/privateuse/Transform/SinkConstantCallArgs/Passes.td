#ifndef PRIVATEUSE_TRANSFORM_SINKCONSTANTCALLARGS_PASSES_TD
#define PRIVATEUSE_TRANSFORM_SINKCONSTANTCALLARGS_PASSES_TD

include "mlir/Pass/PassBase.td"

def SinkConstantCallArgs : Pass<"privateuse-sink-constant-call-args", "mlir::ModuleOp"> {
  let summary = "Move constant call arguments into the callee and drop unused arguments";
  let description = [{
    A private function that is only called with the same constant in an
    argument position loses that argument: the constant is rebuilt inside the
    callee, the block argument uses are redirected to the clone and the call
    drops the operand. An argument without uses inside the callee is erased as
    well, so the call sites also drop it. This runs after the pumem tuple ABI
    expansion, which turns the static sizes, strides and offsets of a memref
    into constant call arguments or into unused ones.
  }];
  let constructor = "mlir::triton::createSinkConstantCallArgsPass()";
}

#endif
