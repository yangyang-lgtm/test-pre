#ifndef SPECIALIZE_KERNEL_TRANSFORM_PASSES_TD
#define SPECIALIZE_KERNEL_TRANSFORM_PASSES_TD

include "mlir/Pass/PassBase.td"

def SpecializeKernel : Pass<"privateuse-specialize-kernel", "mlir::ModuleOp"> {
  let summary = "Fold the launch-time values into the kernel";
  let description = [{
    Turns the launch-time information into compile-time constants:

    * every scalar kernel argument listed by index becomes an arith.constant
      of its own type and the argument is dropped from the signature;
    * tt.get_num_programs is replaced by the matching grid dimension;
    * the grid itself is recorded as the module attribute privateuse.grid so
      that later passes (privateuse-pid-loop) can use it.

    The (index, value) constants are supplied by the driver through the pass
    constructor, so a module is compiled for exactly one launch shape.
  }];
  let constructor = "mlir::triton::createSpecializeKernelPass()";
}

#endif
