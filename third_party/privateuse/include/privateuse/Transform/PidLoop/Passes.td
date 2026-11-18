#ifndef PRIVATEUSE_TRANSFORM_PIDLOOP_PASSES_TD
#define PRIVATEUSE_TRANSFORM_PIDLOOP_PASSES_TD

include "mlir/Pass/PassBase.td"

def PidLoop : Pass<"privateuse-pid-loop", "mlir::ModuleOp"> {
  let summary = "Map program ids onto physical threads";
  let description = [{
    Maps the program ids of every public kernel onto the physical threads:

      %tid = pumem.get_physical_id
      scf.for %l = %tid to %total step num_pids {
        ... kernel body, with every tt.get_program_id replaced by the
            decomposition of %l into the x, y and z program ids ...
      }

    The grid is read from the module attribute privateuse.grid, so the loop
    bound is a compile-time constant and the decomposition folds the axes the
    program id never leaves: a one-dimensional grid keeps x = %l, a grid
    without a z axis has y = %l / nx, and z is zero. num_pids is the number
    of physical threads and becomes the loop step.

    When the grid has exactly num_pids blocks every thread runs the body
    once, so the loop is elided and the body is inlined with %tid as the
    linear program id. When the grid has fewer blocks, the loop is replaced
    by scf.if %tid < total so threads without a block skip the body.

    Kernels are the public, non-external functions with a single block and no
    results; private helpers are left untouched.
  }];
  let options = [
    Option<"numPids", "num-pids", "int64_t", /*default=*/"0",
           "Number of physical threads; the program loop strides by it">
  ];
  let constructor = "mlir::triton::createPidLoopPass()";
}

#endif
