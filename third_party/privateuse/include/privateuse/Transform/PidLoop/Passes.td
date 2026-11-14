#ifndef PRIVATEUSE_TRANSFORM_PIDLOOP_PASSES_TD
#define PRIVATEUSE_TRANSFORM_PIDLOOP_PASSES_TD

include "mlir/Pass/PassBase.td"

def PidLoop : Pass<"privateuse-pid-loop", "mlir::ModuleOp"> {
  let summary = "Turn the program ids of a kernel into a loop over physical threads";
  let description = [{
    Wraps the body of every public kernel in a grid-stride loop over the
    linear program id:

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
