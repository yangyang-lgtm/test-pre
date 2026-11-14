#ifndef PUC_VOID_PTR_ARGS_TRANSFORM_PASSES_TD
#define PUC_VOID_PTR_ARGS_TRANSFORM_PASSES_TD

include "mlir/Pass/PassBase.td"

def PucVoidPtrArgs : Pass<"puc-void-ptr-args", "mlir::ModuleOp"> {
  let summary = "Erase the pointee type of puc function pointer arguments";
  let description = [{
    Replaces every `!pumem.ptr<T, space>` argument of a `puc.func` with
    `!pumem.ptr<void, space>` and casts it back with `puc.point_cast` at the
    start of the entry block, leaving the rest of the body untouched. This
    gives the generated C functions a uniform `void*` ABI for pointer
    arguments, matching the launcher declaration the driver compiles against.

    External declarations and arguments that already point to `void` are left
    alone. Runs after `-puc-simplify-assign`, before the launcher is created.
  }];
  let constructor = "mlir::triton::createPucVoidPtrArgsPass()";
}

#endif
