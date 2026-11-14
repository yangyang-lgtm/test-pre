#ifndef PUC_SIMPLIFY_ASSIGN_TRANSFORM_PASSES_TD
#define PUC_SIMPLIFY_ASSIGN_TRANSFORM_PASSES_TD

include "mlir/Pass/PassBase.td"

def PucSimplifyAssign : Pass<"puc-simplify-assign", "mlir::ModuleOp"> {
  let summary = "Remove redundant puc.local/puc.assign pairs";
  let description = [{
    Cleans up after `puc-localize-carried`:

    1. **constant locals** - a local whose initial value is a constant and
       whose assignments always store that same constant is replaced by the
       constant; its assignments and the local are erased. This removes the
       invariant descriptor components (offset / sizes / strides) that used to
       be carried through the loop.
    2. **self assignments** - `puc.assign %v, %v` is erased.
    3. **staging inlining** - a local that is never written and feeds one
       assignment is folded into it, unless its source is reassigned later
       (that case needs the staging copy to implement a parallel assignment).
    4. **dead locals** - locals without uses and without assignments are erased.

    Runs right after `puc-localize-carried`, i.e. still after
    `-privateuse-to-puc` and before translation.
  }];
  let constructor = "mlir::triton::createPucSimplifyAssignPass()";
}

#endif
