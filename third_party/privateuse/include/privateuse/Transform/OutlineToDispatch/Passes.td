#ifndef PRIVATEUSE_TRANSFORM_OUTLINETODISPATCH_PASSES_TD
#define PRIVATEUSE_TRANSFORM_OUTLINETODISPATCH_PASSES_TD

include "mlir/Pass/PassBase.td"

def OutlineToDispatch : Pass<"privateuse-outline-to-dispatch", "mlir::ModuleOp"> {
  let summary = "Outline buffer-level linalg ops and turn them into async dispatches";
  let description = [{
    Every top-level linalg op with pure buffer semantics is moved into a
    private func.func named outlined_<op>_<body ops>_<n>. The outlined
    function takes the operand values (and any values captured from the
    enclosing scope) as arguments; the original site becomes a
    `pustream.async.dispatch` of that function. Memrefs keep their ranked
    type, so the outlined functions feed the same conversion pipeline as the
    kernel itself.

    The `reads` and `writes` operand indices of the dispatch are derived from
    the linalg op's own memory effects; a memref captured from the enclosing
    scope cannot be attributed that way and is conservatively marked as both
    read and written.
  }];
  let options = [
    Option<"outlinedAttrValue", "outlined-attr-value", "std::string", "",
           "Value assigned to the `pumem.outlined` attribute on newly outlined funcs">
  ];
  let dependentDialects = ["::mlir::pustream::PuStreamDialect"];
  let constructor = "mlir::triton::createOutlineToDispatchPass()";
}

#endif
