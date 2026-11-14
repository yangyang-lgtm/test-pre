#ifndef PRIVATEUSE_TRANSFORM_OUTLINELINALGOPS_PASSES_TD
#define PRIVATEUSE_TRANSFORM_OUTLINELINALGOPS_PASSES_TD

include "mlir/Pass/PassBase.td"

def OutlineLinalgOps : Pass<"privateuse-outline-linalg-ops", "mlir::ModuleOp"> {
  let summary = "Outline buffer-level linalg ops into private functions";
  let description = [{
    Every top-level linalg op with pure buffer semantics is moved into a
    private func.func named outlined_<op>_<body ops>_<n>. The outlined
    function takes the operand values (and any values captured from the
    enclosing scope) as arguments and returns the op results; the original
    site becomes a func.call. Memrefs keep their ranked type, so the outlined
    functions feed the same conversion pipeline as the kernel itself.
  }];
  let options = [
    Option<"outlinedAttrValue", "outlined-attr-value", "std::string", "",
           "Value assigned to the `pumem.outlined` attribute on newly outlined funcs">
  ];
  let constructor = "mlir::triton::createOutlineLinalgOpsPass()";
}

#endif
