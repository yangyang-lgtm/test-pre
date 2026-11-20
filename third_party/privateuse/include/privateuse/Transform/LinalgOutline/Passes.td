#ifndef PRIVATEUSE_TRANSFORM_LINALGOUTLINE_PASSES_TD
#define PRIVATEUSE_TRANSFORM_LINALGOUTLINE_PASSES_TD

include "mlir/Pass/PassBase.td"

def LinalgOutline : Pass<"privateuse-linalg-outline", "mlir::ModuleOp"> {
  let summary = "Move buffer computes into functions and call them from the kernel";

  let description = [{
    A buffer-semantics linalg op becomes a call: the op moves into a private
    function whose parameters are the op's operands, the function is marked with
    pumem.outlined so the C++ emitter can print its prefix, and the kernel keeps
    a pumem.runtime_call that names it.

    The callee takes the descriptor of each memref operand (pointer, offset,
    sizes, strides) and the scalars as they are; the pumem conversion expands the
    call accordingly.
  }];
  let options = [
    Option<"outlinedAttr", "outlined-attr", "std::string", "\"OUTLINED\"",
           "Prefix the C++ emitter prints on outlined functions">,
  ];
  let constructor = "mlir::triton::createLinalgOutlinePass()";
}

#endif
