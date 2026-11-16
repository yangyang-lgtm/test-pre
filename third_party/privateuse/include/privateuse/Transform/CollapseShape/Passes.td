#ifndef PRIVATEUSE_TRANSFORM_COLLAPSESHAPE_PASSES_TD
#define PRIVATEUSE_TRANSFORM_COLLAPSESHAPE_PASSES_TD

include "mlir/Pass/PassBase.td"

def CollapseShape : Pass<"collapse-shape", "mlir::ModuleOp"> {
  let summary = "Compress tensor dimensions to improve linalg{broadcast,elementwise,transpose,fill,reduce} efficiency";
  let constructor = "mlir::triton::createCollapseShapePass()";
}

#endif
