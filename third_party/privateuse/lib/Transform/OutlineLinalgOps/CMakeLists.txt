add_triton_library(OutlineLinalgOps
  OutlineLinalgOpsPass.cpp

  DEPENDS
  OutlineLinalgOpsTransformPassIncGen

  LINK_LIBS PUBLIC
  MLIRFuncDialect
  MLIRFunctionInterfaces
  MLIRIR
  MLIRLinalgDialect
  MLIRMemRefDialect
  MLIRPass
)
