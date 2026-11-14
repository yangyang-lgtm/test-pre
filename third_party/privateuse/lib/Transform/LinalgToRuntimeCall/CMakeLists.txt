add_triton_library(LinalgToRuntimeCall
  LinalgToRuntimeCallPass.cpp

  DEPENDS
  LinalgToRuntimeCallTransformPassIncGen

  LINK_LIBS PUBLIC
  MLIRArithDialect
  MLIRIR
  MLIRLinalgDialect
  MLIRMemRefDialect
  MLIRPass
  MLIRTransforms
  PUMemIR
)
