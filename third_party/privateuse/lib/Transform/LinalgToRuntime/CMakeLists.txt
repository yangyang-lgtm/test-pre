add_triton_library(LinalgToRuntime
  LinalgToRuntimePass.cpp

  DEPENDS
  LinalgToRuntimeTransformPassIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRLinalgDialect
  MLIRPass
  MLIRTransforms
  PUMemIR
)
