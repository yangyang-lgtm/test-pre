add_triton_library(LinalgOutline
  LinalgOutlinePass.cpp

  DEPENDS
  LinalgOutlineTransformPassIncGen

  LINK_LIBS PUBLIC
  MLIRFuncDialect
  MLIRIR
  MLIRLinalgDialect
  MLIRPass
  MLIRTransforms
  PUMemIR
  PrivateUseUtils
)
