add_triton_library(MemrefToPuStream
  MemrefToPuStreamPass.cpp

  DEPENDS
  MemrefToPuStreamTransformPassIncGen

  LINK_LIBS PUBLIC
  MLIRFuncDialect
  MLIRIR
  MLIRLinalgDialect
  MLIRPass
  MLIRTransforms
  PUMemIR
  PrivateUseUtils
)
