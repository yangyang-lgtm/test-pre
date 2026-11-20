add_triton_library(PrivateUseMemrefToPuStream
  MemrefToPuStreamPass.cpp

  DEPENDS
  PrivateUseMemrefToPuStreamPassIncGen

  LINK_LIBS PUBLIC
  MLIRFuncDialect
  MLIRIR
  MLIRLinalgDialect
  MLIRPass
  MLIRSCFDialect
  MLIRTransforms
  PUMemIR
  PuStreamIR
  PrivateUseUtils
)
