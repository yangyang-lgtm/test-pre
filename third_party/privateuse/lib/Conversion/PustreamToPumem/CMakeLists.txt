add_triton_library(PrivateUsePustreamToPumem
  PustreamToPumem.cpp

  DEPENDS
  PrivateUsePustreamToPumemPassIncGen

  LINK_LIBS PUBLIC
  MLIRArithDialect
  MLIRFuncDialect
  MLIRIR
  MLIRMemRefDialect
  MLIRPass
  MLIRSCFDialect
  MLIRSCFTransforms
  MLIRSCFDialect
  MLIRSCFTransforms
  MLIRSupport
  MLIRTransforms
  PUMemIR
  PuStreamIR
  PrivateUseUtils
)
