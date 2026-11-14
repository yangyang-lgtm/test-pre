add_triton_library(PidLoop
  PidLoopPass.cpp

  DEPENDS
  PidLoopTransformPassIncGen

  LINK_LIBS PUBLIC
  MLIRArithDialect
  MLIRFuncDialect
  MLIRIR
  MLIRPass
  MLIRSCFDialect
  PrivateUseUtils
  PUMemIR
  TritonIR
)
