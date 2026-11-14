add_triton_library(PucIR
  PucDialect.cpp

  DEPENDS
  PucTableGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRSupport
  MLIRCallInterfaces
  MLIRControlFlowInterfaces
  MLIRFunctionInterfaces
  MLIRLoopLikeInterface
  MLIRSideEffectInterfaces
  PUMemIR
)
