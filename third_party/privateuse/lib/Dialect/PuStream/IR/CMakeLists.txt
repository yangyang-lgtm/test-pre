add_triton_library(PuStreamIR
  PuStreamDialect.cpp

  DEPENDS
  PuStreamTableGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRMemRefDialect
  MLIRSCFDialect
  MLIRSupport
  MLIRSideEffectInterfaces
)
