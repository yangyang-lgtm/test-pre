add_triton_library(PUMemIR
  PUMemDialect.cpp

  DEPENDS
  PUMemTableGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRSupport
  MLIRSideEffectInterfaces
)
