add_triton_library(PucLocalizeCarried
  PucLocalizeCarriedPass.cpp

  DEPENDS
  PucLocalizeCarriedTransformPassIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRPass
  MLIRSupport
  PucIR
)
