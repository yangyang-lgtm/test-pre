add_triton_library(PucTranslate
  PucTranslate.cpp

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRSupport
  PrivateUseUtils
  PUMemIR
  PucIR
)
