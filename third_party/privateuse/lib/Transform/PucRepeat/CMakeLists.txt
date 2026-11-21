add_triton_library(PucRepeat
  PucRepeatPass.cpp

  DEPENDS
  PucRepeatTransformPassIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRPass
  PucIR
)
