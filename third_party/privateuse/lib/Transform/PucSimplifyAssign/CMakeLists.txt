add_triton_library(PucSimplifyAssign
  PucSimplifyAssignPass.cpp

  DEPENDS
  PucSimplifyAssignTransformPassIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRPass
  MLIRSupport
  PucIR
)
