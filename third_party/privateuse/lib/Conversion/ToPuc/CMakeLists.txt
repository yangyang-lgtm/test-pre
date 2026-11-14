add_triton_library(PrivateUseToPuc
  FuncToPuc.cpp
  PumemToPuc.cpp
  ScalarToPuc.cpp
  ToPucPass.cpp

  DEPENDS
  PrivateUseToPucPassIncGen

  LINK_LIBS PUBLIC
  MLIRArithDialect
  MLIRFuncDialect
  MLIRIR
  MLIRPass
  MLIRSupport
  MLIRTransforms
  PUMemIR
  PuArithIR
  PuMathIR
  PucIR
)
