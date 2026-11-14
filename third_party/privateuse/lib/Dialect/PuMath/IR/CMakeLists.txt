add_triton_library(PuMathIR
  PuMathOps.cpp
  PuMathDialect.cpp

  DEPENDS
  PuMathTableGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRSupport
  MLIRFuncDialect
  PuArithIR
)
