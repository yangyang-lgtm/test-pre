add_triton_library(MathToPumath
  MathToPumathPass.cpp

  DEPENDS
  MathToPumathConversionPassIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRPass
  MLIRSupport
  MLIRMathDialect
  MLIRFuncDialect
  MLIRFuncTransforms
  MLIRMemRefDialect
  MLIRVectorDialect
  MLIRTransformUtils
  PuMathIR
  PuArithIR
)
