add_triton_library(ArithToPuarith
  ArithToPuarithPass.cpp

  DEPENDS
  ArithToPuarithConversionPassIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRPass
  MLIRSupport
  MLIRFuncDialect
  MLIRFuncTransforms
  MLIRArithDialect
  MLIRMemRefDialect
  MLIRVectorDialect
  MLIRTransformUtils
  PuArithIR
)
