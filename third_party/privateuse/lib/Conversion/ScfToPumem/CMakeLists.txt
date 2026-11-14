add_triton_library(ScfToPumem
  ScfToPumemPass.cpp

  DEPENDS
  ScfToPumemConversionPassIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRPass
  MLIRSupport
  MLIRSCFDialect
  MLIRArithDialect
  MLIRFuncDialect
  MLIRFuncTransforms
  MLIRMemRefDialect
  MLIRVectorDialect
  MLIRTransformUtils
  PUMemIR
)
