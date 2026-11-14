add_triton_library(PrivateUseMemrefToPumem
  MemrefToPumemPass.cpp

  DEPENDS
  PrivateUseMemrefToPumemPassIncGen

  LINK_LIBS PUBLIC
  MLIRArithDialect
  MLIRDialectUtils
  MLIRFuncDialect
  MLIRFuncTransforms
  MLIRIR
  MLIRMemRefDialect
  MLIRPass
  MLIRSCFDialect
  MLIRSupport
  MLIRTransforms
  PUMemIR
  PrivateUseUtils
  TritonIR
)
