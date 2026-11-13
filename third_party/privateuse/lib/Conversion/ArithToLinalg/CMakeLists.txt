add_triton_library(PrivateUseArithToLinalg
  ArithToLinalgPass.cpp

  DEPENDS
  PrivateUseArithToLinalgPassIncGen

  LINK_LIBS PUBLIC
  MLIRLinalgTransforms
  MLIRAffineDialect
  MLIRArithDialect
  MLIRBufferizationDialect
  MLIRControlFlowDialect
  MLIRDialectUtils
  MLIRFuncDialect
  MLIRIR
  MLIRLinalgDialect
  MLIRMathDialect
  MLIRMemRefDialect
  MLIRPass
  MLIRSCFDialect
  MLIRSupport
  MLIRTensorDialect
  MLIRTransforms
  PUMemIR
  TritonIR
  PrivateUseUtils
)