add_triton_library(PrivateUseAccessLowering
  LowerStructuredAccessPass.cpp
  LowerUnstructuredAccessPass.cpp
  PtrToMemrefPass.cpp

  DEPENDS
  PrivateUseAccessLoweringPassIncGen

  LINK_LIBS PUBLIC
  MLIRArithDialect
  MLIRBufferizationDialect
  MLIRIR
  MLIRLinalgDialect
  MLIRMemRefDialect
  MLIRSCFDialect
  MLIRSCFTransforms
  MLIRTensorDialect
  MLIRTransforms
  PrivateUseAnalysis
  PUMemIR
  TritonIR
)
