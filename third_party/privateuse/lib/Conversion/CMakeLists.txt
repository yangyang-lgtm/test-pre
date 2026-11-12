add_triton_library(PrivateUseConversions
  AccessReportPass.cpp
  LowerStructuredAccessPass.cpp
  LowerUnstructuredAccessPass.cpp
  PtrToMemrefPass.cpp

  DEPENDS
  PrivateUseConversionPassIncGen

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
