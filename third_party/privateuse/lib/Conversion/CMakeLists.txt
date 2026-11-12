add_triton_library(PrivateUseConversions
  AccessReportPass.cpp
  LowerStructuredAccessPass.cpp
  PtrToMemrefPass.cpp

  DEPENDS
  PrivateUseConversionPassIncGen

  LINK_LIBS PUBLIC
  MLIRArithDialect
  MLIRBufferizationDialect
  MLIRIR
  MLIRMemRefDialect
  MLIRSCFDialect
  MLIRSCFTransforms
  MLIRTensorDialect
  MLIRTransforms
  PrivateUseAnalysis
  TritonIR
)
