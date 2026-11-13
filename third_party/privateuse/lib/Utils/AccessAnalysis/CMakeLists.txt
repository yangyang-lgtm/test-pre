add_triton_library(PrivateUseAccessAnalysis
  AccessReportPass.cpp

  DEPENDS
  PrivateUseAccessAnalysisPassIncGen

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
