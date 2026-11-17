add_triton_library(PrivateUseDataflowAnalysis
  DataflowReportPass.cpp

  DEPENDS
  PrivateUseDataflowAnalysisPassIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRSupport
  PrivateUseAnalysis
)