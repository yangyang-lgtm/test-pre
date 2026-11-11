add_triton_library(PrivateUseConversions
  AccessReportPass.cpp

  DEPENDS
  PrivateUseConversionPassIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  PrivateUseAnalysis
  TritonIR
)
