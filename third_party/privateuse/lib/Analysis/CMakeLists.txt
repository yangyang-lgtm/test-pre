add_triton_library(PrivateUseAnalysis
  AccessAnalysis.cpp
  ValueRange.cpp

  LINK_LIBS PUBLIC
  MLIRArithDialect
  MLIRIR
  MLIRSCFDialect
  MLIRSupport
  PrivateUseUtils
  TritonIR
)
