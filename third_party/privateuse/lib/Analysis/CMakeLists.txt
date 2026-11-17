add_triton_library(PrivateUseAnalysis
  AccessAnalysis.cpp
  DataflowAnalysis.cpp
  ValueRange.cpp

  LINK_LIBS PUBLIC
  MLIRArithDialect
  MLIRCallInterfaces
  MLIRControlFlowInterfaces
  MLIRFunctionInterfaces
  MLIRIR
  MLIRLinalgDialect
  MLIRMemRefDialect
  MLIRSCFDialect
  MLIRSupport
  MLIRViewLikeInterface
  PrivateUseUtils
  PUMemIR
  TritonIR
)
