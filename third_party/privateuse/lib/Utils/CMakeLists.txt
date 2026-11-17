add_triton_library(PrivateUseUtils
  Utils.cpp

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRSupport
  MLIRLinalgDialect
  TritonIR
)

add_subdirectory(AccessAnalysis)
add_subdirectory(DataflowAnalysis)
