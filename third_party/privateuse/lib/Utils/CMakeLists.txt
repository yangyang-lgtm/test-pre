add_triton_library(PrivateUseUtils
  Utils.cpp

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRSupport
  MLIRLinalgDialect
  TritonIR
)
