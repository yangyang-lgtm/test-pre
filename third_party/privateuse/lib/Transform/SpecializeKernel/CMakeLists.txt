add_triton_library(SpecializeKernel
  SpecializeKernelPass.cpp

  DEPENDS
  SpecializeKernelTransformPassIncGen

  LINK_LIBS PUBLIC
  MLIRArithDialect
  MLIRFuncDialect
  MLIRIR
  PrivateUseUtils
  TritonIR
)
