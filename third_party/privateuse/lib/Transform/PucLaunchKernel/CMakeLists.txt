add_triton_library(PucLaunchKernel
  PucLaunchKernelPass.cpp

  DEPENDS
  PucLaunchKernelTransformPassIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRPass
  PrivateUseUtils
  PUMemIR
  PucIR
)
