add_subdirectory(SpecializeKernel)

add_triton_library(PrivateUseTransforms
  DEPENDS
  SpecializeKernelTransformPassIncGen

  LINK_LIBS PUBLIC
  SpecializeKernel
)
