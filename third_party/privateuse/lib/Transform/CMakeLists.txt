add_subdirectory(LowerLinalgReduce)
add_subdirectory(OutlineLinalgOps)
add_subdirectory(PidLoop)
add_subdirectory(SinkConstantCallArgs)
add_subdirectory(SpecializeKernel)

add_triton_library(PrivateUseTransforms
  DEPENDS
  LowerLinalgReduceTransformPassIncGen
  OutlineLinalgOpsTransformPassIncGen
  PidLoopTransformPassIncGen
  SinkConstantCallArgsTransformPassIncGen
  SpecializeKernelTransformPassIncGen

  LINK_LIBS PUBLIC
  LowerLinalgReduce
  OutlineLinalgOps
  PidLoop
  SinkConstantCallArgs
  SpecializeKernel
)
