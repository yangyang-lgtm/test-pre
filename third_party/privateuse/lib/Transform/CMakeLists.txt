add_subdirectory(LinalgToRuntimeCall)
add_subdirectory(LowerLinalgReduce)
add_subdirectory(OutlineLinalgOps)
add_subdirectory(PidLoop)
add_subdirectory(PucLaunchKernel)
add_subdirectory(PucLocalizeCarried)
add_subdirectory(PucSimplifyAssign)
add_subdirectory(SinkConstantCallArgs)
add_subdirectory(SpecializeKernel)

add_triton_library(PrivateUseTransforms
  DEPENDS
  LinalgToRuntimeCallTransformPassIncGen
  LowerLinalgReduceTransformPassIncGen
  OutlineLinalgOpsTransformPassIncGen
  PidLoopTransformPassIncGen
  PucLaunchKernelTransformPassIncGen
  PucLocalizeCarriedTransformPassIncGen
  PucSimplifyAssignTransformPassIncGen
  SinkConstantCallArgsTransformPassIncGen
  SpecializeKernelTransformPassIncGen

  LINK_LIBS PUBLIC
  LinalgToRuntimeCall
  LowerLinalgReduce
  OutlineLinalgOps
  PidLoop
  PucLaunchKernel
  PucLocalizeCarried
  PucSimplifyAssign
  SinkConstantCallArgs
  SpecializeKernel
)
