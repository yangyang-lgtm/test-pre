add_subdirectory(CollapseShape)
add_subdirectory(LowerLinalgReduce)
add_subdirectory(PidLoop)
add_subdirectory(PucLaunchKernel)
add_subdirectory(PucLocalizeCarried)
add_subdirectory(PucSimplifyAssign)
add_subdirectory(PucVoidPtrArgs)
add_subdirectory(SinkConstantCallArgs)
add_subdirectory(SpecializeKernel)

add_triton_library(PrivateUseTransforms
  DEPENDS
  CollapseShapeTransformPassIncGen
  LowerLinalgReduceTransformPassIncGen
  PidLoopTransformPassIncGen
  PucLaunchKernelTransformPassIncGen
  PucLocalizeCarriedTransformPassIncGen
  PucSimplifyAssignTransformPassIncGen
  PucVoidPtrArgsTransformPassIncGen
  SinkConstantCallArgsTransformPassIncGen
  SpecializeKernelTransformPassIncGen

  LINK_LIBS PUBLIC
  CollapseShape
  LowerLinalgReduce
  PidLoop
  PucLaunchKernel
  PucLocalizeCarried
  PucSimplifyAssign
  PucVoidPtrArgs
  SinkConstantCallArgs
  SpecializeKernel
)
