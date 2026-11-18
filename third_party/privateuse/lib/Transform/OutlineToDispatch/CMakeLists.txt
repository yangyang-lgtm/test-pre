add_triton_library(OutlineToDispatch
  OutlineToDispatchPass.cpp

  DEPENDS
  OutlineToDispatchTransformPassIncGen

  LINK_LIBS PUBLIC
  MLIRFuncDialect
  MLIRFunctionInterfaces
  MLIRIR
  MLIRLinalgDialect
  MLIRMemRefDialect
  MLIRPass
  MLIRSideEffectInterfaces
  PUMemIR
  PuStreamIR
)
