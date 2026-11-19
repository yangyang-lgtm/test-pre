add_triton_library(SinkConstantCallArgs
  SinkConstantCallArgsPass.cpp

  DEPENDS
  SinkConstantCallArgsTransformPassIncGen

  LINK_LIBS PUBLIC
  MLIRFuncDialect
  MLIRFunctionInterfaces
  MLIRIR
  MLIRPass
  PUMemIR
)
