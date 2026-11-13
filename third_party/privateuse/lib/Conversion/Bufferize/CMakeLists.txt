add_triton_library(PrivateUseBufferize
  BufferizePass.cpp

  DEPENDS
  PrivateUseBufferizePassIncGen

  LINK_LIBS PUBLIC
  MLIRBufferizationDialect
  MLIRBufferizationTransforms
  MLIRLinalgDialect
  MLIRTensorDialect
  MLIRIR
  MLIRPass
  MLIRTransforms
  TritonIR
)
