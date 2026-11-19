add_triton_library(Pipeline
  PipelinePass.cpp

  DEPENDS
  PipelineTransformPassIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRPass
  MLIRSCFDialect
  MLIRMemRefDialect
  PuStreamIR
)
