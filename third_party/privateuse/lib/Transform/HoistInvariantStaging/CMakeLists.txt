add_triton_library(HoistInvariantStaging
  HoistInvariantStagingPass.cpp

  DEPENDS
  HoistInvariantStagingTransformPassIncGen

  LINK_LIBS PUBLIC
  MLIRBufferizationDialect
  MLIRDestinationStyleOpInterface
  MLIRIR
  MLIRMemRefDialect
  MLIRPass
  MLIRSCFDialect
  MLIRSideEffectInterfaces
)
