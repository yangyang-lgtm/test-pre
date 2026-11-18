add_triton_library(ScheduleTimepoints
  ScheduleTimepointsPass.cpp

  DEPENDS
  ScheduleTimepointsTransformPassIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRMemRefDialect
  MLIRPass
  MLIRSCFDialect
  PuStreamIR
)
