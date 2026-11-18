add_triton_library(ToPuStream
  ToPuStreamPass.cpp

  DEPENDS
  ToPuStreamConversionPassIncGen

  LINK_LIBS PUBLIC
  MLIRArithDialect
  MLIRIR
  MLIRLinalgDialect
  MLIRMemRefDialect
  MLIRPass
  MLIRTransforms
  PuStreamIR
)
