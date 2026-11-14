add_triton_library(LowerLinalgReduce
  LowerLinalgReducePass.cpp

  DEPENDS
  LowerLinalgReduceTransformPassIncGen

  LINK_LIBS PUBLIC
  MLIRArithDialect
  MLIRFuncDialect
  MLIRIR
  MLIRLinalgDialect
  MLIRMemRefDialect
  MLIRPass
  MLIRSCFDialect
)
