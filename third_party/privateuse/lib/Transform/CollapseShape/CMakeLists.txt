add_triton_library(CollapseShape
  CollapseShapePass.cpp

  DEPENDS
  CollapseShapeTransformPassIncGen

  LINK_LIBS PUBLIC
  MLIRAffineDialect
  MLIRIR
  MLIRLinalgDialect
  MLIRMemRefDialect
  MLIRPass
  MLIRSupport
  MLIRTensorDialect
  MLIRTransforms
)
