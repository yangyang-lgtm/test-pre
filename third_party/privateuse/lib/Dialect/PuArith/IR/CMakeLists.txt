add_triton_library(PuArithIR
  PuArithOps.cpp
  PuArithDialect.cpp

  DEPENDS
  PuArithTableGen
  PuArithOpsInterfacesIncGen
  PuArithCanonicalizationIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRSupport
  MLIRCastInterfaces
  MLIRInferTypeOpInterface
  MLIRTransformUtils
)
