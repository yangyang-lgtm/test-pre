add_triton_library(PucVoidPtrArgs
  PucVoidPtrArgsPass.cpp

  DEPENDS
  PucVoidPtrArgsTransformPassIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRPass
  MLIRSupport
  PUMemIR
  PucIR
)
