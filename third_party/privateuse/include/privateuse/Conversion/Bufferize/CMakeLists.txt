set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls --name Bufferize)
add_public_tablegen_target(PrivateUseBufferizePassIncGen)
