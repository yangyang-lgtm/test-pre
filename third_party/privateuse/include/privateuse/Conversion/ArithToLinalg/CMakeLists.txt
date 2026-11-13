set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls --name ArithToLinalg)
add_public_tablegen_target(PrivateUseArithToLinalgPassIncGen)