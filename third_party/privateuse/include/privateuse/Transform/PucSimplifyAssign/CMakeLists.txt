set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls --name PucSimplifyAssign)
add_public_tablegen_target(PucSimplifyAssignTransformPassIncGen)
