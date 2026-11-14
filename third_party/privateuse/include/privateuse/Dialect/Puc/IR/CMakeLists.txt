set(LLVM_TARGET_DEFINITIONS PucDialect.td)
mlir_tablegen(PucDialect.h.inc -gen-dialect-decls -dialect=puc)
mlir_tablegen(PucDialect.cpp.inc -gen-dialect-defs -dialect=puc)
mlir_tablegen(PucOps.h.inc -gen-op-decls)
mlir_tablegen(PucOps.cpp.inc -gen-op-defs)

add_public_tablegen_target(PucTableGen)
