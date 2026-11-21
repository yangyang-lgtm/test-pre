set(LLVM_TARGET_DEFINITIONS PucDialect.td)
mlir_tablegen(PucDialect.h.inc -gen-dialect-decls -dialect=puc)
mlir_tablegen(PucDialect.cpp.inc -gen-dialect-defs -dialect=puc)
mlir_tablegen(PucOps.h.inc -gen-op-decls)
mlir_tablegen(PucOps.cpp.inc -gen-op-defs)
mlir_tablegen(PucTypes.h.inc -gen-typedef-decls -typedefs-dialect=puc)
mlir_tablegen(PucTypes.cpp.inc -gen-typedef-defs -typedefs-dialect=puc)

add_public_tablegen_target(PucTableGen)
