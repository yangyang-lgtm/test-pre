set(LLVM_TARGET_DEFINITIONS PuStreamDialect.td)
mlir_tablegen(PuStreamDialect.h.inc -gen-dialect-decls -dialect=pustream)
mlir_tablegen(PuStreamDialect.cpp.inc -gen-dialect-defs -dialect=pustream)
mlir_tablegen(PuStreamTypes.h.inc -gen-typedef-decls -typedefs-dialect=pustream)
mlir_tablegen(PuStreamTypes.cpp.inc -gen-typedef-defs -typedefs-dialect=pustream)
mlir_tablegen(PuStreamOps.h.inc -gen-op-decls)
mlir_tablegen(PuStreamOps.cpp.inc -gen-op-defs)

add_public_tablegen_target(PuStreamTableGen)