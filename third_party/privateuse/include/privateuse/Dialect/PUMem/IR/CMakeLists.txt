set(LLVM_TARGET_DEFINITIONS PUMemDialect.td)
mlir_tablegen(PUMemDialect.h.inc -gen-dialect-decls -dialect=pumem)
mlir_tablegen(PUMemDialect.cpp.inc -gen-dialect-defs -dialect=pumem)
mlir_tablegen(PUMemTypes.h.inc -gen-typedef-decls -typedefs-dialect=pumem)
mlir_tablegen(PUMemTypes.cpp.inc -gen-typedef-defs -typedefs-dialect=pumem)
mlir_tablegen(PUMemOps.h.inc -gen-op-decls)
mlir_tablegen(PUMemOps.cpp.inc -gen-op-defs)

add_public_tablegen_target(PUMemTableGen)
