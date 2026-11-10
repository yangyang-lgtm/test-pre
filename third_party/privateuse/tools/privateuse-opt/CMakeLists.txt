get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
get_property(triton_libs GLOBAL PROPERTY TRITON_LIBS)

add_llvm_executable(privateuse-opt privateuse-opt.cpp PARTIAL_SOURCES_INTENDED)

llvm_update_compile_flags(privateuse-opt)
target_link_libraries(privateuse-opt PRIVATE
  TritonTransforms
  ${dialect_libs}
  ${conversion_libs}
  ${triton_libs}
  MLIROptLib
  MLIRPass
  MLIRRegisterAllPasses
  MLIRTransforms
)

mlir_check_all_link_libraries(privateuse-opt)

add_llvm_executable(privateuse-lsp privateuse-lsp.cpp PARTIAL_SOURCES_INTENDED)

llvm_update_compile_flags(privateuse-lsp)
target_link_libraries(privateuse-lsp PRIVATE
  TritonTransforms
  ${dialect_libs}
  ${conversion_libs}
  ${triton_libs}
  TritonTestAnalysis
  TritonTestDialect
  TritonAMDGPUTestAnalysis
  TritonTestProton
  MLIRLspServerLib
  MLIRPass
  MLIRRegisterAllDialects
  MLIRRegisterAllPasses
  MLIRTransforms
)

mlir_check_all_link_libraries(privateuse-lsp)
