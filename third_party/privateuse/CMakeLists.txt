include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
include_directories(${CMAKE_CURRENT_BINARY_DIR}/include)

add_subdirectory(include)
add_subdirectory(lib)
add_subdirectory(tools/privateuse-opt)

get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
get_property(extension_libs GLOBAL PROPERTY MLIR_EXTENSION_LIBS)

add_triton_plugin(TritonPrivateUse ${CMAKE_CURRENT_SOURCE_DIR}/privateuse.cc LINK_LIBS
  PrivateUseTransforms
  PrivateUseUtils
  PUMemIR
  ${conversion_libs}
  ${extension_libs}
  MLIRSparseTensorTransforms
  MLIRControlFlowTransforms
  MLIRTensorInferTypeOpInterfaceImpl
  MLIRArithValueBoundsOpInterfaceImpl
  MLIRMemRefDialect
  MLIRSCFDialect
  MLIRLinalgDialect
  MLIRTensorDialect
  MLIRAffineDialect
)
target_link_libraries(TritonPrivateUse PRIVATE Python3::Module nanobind-static)
