add_triton_library(PrivateUseUtils
  Utils.cpp

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRSupport
  MLIRLinalgDialect
  TritonIR
)

# runWithRestore reports rewrite failures by catching RewriteError.
if(NOT MSVC)
  target_compile_options(PrivateUseUtils PRIVATE -fexceptions)
endif()

add_subdirectory(AccessAnalysis)
