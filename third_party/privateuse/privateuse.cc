#include "privateuse/Conversion/AccessLowering/Passes.h"
#include "privateuse/Conversion/ArithToLinalg/Passes.h"
#include "privateuse/Conversion/Bufferize/Passes.h"
#include "privateuse/Dialect/PUMem/IR/PUMemDialect.h"
#include "privateuse/Transform/PidLoop/Passes.h"
#include "privateuse/Transform/SpecializeKernel/SpecializeKernel.h"
#include "privateuse/Utils/AccessAnalysis/Passes.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Bufferization/Transforms/FuncBufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Bufferization/Transforms/Passes.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/ControlFlow/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Func/Extensions/InlinerExtension.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tensor/IR/TensorInferTypeOpInterfaceImpl.h"
#include "mlir/Dialect/Tensor/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Pass/PassManager.h"
#include "passes.h"
#include "triton/Dialect/Triton/IR/Dialect.h"

#include <nanobind/nanobind.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/variant.h>
#include <nanobind/stl/vector.h>

#include <cstdio>
#include <stdexcept>

namespace py = nanobind;
using namespace mlir;

void init_triton_privateuse(py::module_& m)
{
  m.doc() = "Python bindings to the privateuse backend";

  m.def("load_dialects", [](mlir::MLIRContext& context) {
    mlir::DialectRegistry registry;
    // clang-format off
    registry.insert<
        mlir::pumem::PUMemDialect,
        mlir::triton::TritonDialect,
        mlir::func::FuncDialect,
        mlir::memref::MemRefDialect,
        mlir::linalg::LinalgDialect,
        mlir::tensor::TensorDialect,
        mlir::scf::SCFDialect,
        mlir::arith::ArithDialect,
        mlir::affine::AffineDialect,
        mlir::vector::VectorDialect,
        mlir::math::MathDialect,
        mlir::cf::ControlFlowDialect,
        mlir::bufferization::BufferizationDialect
    >();
    // clang-format on

    mlir::func::registerInlinerExtension(registry);

    mlir::arith::registerBufferizableOpInterfaceExternalModels(registry);
    mlir::bufferization::func_ext::registerBufferizableOpInterfaceExternalModels(registry);
    mlir::cf::registerBufferizableOpInterfaceExternalModels(registry);
    mlir::linalg::registerBufferizableOpInterfaceExternalModels(registry);
    mlir::scf::registerBufferizableOpInterfaceExternalModels(registry);
    mlir::tensor::registerBufferizableOpInterfaceExternalModels(registry);
    mlir::tensor::registerInferTypeOpInterfaceExternalModels(registry);

    context.appendDialectRegistry(registry);
    context.loadAllAvailableDialects();
  });

  m.def(
    "add_specialize_kernel",
    [](PassManager& pm, std::vector<triton::SpecializeConstant> constants, std::vector<int64_t> grid) {
      pm.addPass(triton::createSpecializeKernelPass(std::move(constants), std::move(grid)));
    },
    py::arg("pm"), py::arg("constants"), py::arg("grid")
  );

  m.def("add_analyze_access", [](PassManager& pm) { pm.addPass(triton::createAnalyzeAccessPass()); }, py::arg("pm"));

  m.def(
    "add_pid_loop", [](PassManager& pm, int64_t numPids) { pm.addPass(triton::createPidLoopPass(numPids)); },
    py::arg("pm"), py::arg("num_pids")
  );

  m.def(
    "add_empty_tensor_to_alloc_tensor",
    [](PassManager& pm) { pm.addPass(bufferization::createEmptyTensorToAllocTensorPass()); }, py::arg("pm")
  );

  m.def(
    "add_lower_structured_access", [](PassManager& pm) { pm.addPass(triton::createLowerStructuredAccessPass()); },
    py::arg("pm")
  );

  m.def(
    "add_lower_unstructured_access", [](PassManager& pm) { pm.addPass(triton::createLowerUnstructuredAccessPass()); },
    py::arg("pm")
  );

  m.def("add_ptr_to_memref", [](PassManager& pm) { pm.addPass(triton::createPtrToMemrefPass()); }, py::arg("pm"));

  m.def("add_arith_to_linalg", [](PassManager& pm) { pm.addPass(triton::createArithToLinalgPass()); }, py::arg("pm"));

  m.def("add_bufferize", [](PassManager& pm) { pm.addPass(triton::createBufferizePass()); }, py::arg("pm"));
}
