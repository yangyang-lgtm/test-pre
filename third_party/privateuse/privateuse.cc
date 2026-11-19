#include "privateuse/Conversion/AccessLowering/Passes.h"
#include "privateuse/Conversion/ArithToLinalg/Passes.h"
#include "privateuse/Conversion/Bufferize/Passes.h"
#include "privateuse/Conversion/MemrefToPumem/Passes.h"
#include "privateuse/Conversion/ToPuc/Passes.h"
#include "privateuse/Conversion/MathToPumath/MathToPumath.h"
#include "privateuse/Conversion/ArithToPuarith/ArithToPuarith.h"
#include "privateuse/Conversion/ScfToPumem/ScfToPumem.h"
#include "privateuse/Dialect/PUMem/IR/PUMemDialect.h"
#include "privateuse/Dialect/Puc/IR/PucDialect.h"
#include "privateuse/Transform/CollapseShape/Passes.h"
#include "privateuse/Transform/LowerLinalgReduce/Passes.h"
#include "privateuse/Transform/PidLoop/Passes.h"
#include "privateuse/Transform/PucLaunchKernel/Passes.h"
#include "privateuse/Transform/PucLocalizeCarried/PucLocalizeCarried.h"
#include "privateuse/Transform/PucSimplifyAssign/PucSimplifyAssign.h"
#include "privateuse/Transform/PucVoidPtrArgs/PucVoidPtrArgs.h"
#include "privateuse/Transform/SinkConstantCallArgs/Passes.h"
#include "privateuse/Transform/SpecializeKernel/SpecializeKernel.h"
#include "privateuse/Utils/AccessAnalysis/Passes.h"
#include "privateuse/PucTranslate/PucTranslate.h"

#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Conversion/ReconcileUnrealizedCasts/ReconcileUnrealizedCasts.h"
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
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/Linalg/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/MemRef/Transforms/Passes.h"
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
        mlir::puc::PucDialect,
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

  m.def("add_collapse_shape", [](PassManager& pm) { pm.addPass(triton::createCollapseShapePass()); }, py::arg("pm"));

  m.def("add_arith_to_puarith", [](PassManager& pm) { pm.addPass(triton::createArithToPuarithPass()); }, py::arg("pm"));

  m.def("add_math_to_pumath", [](PassManager& pm) { pm.addPass(triton::createMathToPumathPass()); }, py::arg("pm"));

  m.def("add_scf_to_pumem", [](PassManager& pm) { pm.addPass(triton::createScfToPumemPass()); }, py::arg("pm"));





  m.def(
    "add_lower_linalg_reduce", [](PassManager& pm) { pm.addPass(triton::createLowerLinalgReducePass()); }, py::arg("pm")
  );

  m.def(
    "add_convert_linalg_to_loops", [](PassManager& pm) { pm.addPass(mlir::createConvertLinalgToLoopsPass()); },
    py::arg("pm")
  );

  m.def(
    "add_sink_constant_call_args", [](PassManager& pm) { pm.addPass(triton::createSinkConstantCallArgsPass()); },
    py::arg("pm")
  );

  m.def("add_bufferize", [](PassManager& pm) { pm.addPass(triton::createBufferizePass()); }, py::arg("pm"));

  m.def(
    "add_expand_strided_metadata", [](PassManager& pm) { pm.addPass(memref::createExpandStridedMetadataPass()); },
    py::arg("pm")
  );

  m.def("add_memref_expand", [](PassManager& pm) { pm.addPass(memref::createExpandOpsPass()); }, py::arg("pm"));

  m.def("add_lower_affine", [](PassManager& pm) { pm.addPass(mlir::createLowerAffinePass()); }, py::arg("pm"));

  m.def("add_memref_to_pumem", [](PassManager& pm) { pm.addPass(triton::createMemrefToPumemPass()); }, py::arg("pm"));

  m.def(
    "add_puc_launch",
    [](PassManager& pm, const std::string& launcherAttr, const std::string& kernelAttr) {
      pm.addPass(triton::createPucLaunchKernelPass(launcherAttr, kernelAttr));
    },
    py::arg("pm"), py::arg("launcher_attr") = "extern \"C\"", py::arg("kernel_attr") = "GLOBAL"
  );

  m.def(
    "add_puc_localize_carried", [](PassManager& pm) { pm.addPass(triton::createPucLocalizeCarriedPass()); },
    py::arg("pm")
  );

  m.def(
    "add_puc_simplify_assign", [](PassManager& pm) { pm.addPass(triton::createPucSimplifyAssignPass()); }, py::arg("pm")
  );

  m.def(
    "add_puc_void_ptr_args", [](PassManager& pm) { pm.addPass(triton::createPucVoidPtrArgsPass()); }, py::arg("pm")
  );

  m.def(
    "add_reconcile_unrealized_casts", [](PassManager& pm) { pm.addPass(mlir::createReconcileUnrealizedCastsPass()); },
    py::arg("pm")
  );

  m.def("add_to_puc", [](PassManager& pm) { pm.addPass(triton::createToPucPass()); }, py::arg("pm"));

  m.def(
    "translate_puc_to_cpp",
    [](ModuleOp& mod, const std::string& backend, int64_t numPids) -> std::string {
      std::string out;
      if (!puc::translatePucToCppString(mod.getOperation(), out, backend, numPids))
        throw std::runtime_error("puc to cpp translation failed");
      return out;
    },
    py::arg("mod"), py::arg("backend") = "cpu", py::arg("num_pids") = 32
  );
}
