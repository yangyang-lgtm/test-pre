#ifndef PRIVATEUSE_TRANSFORM_PUCLAUNCHKERNEL_PASSES_TD
#define PRIVATEUSE_TRANSFORM_PUCLAUNCHKERNEL_PASSES_TD

include "mlir/Pass/PassBase.td"

def PucLaunchKernel : Pass<"privateuse-puc-launch-kernel", "mlir::ModuleOp"> {
  let summary = "Add an extern C launcher that forwards to each kernel entry";

  let description = [{
    Runs right before the C++ translation. For every function that is not
    called by any other function (the kernel entry) this pass adds a
    <kernel>_launcher wrapper marked extern "C" whose body is a single
    puc.launch of the kernel, and marks the kernel itself GLOBAL. The
    launcher therefore owns the exported symbol while the kernel keeps its
    own calling convention.
  }];

  let options = [
    Option<"launcherAttrValue", "launcher-attr-value", "std::string",
           "\"extern \\\"C\\\"\"",
           "Value of the pumem.outlined attribute on the launcher wrapper">,
    Option<"kernelAttrValue", "kernel-attr-value", "std::string", "\"GLOBAL\"",
           "Value of the pumem.outlined attribute on the kernel entry">
  ];

  let constructor = "mlir::triton::createPucLaunchKernelPass()";
}

#endif
