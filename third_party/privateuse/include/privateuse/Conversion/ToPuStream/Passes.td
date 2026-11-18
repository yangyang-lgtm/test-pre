#ifndef PRIVATEUSE_CONVERSION_TOPUSTREAM_PASSES_TD
#define PRIVATEUSE_CONVERSION_TOPUSTREAM_PASSES_TD

include "mlir/Pass/PassBase.td"

def ToPuStream : Pass<"privateuse-to-pustream", "mlir::ModuleOp"> {
  let summary = "Lower runtime linalg ops and cross-space copies to pustream async ops";
  let description = [{
    Statically shaped `linalg.matmul` and `linalg.transpose` become
    `pustream.async.dispatch` calls to the `runtime_matmul` and
    `runtime_transpose` runtime functions. Every `memref.copy` whose source and
    target live in different memory spaces becomes a `pustream.async.copy`;
    shape, strides, element type and dynamic offsets all come from the memref
    operands, and copies that stay inside one memory space are left untouched.

    The returned timepoints are not consumed yet; a later scheduling pass
    attaches them to the awaits of dependent operations.
  }];
  let dependentDialects = [
    "::mlir::arith::ArithDialect",
    "::mlir::linalg::LinalgDialect",
    "::mlir::memref::MemRefDialect",
    "::mlir::pustream::PuStreamDialect"
  ];
  let constructor = "mlir::triton::createToPuStreamPass()";
}

#endif
