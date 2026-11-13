#ifndef PRIVATEUSE_CONVERSION_ARITHTOLINALG_PASSES_TD
#define PRIVATEUSE_CONVERSION_ARITHTOLINALG_PASSES_TD

include "mlir/Pass/PassBase.td"

def ArithToLinalg : Pass<"privateuse-arith-to-linalg", "mlir::ModuleOp"> {
  let summary = "Convert residual tensor value ops to linalg/tensor";
  let description = [{
    Runs after the access lowering pipeline (LowerStructuredAccess /
    LowerUnstructuredAccess / PtrToMemref) and converts the tensor value
    operations that remain:

    * elementwise arith / math on ranked tensors become linalg.generic via the
      upstream elementwise-to-linalg helper;
    * dense splat arith.constant becomes a scalar constant plus tensor.splat;
    * value tt.splat becomes tensor.splat, tt.make_range becomes a linalg.generic
      with linalg.index, tt.broadcast becomes a broadcast linalg.generic,
      tt.expand_dims becomes tensor.expand_shape, tt.reshape becomes
      tensor.collapse_shape / tensor.expand_shape / tensor.reshape, and value
      tt.bitcast becomes arith.bitcast.

    Pointer-valued operations are preserved: tt.splat / tt.broadcast /
    tt.expand_dims / tt.reshape / tt.bitcast whose result is a pointer or a
    tensor of pointers stay legal, and unrealized_conversion_cast bridges are
    kept, so the address chains retained by PtrToMemref in soft-skipped
    functions are untouched. Modules containing tt.call are skipped.

    A trailing walk warns about non-pointer tensor-valued Triton ops that no
    converter handles, so silent gaps surface early instead of at bufferize.
  }];
  let constructor = "mlir::triton::createArithToLinalgPass()";
}

#endif
