// RUN: privateuse-opt %s -privateuse-arith-to-linalg 2>&1 | FileCheck %s

// A non-pointer tensor value op outside the conversion whitelist stays legal
// (whole TritonDialect is legal) but the trailing walk warns about it, so the
// gap surfaces here instead of silently at bufferize. The warning is non-fatal.
// CHECK: warning: privateuse-arith-to-linalg: unconverted tensor value op 'tt.trans'
// CHECK: tt.trans
module {
  tt.func @unconverted(%arg: tensor<2x4xf32>) -> tensor<4x2xf32> {
    %t = tt.trans %arg {order = array<i32: 1, 0>} : tensor<2x4xf32> -> tensor<4x2xf32>
    tt.return %t : tensor<4x2xf32>
  }
}