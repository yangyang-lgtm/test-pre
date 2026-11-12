// RUN: privateuse-opt %s -privateuse-lower-structured-access -privateuse-ptr-to-memref | FileCheck %s \
// RUN:   --implicit-check-not=memref.

module attributes {privateuse.grid = array<i64: 4, 1, 1>} {
  tt.func private @add_one(%p: !tt.ptr<f32>) {
    %r = tt.make_range {start = 0 : i32, end = 128 : i32} : tensor<128xi32>
    %pS = tt.splat %p : !tt.ptr<f32> -> tensor<128x!tt.ptr<f32>>
    %a = tt.addptr %pS, %r : tensor<128x!tt.ptr<f32>>, tensor<128xi32>
    %v = tt.load %a : tensor<128x!tt.ptr<f32>>
    %c = arith.constant 1.000000e+00 : f32
    %cS = tt.splat %c : f32 -> tensor<128xf32>
    %w = arith.addf %v, %cS : tensor<128xf32>
    tt.store %a, %w : tensor<128x!tt.ptr<f32>>
    tt.return
  }

  tt.func @kernel(%p: !tt.ptr<f32>) {
    tt.call @add_one(%p) : (!tt.ptr<f32>) -> ()
    tt.return
  }
}

// CHECK-LABEL: tt.func private @add_one(
// CHECK-SAME: %arg0: !tt.ptr<f32>
// CHECK: tt.load
// CHECK: tt.store
// CHECK: tt.return

// CHECK-LABEL: tt.func @kernel(
// CHECK-SAME: %arg0: !tt.ptr<f32>
// CHECK: tt.call @add_one
// CHECK: tt.return
