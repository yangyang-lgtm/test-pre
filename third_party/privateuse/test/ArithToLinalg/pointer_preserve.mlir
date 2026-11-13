// RUN: privateuse-opt %s -privateuse-arith-to-linalg | FileCheck %s

module {
  // make_range is the only value op that changes inside a preserved pointer chain;
  // the pointer tt.splat / tt.addptr stay untouched.
  // CHECK-LABEL: tt.func @keep_chain(
  tt.func @keep_chain(%src: !tt.ptr<f32>) -> tensor<8x!tt.ptr<f32>> {
    %r = tt.make_range {start = 0 : i32, end = 8 : i32} : tensor<8xi32>
    %pS = tt.splat %src : !tt.ptr<f32> -> tensor<8x!tt.ptr<f32>>
    %p = tt.addptr %pS, %r : tensor<8x!tt.ptr<f32>>, tensor<8xi32>
    tt.return %p : tensor<8x!tt.ptr<f32>>
  }
  // CHECK: linalg.generic
  // CHECK: tt.splat
  // CHECK: tt.addptr
  // CHECK: tt.return

  // A pointer-result tt.bitcast is preserved (reinterpret to ptr).
  // CHECK-LABEL: tt.func @keep_ptr_bitcast(
  tt.func @keep_ptr_bitcast(%p: tensor<8x!tt.ptr<i1>>) -> tensor<8x!tt.ptr<i8>> {
    %b = tt.bitcast %p : tensor<8x!tt.ptr<i1>> -> tensor<8x!tt.ptr<i8>>
    tt.return %b : tensor<8x!tt.ptr<i8>>
  }
  // CHECK: tt.bitcast
  // CHECK: tt.return
}