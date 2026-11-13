// RUN: privateuse-opt %s -privateuse-arith-to-linalg | FileCheck %s \
// RUN:   --implicit-check-not=tt.make_range --implicit-check-not=tt.splat

// The unstructured access path wraps index/value tensors in bufferization.to_buffer
// but leaves make_range / dense constant / subi as tensor ops; this pass converts them.
module {
  // CHECK-LABEL: tt.func @index_form(
  tt.func @index_form() {
    %r = tt.make_range {start = 0 : i32, end = 4 : i32} : tensor<4xi32>
    %zero = arith.constant dense<0> : tensor<4xi32>
    %neg = arith.subi %zero, %r : tensor<4xi32>
    %idx = bufferization.to_buffer %neg : tensor<4xi32> to memref<4xi32, 128>
    %dst = memref.alloc() : memref<4xi32, 128>
    memref.copy %idx, %dst : memref<4xi32, 128> to memref<4xi32, 128>
    tt.return
  }
  // CHECK: linalg.generic
  // CHECK: linalg.fill
  // CHECK: linalg.generic
  // CHECK: bufferization.to_buffer
  // CHECK: memref.copy
}