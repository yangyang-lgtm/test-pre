// RUN: privateuse-opt %s -empty-tensor-to-alloc-tensor -split-input-file | FileCheck %s

// An empty tensor whose only user writes it lowers to an alloc_tensor in
// place: no tensor.empty survives.

// CHECK-LABEL: tt.func @kernel
// CHECK-NOT: tensor.empty
// CHECK: %[[T:.*]] = bufferization.alloc_tensor() : tensor<16xf32>
// CHECK: linalg.fill ins({{.*}} : f32) outs(%[[T]] : tensor<16xf32>) -> tensor<16xf32>
// CHECK: tt.return
module {
  tt.func @kernel() -> tensor<16xf32> {
    %0 = tensor.empty() : tensor<16xf32>
    %c1 = arith.constant 1.000000e+00 : f32
    %1 = linalg.fill ins(%c1 : f32) outs(%0 : tensor<16xf32>) -> tensor<16xf32>
    tt.return %1 : tensor<16xf32>
  }
}
