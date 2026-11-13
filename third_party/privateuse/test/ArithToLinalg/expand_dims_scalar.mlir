// RUN: privateuse-opt %s -privateuse-arith-to-linalg | FileCheck %s \
// RUN:   --implicit-check-not=tt.expand_dims

module {
  // CHECK-LABEL: tt.func @expand_dims_scalar(
  tt.func @expand_dims_scalar(%arg: tensor<f32>) -> tensor<1xf32> {
    %e = tt.expand_dims %arg {axis = 0 : i32} : tensor<f32> -> tensor<1xf32>
    tt.return %e : tensor<1xf32>
  }
  // CHECK: tensor.expand_shape
}