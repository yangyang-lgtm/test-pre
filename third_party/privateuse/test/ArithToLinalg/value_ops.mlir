// RUN: privateuse-opt %s -privateuse-arith-to-linalg | FileCheck %s \
// RUN:   --implicit-check-not=tt.splat --implicit-check-not=tt.make_range \
// RUN:   --implicit-check-not=tt.broadcast --implicit-check-not=tt.expand_dims \
// RUN:   --implicit-check-not=tt.reshape --implicit-check-not=tt.bitcast

module {
  // CHECK-LABEL: tt.func @make_range_zero(
  tt.func @make_range_zero() -> tensor<8xi32> {
    %r = tt.make_range {start = 0 : i32, end = 8 : i32} : tensor<8xi32>
    tt.return %r : tensor<8xi32>
  }
  // CHECK: linalg.generic
  // CHECK: linalg.index
  // CHECK: arith.index_cast

  // CHECK-LABEL: tt.func @make_range_offset(
  tt.func @make_range_offset() -> tensor<8xi32> {
    %r = tt.make_range {start = 5 : i32, end = 13 : i32} : tensor<8xi32>
    tt.return %r : tensor<8xi32>
  }
  // CHECK: linalg.generic
  // CHECK: arith.addi

  // CHECK-LABEL: tt.func @make_range_size1(
  tt.func @make_range_size1() -> tensor<1xi32> {
    %r = tt.make_range {start = 5 : i32, end = 6 : i32} : tensor<1xi32>
    tt.return %r : tensor<1xi32>
  }
  // size-1 make_range folds to a dense splat constant, then to scalar + linalg.fill.
  // CHECK: linalg.fill

  // CHECK-LABEL: tt.func @splat_f32(
  tt.func @splat_f32(%x: f32) -> tensor<4xf32> {
    %s = tt.splat %x : f32 -> tensor<4xf32>
    tt.return %s : tensor<4xf32>
  }
  // CHECK: linalg.fill

  // CHECK-LABEL: tt.func @splat_i1(
  tt.func @splat_i1(%x: i1) -> tensor<4xi1> {
    %s = tt.splat %x : i1 -> tensor<4xi1>
    tt.return %s : tensor<4xi1>
  }
  // CHECK: linalg.fill

  // CHECK-LABEL: tt.func @dense_const(
  tt.func @dense_const() -> tensor<4xf32> {
    %c = arith.constant dense<2.000000e+00> : tensor<4xf32>
    tt.return %c : tensor<4xf32>
  }
  // CHECK: linalg.fill

  // CHECK-LABEL: tt.func @broadcast_1d(
  tt.func @broadcast_1d(%arg: tensor<4x1xf32>) -> tensor<4x4xf32> {
    %b = tt.broadcast %arg : tensor<4x1xf32> -> tensor<4x4xf32>
    tt.return %b : tensor<4x4xf32>
  }
  // CHECK: linalg.generic

  // CHECK-LABEL: tt.func @broadcast_same(
  tt.func @broadcast_same(%arg: tensor<4x4xf32>) -> tensor<4x4xf32> {
    %b = tt.broadcast %arg : tensor<4x4xf32> -> tensor<4x4xf32>
    tt.return %b : tensor<4x4xf32>
  }
  // Same-shape broadcast folds to identity: no fatal, no linalg.generic.
  // CHECK: tt.return %arg0 : tensor<4x4xf32>

  // CHECK-LABEL: tt.func @expand_dims(
  tt.func @expand_dims(%arg: tensor<4xi32>) -> tensor<4x1xi32> {
    %e = tt.expand_dims %arg {axis = 1 : i32} : tensor<4xi32> -> tensor<4x1xi32>
    tt.return %e : tensor<4x1xi32>
  }
  // CHECK: tensor.expand_shape

  // CHECK-LABEL: tt.func @reshape_collapse(
  tt.func @reshape_collapse(%arg: tensor<2x4xi32>) -> tensor<8xi32> {
    %r = tt.reshape %arg : tensor<2x4xi32> -> tensor<8xi32>
    tt.return %r : tensor<8xi32>
  }
  // CHECK: tensor.collapse_shape

  // CHECK-LABEL: tt.func @reshape_nonassoc(
  tt.func @reshape_nonassoc(%arg: tensor<2x4xi32>) -> tensor<4x2xi32> {
    %r = tt.reshape %arg : tensor<2x4xi32> -> tensor<4x2xi32>
    tt.return %r : tensor<4x2xi32>
  }
  // CHECK: tensor.reshape

  // CHECK-LABEL: tt.func @bitcast_value(
  tt.func @bitcast_value(%arg: tensor<4xi32>) -> tensor<4xf32> {
    %b = tt.bitcast %arg : tensor<4xi32> -> tensor<4xf32>
    tt.return %b : tensor<4xf32>
  }
  // CHECK: arith.bitcast

  // CHECK-LABEL: tt.func @elementwise(
  tt.func @elementwise(%a: tensor<4xf32>, %b: tensor<4xf32>) -> tensor<4xf32> {
    %m = arith.mulf %a, %b : tensor<4xf32>
    tt.return %m : tensor<4xf32>
  }
  // CHECK: linalg.generic
  // CHECK: arith.mulf
}