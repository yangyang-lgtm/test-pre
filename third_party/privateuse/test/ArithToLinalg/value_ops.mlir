// RUN: privateuse-opt %s -privateuse-arith-to-linalg | FileCheck %s \
// RUN:   --implicit-check-not=tt.splat --implicit-check-not=tt.make_range \
// RUN:   --implicit-check-not=tt.broadcast --implicit-check-not=tt.expand_dims \
// RUN:   --implicit-check-not=tt.reshape --implicit-check-not=tt.bitcast --implicit-check-not=tt.reduce \
// RUN:   --implicit-check-not=tt.dot --implicit-check-not=tt.trans

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

  // CHECK-LABEL: tt.func @reduce_full(
  tt.func @reduce_full(%x: tensor<64xf32>) -> f32 {
    %r = "tt.reduce"(%x) <{axis = 0 : i32}> ({
    ^bb0(%a: f32, %b: f32):
      %s = arith.addf %a, %b : f32
      tt.reduce.return %s : f32
    }) : (tensor<64xf32>) -> f32
    tt.return %r : f32
  }
  // Full (rank-1) reduce: rank-0 init + reduction iterator, then scalar extract.
  // CHECK: tensor.empty() : tensor<f32>
  // CHECK: linalg.generic
  // CHECK-SAME: iterator_types = ["reduction"]
  // CHECK-SAME: outs(%{{.+}} : tensor<f32>)
  // CHECK: arith.addf
  // CHECK: tensor.extract

  // CHECK-LABEL: tt.func @reduce_axis1(
  tt.func @reduce_axis1(%x: tensor<8x16xf32>) -> tensor<8xf32> {
    %r = "tt.reduce"(%x) <{axis = 1 : i32}> ({
    ^bb0(%a: f32, %b: f32):
      %s = arith.maximumf %a, %b : f32
      tt.reduce.return %s : f32
    }) : (tensor<8x16xf32>) -> tensor<8xf32>
    tt.return %r : tensor<8xf32>
  }
  // Non-full reduce keeps the parallel dim; combine body is cloned into the region.
  // CHECK: tensor.empty() : tensor<8xf32>
  // CHECK: linalg.generic
  // CHECK-SAME: iterator_types = ["parallel", "reduction"]
  // CHECK: arith.maximumf

  // CHECK-LABEL: tt.func @dot_zero(
  tt.func @dot_zero(%a: tensor<16x8xf32>, %b: tensor<8x16xf32>) -> tensor<16x16xf32> {
    %z = arith.constant dense<0.000000e+00> : tensor<16x16xf32>
    %d = tt.dot %a, %b, %z : tensor<16x8xf32> * tensor<8x16xf32> -> tensor<16x16xf32>
    tt.return %d : tensor<16x16xf32>
  }
  // Zero accumulator is detected and dropped: plain matmul, no add-back.
  // CHECK: linalg.matmul
  // CHECK-NOT: arith.addf
  // CHECK: tt.return

  // CHECK-LABEL: tt.func @dot_acc(
  tt.func @dot_acc(%a: tensor<16x8xf32>, %b: tensor<8x16xf32>, %c: tensor<16x16xf32>) -> tensor<16x16xf32> {
    %d = tt.dot %a, %b, %c : tensor<16x8xf32> * tensor<8x16xf32> -> tensor<16x16xf32>
    tt.return %d : tensor<16x16xf32>
  }
  // Non-zero accumulator: matmul then add-C back.
  // CHECK: linalg.matmul
  // CHECK: linalg.generic
  // CHECK: arith.addf

  // CHECK-LABEL: tt.func @dot_batch(
  tt.func @dot_batch(%a: tensor<4x16x8xf32>, %b: tensor<4x8x16xf32>, %c: tensor<4x16x16xf32>) -> tensor<4x16x16xf32> {
    %d = tt.dot %a, %b, %c : tensor<4x16x8xf32> * tensor<4x8x16xf32> -> tensor<4x16x16xf32>
    tt.return %d : tensor<4x16x16xf32>
  }
  // CHECK: linalg.batch_matmul

  // CHECK-LABEL: tt.func @trans2d(
  tt.func @trans2d(%x: tensor<4x8xf32>) -> tensor<8x4xf32> {
    %t = tt.trans %x {order = array<i32: 1, 0>} : tensor<4x8xf32> -> tensor<8x4xf32>
    tt.return %t : tensor<8x4xf32>
  }
  // CHECK: linalg.transpose
  // CHECK-SAME: permutation = [1, 0]

  // CHECK-LABEL: tt.func @permute3d(
  tt.func @permute3d(%x: tensor<1x2x4xi32>) -> tensor<4x1x2xi32> {
    %t = tt.trans %x {order = array<i32: 2, 0, 1>} : tensor<1x2x4xi32> -> tensor<4x1x2xi32>
    tt.return %t : tensor<4x1x2xi32>
  }
  // CHECK: linalg.transpose
  // CHECK-SAME: permutation = [2, 0, 1]
}