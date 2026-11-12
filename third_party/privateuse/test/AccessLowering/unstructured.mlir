// RUN: privateuse-opt %s -privateuse-lower-structured-access -privateuse-lower-unstructured-access \
// RUN:   -privateuse-ptr-to-memref -canonicalize | FileCheck %s \
// RUN:   --implicit-check-not=tt.load --implicit-check-not=tt.store --implicit-check-not=tt.splat

module attributes {privateuse.grid = array<i64: 4, 3, 1>} {
  tt.func @transform(%src: !tt.ptr<f32>, %dst: !tt.ptr<f32>, %map: !tt.ptr<i32>) {
    // CHECK-LABEL: tt.func @transform(
    // CHECK-SAME: %arg0: memref<?xf32>, %arg1: memref<?xf32>, %arg2: memref<?xi32>)
    %r = tt.make_range {start = 0 : i32, end = 4 : i32} : tensor<4xi32>
    %mS = tt.splat %map : !tt.ptr<i32> -> tensor<4x!tt.ptr<i32>>
    %mp = tt.addptr %mS, %r : tensor<4x!tt.ptr<i32>>, tensor<4xi32>
    %m = tt.load %mp : tensor<4x!tt.ptr<i32>>
    %pS = tt.splat %src : !tt.ptr<f32> -> tensor<4x!tt.ptr<f32>>
    %p = tt.addptr %pS, %m : tensor<4x!tt.ptr<f32>>, tensor<4xi32>
    %v = tt.load %p : tensor<4x!tt.ptr<f32>>
    %qS = tt.splat %dst : !tt.ptr<f32> -> tensor<4x!tt.ptr<f32>>
    %q = tt.addptr %qS, %m : tensor<4x!tt.ptr<f32>>, tensor<4xi32>
    tt.store %q, %v : tensor<4x!tt.ptr<f32>>
    tt.return
    // CHECK: pumem.gather {{.*}} : memref<?xf32>, memref<4xi32, 128>, memref<4xf32, 128>
    // CHECK: pumem.scatter {{.*}} : memref<4xf32, 128>, memref<4xi32, 128>, memref<?xf32>
  }

  tt.func @scatter_overlap(%dst: !tt.ptr<f32>) {
    // CHECK-LABEL: tt.func @scatter_overlap(
    // CHECK-SAME: %arg0: memref<?xf32>)
    %rows = tt.make_range {start = 0 : i32, end = 2 : i32} : tensor<2xi32>
    %rowsE = tt.expand_dims %rows {axis = 1 : i32} : tensor<2xi32> -> tensor<2x1xi32>
    %rowBc = tt.broadcast %rowsE : tensor<2x1xi32> -> tensor<2x2xi32>
    %colsE = tt.expand_dims %rows {axis = 0 : i32} : tensor<2xi32> -> tensor<1x2xi32>
    %colsBc = tt.broadcast %colsE : tensor<1x2xi32> -> tensor<2x2xi32>
    %idx = arith.addi %rowBc, %colsBc : tensor<2x2xi32>
    %cst = arith.constant 1.000000e+00 : f32
    %v = tt.splat %cst : f32 -> tensor<2x2xf32>
    %dS = tt.splat %dst : !tt.ptr<f32> -> tensor<2x2x!tt.ptr<f32>>
    %d = tt.addptr %dS, %idx : tensor<2x2x!tt.ptr<f32>>, tensor<2x2xi32>
    tt.store %d, %v : tensor<2x2x!tt.ptr<f32>>
    tt.return
    // CHECK: bufferization.to_buffer {{.*}} : tensor<2x2xi32> to memref<2x2xi32, 128>
    // CHECK: pumem.scatter {{.*}} : memref<2x2xf32, 128>, memref<2x2xi32, 128>, memref<?xf32>
  }

  tt.func @broadcast_ptr(%src: !tt.ptr<f32>, %dst: !tt.ptr<f32>, %map: !tt.ptr<i32>) {
    // CHECK-LABEL: tt.func @broadcast_ptr(
    // CHECK-SAME: %arg0: memref<?xf32>, %arg1: memref<?xf32>, %arg2: memref<?xi32>)
    %r = tt.make_range {start = 0 : i32, end = 4 : i32} : tensor<4xi32>
    %mS = tt.splat %map : !tt.ptr<i32> -> tensor<4x!tt.ptr<i32>>
    %mp = tt.addptr %mS, %r : tensor<4x!tt.ptr<i32>>, tensor<4xi32>
    %m = tt.load %mp : tensor<4x!tt.ptr<i32>>
    %pS = tt.splat %src : !tt.ptr<f32> -> tensor<4x!tt.ptr<f32>>
    %p = tt.addptr %pS, %m : tensor<4x!tt.ptr<f32>>, tensor<4xi32>
    %pe = tt.expand_dims %p {axis = 0 : i32} : tensor<4x!tt.ptr<f32>> -> tensor<1x4x!tt.ptr<f32>>
    %pb = tt.broadcast %pe : tensor<1x4x!tt.ptr<f32>> -> tensor<2x4x!tt.ptr<f32>>
    %v = tt.load %pb : tensor<2x4x!tt.ptr<f32>>
    tt.store %pb, %v : tensor<2x4x!tt.ptr<f32>>
    tt.return
    // CHECK: bufferization.to_buffer {{.*}} : tensor<2x4xi32> to memref<2x4xi32, 128>
    // CHECK: pumem.gather {{.*}} : memref<?xf32>, memref<2x4xi32, 128>, memref<2x4xf32, 128>
    // CHECK: pumem.scatter {{.*}} : memref<2x4xf32, 128>, memref<2x4xi32, 128>, memref<?xf32>
  }

  tt.func @wide_offset(%src: !tt.ptr<f32>, %map: !tt.ptr<i64>) {
    // CHECK-LABEL: tt.func @wide_offset(
    // CHECK-SAME: %arg0: memref<?xf32>, %arg1: memref<?xi64>)
    %r = tt.make_range {start = 0 : i32, end = 4 : i32} : tensor<4xi32>
    %mS = tt.splat %map : !tt.ptr<i64> -> tensor<4x!tt.ptr<i64>>
    %mp = tt.addptr %mS, %r : tensor<4x!tt.ptr<i64>>, tensor<4xi32>
    %m = tt.load %mp : tensor<4x!tt.ptr<i64>>
    %pS = tt.splat %src : !tt.ptr<f32> -> tensor<4x!tt.ptr<f32>>
    %p = tt.addptr %pS, %m : tensor<4x!tt.ptr<f32>>, tensor<4xi64>
    %v = tt.load %p : tensor<4x!tt.ptr<f32>>
    tt.store %p, %v : tensor<4x!tt.ptr<f32>>
    tt.return
    // CHECK: pumem.gather {{.*}} : memref<?xf32>, memref<4xi64, 128>, memref<4xf32, 128>
    // CHECK: pumem.scatter {{.*}} : memref<4xf32, 128>, memref<4xi64, 128>, memref<?xf32>
  }

  tt.func @splat_store(%dst: !tt.ptr<f32>) {
    // CHECK-LABEL: tt.func @splat_store(
    // CHECK-SAME: %arg0: memref<?xf32>)
    %cst = arith.constant 1.000000e+00 : f32
    %v = tt.splat %cst : f32 -> tensor<4xf32>
    %dS = tt.splat %dst : !tt.ptr<f32> -> tensor<4x!tt.ptr<f32>>
    tt.store %dS, %v : tensor<4x!tt.ptr<f32>>
    tt.return
    // CHECK: bufferization.to_buffer {{.*}} : tensor<4xi32> to memref<4xi32, 128>
    // CHECK: pumem.scatter {{.*}} : memref<4xf32, 128>, memref<4xi32, 128>, memref<?xf32>
  }

  tt.func @mixed_width(%src: !tt.ptr<f32>, %map32: !tt.ptr<i32>, %map64: !tt.ptr<i64>) {
    // CHECK-LABEL: tt.func @mixed_width(
    // CHECK-SAME: %arg0: memref<?xf32>, %arg1: memref<?xi32>, %arg2: memref<?xi64>)
    %r = tt.make_range {start = 0 : i32, end = 4 : i32} : tensor<4xi32>
    %mS = tt.splat %map32 : !tt.ptr<i32> -> tensor<4x!tt.ptr<i32>>
    %mp = tt.addptr %mS, %r : tensor<4x!tt.ptr<i32>>, tensor<4xi32>
    %m = tt.load %mp : tensor<4x!tt.ptr<i32>>
    %nS = tt.splat %map64 : !tt.ptr<i64> -> tensor<4x!tt.ptr<i64>>
    %np = tt.addptr %nS, %r : tensor<4x!tt.ptr<i64>>, tensor<4xi32>
    %n = tt.load %np : tensor<4x!tt.ptr<i64>>
    %pS = tt.splat %src : !tt.ptr<f32> -> tensor<4x!tt.ptr<f32>>
    %p = tt.addptr %pS, %m : tensor<4x!tt.ptr<f32>>, tensor<4xi32>
    %q = tt.addptr %p, %n : tensor<4x!tt.ptr<f32>>, tensor<4xi64>
    %v = tt.load %q : tensor<4x!tt.ptr<f32>>
    tt.store %q, %v : tensor<4x!tt.ptr<f32>>
    tt.return
    // CHECK: arith.extsi {{.*}} : tensor<4xi32> to tensor<4xi64>
    // CHECK: bufferization.to_buffer {{.*}} : tensor<4xi64> to memref<4xi64, 128>
    // CHECK: pumem.gather {{.*}} : memref<?xf32>, memref<4xi64, 128>, memref<4xf32, 128>
    // CHECK: pumem.scatter {{.*}} : memref<4xf32, 128>, memref<4xi64, 128>, memref<?xf32>
  }
}
