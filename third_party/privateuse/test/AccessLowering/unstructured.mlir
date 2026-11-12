// RUN: privateuse-opt %s -privateuse-lower-structured-access -privateuse-lower-unstructured-access \
// RUN:   -privateuse-ptr-to-memref -canonicalize | FileCheck %s \
// RUN:   --implicit-check-not=tt.load --implicit-check-not=tt.store --implicit-check-not=tt.splat \
// RUN:   --implicit-check-not=tt.bitcast --implicit-check-not=linalg.

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

  tt.func @bool_chain(%src: !tt.ptr<i1>, %dst: !tt.ptr<i1>, %map: !tt.ptr<i32>) {
    // CHECK-LABEL: tt.func @bool_chain(
    // CHECK-SAME: %arg0: memref<?xi8>, %arg1: memref<?xi8>, %arg2: memref<?xi32>)
    %r = tt.make_range {start = 0 : i32, end = 4 : i32} : tensor<4xi32>
    %mS = tt.splat %map : !tt.ptr<i32> -> tensor<4x!tt.ptr<i32>>
    %mp = tt.addptr %mS, %r : tensor<4x!tt.ptr<i32>>, tensor<4xi32>
    %m = tt.load %mp : tensor<4x!tt.ptr<i32>>
    %pS = tt.splat %src : !tt.ptr<i1> -> tensor<4x!tt.ptr<i1>>
    %p = tt.addptr %pS, %m : tensor<4x!tt.ptr<i1>>, tensor<4xi32>
    %pb = tt.bitcast %p : tensor<4x!tt.ptr<i1>> -> tensor<4x!tt.ptr<i8>>
    %v = tt.load %pb : tensor<4x!tt.ptr<i8>>
    %qS = tt.splat %dst : !tt.ptr<i1> -> tensor<4x!tt.ptr<i1>>
    %q = tt.addptr %qS, %m : tensor<4x!tt.ptr<i1>>, tensor<4xi32>
    %qb = tt.bitcast %q : tensor<4x!tt.ptr<i1>> -> tensor<4x!tt.ptr<i8>>
    tt.store %qb, %v : tensor<4x!tt.ptr<i8>>
    tt.return
    // CHECK: pumem.gather {{.*}} : memref<?xi8>, memref<4xi32, 128>, memref<4xi8, 128>
    // CHECK: pumem.scatter {{.*}} : memref<4xi8, 128>, memref<4xi32, 128>, memref<?xi8>
  }

  tt.func @masked_flip(%src: !tt.ptr<f32>, %dst: !tt.ptr<f32>) {
    // CHECK-LABEL: tt.func @masked_flip(
    // CHECK-SAME: %arg0: memref<?xf32>, %arg1: memref<?xf32>)
    %c242 = arith.constant 242 : i32
    %c243 = arith.constant 243 : i32
    %r = tt.make_range {start = 0 : i32, end = 256 : i32} : tensor<256xi32>
    %boundS = tt.splat %c243 : i32 -> tensor<256xi32>
    %mask = arith.cmpi slt, %r, %boundS : tensor<256xi32>
    %zero = arith.constant dense<0> : tensor<256xi32>
    %neg = arith.subi %zero, %r : tensor<256xi32>
    %srcP = tt.addptr %src, %c242 : !tt.ptr<f32>, i32
    %pS = tt.splat %srcP : !tt.ptr<f32> -> tensor<256x!tt.ptr<f32>>
    %p = tt.addptr %pS, %neg : tensor<256x!tt.ptr<f32>>, tensor<256xi32>
    %cst = arith.constant -1.000000e+00 : f32
    %otherS = tt.splat %cst : f32 -> tensor<256xf32>
    %v = tt.load %p, %mask, %otherS : tensor<256x!tt.ptr<f32>>
    %qS = tt.splat %dst : !tt.ptr<f32> -> tensor<256x!tt.ptr<f32>>
    %q = tt.addptr %qS, %r : tensor<256x!tt.ptr<f32>>, tensor<256xi32>
    tt.store %q, %v : tensor<256x!tt.ptr<f32>>
    tt.return
    // CHECK: pumem.gather {{.*}} : memref<?xf32>, memref<256xi32, 128>, memref<256xf32, 128>, memref<256xi1, 128>, f32
  }

  tt.func @masked_scatter(%dst: !tt.ptr<f32>) {
    // CHECK-LABEL: tt.func @masked_scatter(
    // CHECK-SAME: %arg0: memref<?xf32>)
    %c242 = arith.constant 242 : i32
    %c243 = arith.constant 243 : i32
    %r = tt.make_range {start = 0 : i32, end = 256 : i32} : tensor<256xi32>
    %boundS = tt.splat %c243 : i32 -> tensor<256xi32>
    %mask = arith.cmpi slt, %r, %boundS : tensor<256xi32>
    %c242S = tt.splat %c242 : i32 -> tensor<256xi32>
    %rev = arith.subi %c242S, %r : tensor<256xi32>
    %cst = arith.constant 1.000000e+00 : f32
    %v = tt.splat %cst : f32 -> tensor<256xf32>
    %dS = tt.splat %dst : !tt.ptr<f32> -> tensor<256x!tt.ptr<f32>>
    %d = tt.addptr %dS, %rev : tensor<256x!tt.ptr<f32>>, tensor<256xi32>
    tt.store %d, %v, %mask : tensor<256x!tt.ptr<f32>>
    tt.return
    // CHECK: pumem.scatter {{.*}} : memref<256xf32, 128>, memref<256xi32, 128>, memref<?xf32>, memref<256xi1, 128>
  }

  tt.func @scalar_offset_chain(%src: !tt.ptr<f32>, %map: !tt.ptr<i32>, %off: i32) {
    // CHECK-LABEL: tt.func @scalar_offset_chain(
    // CHECK-SAME: %arg0: memref<?xf32>, %arg1: memref<?xi32>, %arg2: i32)
    %r = tt.make_range {start = 0 : i32, end = 4 : i32} : tensor<4xi32>
    %mS = tt.splat %map : !tt.ptr<i32> -> tensor<4x!tt.ptr<i32>>
    %mp = tt.addptr %mS, %r : tensor<4x!tt.ptr<i32>>, tensor<4xi32>
    %m = tt.load %mp : tensor<4x!tt.ptr<i32>>
    %srcP = tt.addptr %src, %off : !tt.ptr<f32>, i32
    %pS = tt.splat %srcP : !tt.ptr<f32> -> tensor<4x!tt.ptr<f32>>
    %p = tt.addptr %pS, %m : tensor<4x!tt.ptr<f32>>, tensor<4xi32>
    %v = tt.load %p : tensor<4x!tt.ptr<f32>>
    tt.store %p, %v : tensor<4x!tt.ptr<f32>>
    tt.return
    // CHECK: tensor.splat %{{.*}} : tensor<4xi32>
    // CHECK: pumem.gather {{.*}} : memref<?xf32>, memref<4xi32, 128>, memref<4xf32, 128>
    // CHECK: pumem.scatter {{.*}} : memref<4xf32, 128>, memref<4xi32, 128>, memref<?xf32>
  }

  tt.func @bool_scalar_bitcast(%src: !tt.ptr<i1>, %dst: !tt.ptr<i1>) {
    // CHECK-LABEL: tt.func @bool_scalar_bitcast(
    // CHECK-SAME: %arg0: memref<?xi8>, %arg1: memref<?xi8>)
    %c242 = arith.constant 242 : i32
    %c243 = arith.constant 243 : i32
    %r = tt.make_range {start = 0 : i32, end = 256 : i32} : tensor<256xi32>
    %boundS = tt.splat %c243 : i32 -> tensor<256xi32>
    %mask = arith.cmpi slt, %r, %boundS : tensor<256xi32>
    %zero = arith.constant dense<0> : tensor<256xi32>
    %neg = arith.subi %zero, %r : tensor<256xi32>
    %srcP = tt.addptr %src, %c242 : !tt.ptr<i1>, i32
    %srcP8 = tt.bitcast %srcP : !tt.ptr<i1> -> !tt.ptr<i8>
    %pS = tt.splat %srcP8 : !tt.ptr<i8> -> tensor<256x!tt.ptr<i8>>
    %p = tt.addptr %pS, %neg : tensor<256x!tt.ptr<i8>>, tensor<256xi32>
    %v = tt.load %p, %mask : tensor<256x!tt.ptr<i8>>
    %dS = tt.splat %dst : !tt.ptr<i1> -> tensor<256x!tt.ptr<i1>>
    %q = tt.addptr %dS, %r : tensor<256x!tt.ptr<i1>>, tensor<256xi32>
    %qb = tt.bitcast %q : tensor<256x!tt.ptr<i1>> -> tensor<256x!tt.ptr<i8>>
    tt.store %qb, %v : tensor<256x!tt.ptr<i8>>
    tt.return
    // CHECK: pumem.gather {{.*}} : memref<?xi8>, memref<256xi32, 128>, memref<256xi8, 128>, memref<256xi1, 128>, i8
  }

  tt.func @coupled_mask(%src: !tt.ptr<f32>) {
    // CHECK-LABEL: tt.func @coupled_mask(
    // CHECK-SAME: %arg0: memref<?xf32>)
    %rows = tt.make_range {start = 0 : i32, end = 8 : i32} : tensor<8xi32>
    %cols = tt.make_range {start = 0 : i32, end = 8 : i32} : tensor<8xi32>
    %rowsE = tt.expand_dims %rows {axis = 1 : i32} : tensor<8xi32> -> tensor<8x1xi32>
    %c8 = arith.constant 8 : i32
    %c8S = tt.splat %c8 : i32 -> tensor<8x1xi32>
    %rowOff = arith.muli %rowsE, %c8S : tensor<8x1xi32>
    %rowB = tt.broadcast %rowOff : tensor<8x1xi32> -> tensor<8x8xi32>
    %colsE = tt.expand_dims %cols {axis = 0 : i32} : tensor<8xi32> -> tensor<1x8xi32>
    %colB = tt.broadcast %colsE : tensor<1x8xi32> -> tensor<8x8xi32>
    %idx = arith.addi %rowB, %colB : tensor<8x8xi32>
    %mask = arith.cmpi sge, %rowB, %colB : tensor<8x8xi32>
    %pS = tt.splat %src : !tt.ptr<f32> -> tensor<8x8x!tt.ptr<f32>>
    %a = tt.addptr %pS, %idx : tensor<8x8x!tt.ptr<f32>>, tensor<8x8xi32>
    %v = tt.load %a, %mask : tensor<8x8x!tt.ptr<f32>>
    tt.store %a, %v, %mask : tensor<8x8x!tt.ptr<f32>>
    tt.return
    // CHECK: pumem.gather {{.*}} : memref<?xf32>, memref<8x8xi32, 128>, memref<8x8xf32, 128>, memref<8x8xi1, 128>, f32
    // CHECK: pumem.scatter {{.*}} : memref<8x8xf32, 128>, memref<8x8xi32, 128>, memref<?xf32>, memref<8x8xi1, 128>
  }
}
