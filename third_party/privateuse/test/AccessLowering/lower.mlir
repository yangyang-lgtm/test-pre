// RUN: privateuse-opt %s -privateuse-lower-structured-access -privateuse-ptr-to-memref -canonicalize \
// RUN:   -convert-elementwise-to-linalg -one-shot-bufferize="allow-unknown-ops=1" \
// RUN:   -canonicalize | FileCheck %s \
// RUN:   --implicit-check-not=tt.load --implicit-check-not=tt.store \
// RUN:   --implicit-check-not=tt.splat --implicit-check-not=tensor \
// RUN:   --implicit-check-not=bufferization. --implicit-check-not=pumem.

module attributes {privateuse.grid = array<i64: 4, 3, 1>} {
  tt.func @copy(%src: !tt.ptr<f32>, %dst: !tt.ptr<f32>) {
    %pid = tt.get_program_id x : i32
    %c128 = arith.constant 128 : i32
    %base = arith.muli %pid, %c128 : i32
    %baseS = tt.splat %base : i32 -> tensor<128xi32>
    %r = tt.make_range {start = 0 : i32, end = 128 : i32} : tensor<128xi32>
    %idx = arith.addi %baseS, %r : tensor<128xi32>
    %pS = tt.splat %src : !tt.ptr<f32> -> tensor<128x!tt.ptr<f32>>
    %p = tt.addptr %pS, %idx : tensor<128x!tt.ptr<f32>>, tensor<128xi32>
    %v = tt.load %p : tensor<128x!tt.ptr<f32>>
    %qS = tt.splat %dst : !tt.ptr<f32> -> tensor<128x!tt.ptr<f32>>
    %q = tt.addptr %qS, %idx : tensor<128x!tt.ptr<f32>>, tensor<128xi32>
    tt.store %q, %v : tensor<128x!tt.ptr<f32>>
    tt.return
  }

  tt.func @scale(%src: !tt.ptr<f32>, %dst: !tt.ptr<f32>) {
    %cst = arith.constant 2.000000e+00 : f32
    %c128 = arith.constant 128 : i32
    %r = tt.make_range {start = 0 : i32, end = 128 : i32} : tensor<128xi32>
    %boundS = tt.splat %c128 : i32 -> tensor<128xi32>
    %full = arith.cmpi slt, %r, %boundS : tensor<128xi32>
    %pS = tt.splat %src : !tt.ptr<f32> -> tensor<128x!tt.ptr<f32>>
    %p = tt.addptr %pS, %r : tensor<128x!tt.ptr<f32>>, tensor<128xi32>
    %v = tt.load %p, %full : tensor<128x!tt.ptr<f32>>
    %cstS = tt.splat %cst : f32 -> tensor<128xf32>
    %w = arith.mulf %v, %cstS : tensor<128xf32>
    %qS = tt.splat %dst : !tt.ptr<f32> -> tensor<128x!tt.ptr<f32>>
    %q = tt.addptr %qS, %r : tensor<128x!tt.ptr<f32>>, tensor<128xi32>
    tt.store %q, %w, %full : tensor<128x!tt.ptr<f32>>
    tt.return
  }

  tt.func @two_dim(%src: !tt.ptr<f32>, %dst: !tt.ptr<f32>) {
    %pid = tt.get_program_id x : i32
    %c4096 = arith.constant 4096 : i32
    %base = arith.muli %pid, %c4096 : i32
    %c64 = arith.constant 64 : i32
    %rows = tt.make_range {start = 0 : i32, end = 64 : i32} : tensor<64xi32>
    %cols = tt.make_range {start = 0 : i32, end = 64 : i32} : tensor<64xi32>
    %rowsE = tt.expand_dims %rows {axis = 1 : i32} : tensor<64xi32> -> tensor<64x1xi32>
    %c64S1 = tt.splat %c64 : i32 -> tensor<64x1xi32>
    %rowOff = arith.muli %rowsE, %c64S1 : tensor<64x1xi32>
    %rowBc = tt.broadcast %rowOff : tensor<64x1xi32> -> tensor<64x64xi32>
    %colsE = tt.expand_dims %cols {axis = 0 : i32} : tensor<64xi32> -> tensor<1x64xi32>
    %colsBc = tt.broadcast %colsE : tensor<1x64xi32> -> tensor<64x64xi32>
    %idx0 = arith.addi %rowBc, %colsBc : tensor<64x64xi32>
    %baseS = tt.splat %base : i32 -> tensor<64x64xi32>
    %idx = arith.addi %idx0, %baseS : tensor<64x64xi32>
    %pS = tt.splat %src : !tt.ptr<f32> -> tensor<64x64x!tt.ptr<f32>>
    %a = tt.addptr %pS, %idx : tensor<64x64x!tt.ptr<f32>>, tensor<64x64xi32>
    %v = tt.load %a : tensor<64x64x!tt.ptr<f32>>
    %qS = tt.splat %dst : !tt.ptr<f32> -> tensor<64x64x!tt.ptr<f32>>
    %q = tt.addptr %qS, %idx : tensor<64x64x!tt.ptr<f32>>, tensor<64x64xi32>
    tt.store %q, %v : tensor<64x64x!tt.ptr<f32>>
    tt.return
  }

  tt.func @broadcast_load(%src: !tt.ptr<f32>, %dst: !tt.ptr<f32>) {
    %r = tt.make_range {start = 5 : i32, end = 6 : i32} : tensor<1xi32>
    %bc = tt.broadcast %r : tensor<1xi32> -> tensor<8xi32>
    %pS = tt.splat %src : !tt.ptr<f32> -> tensor<8x!tt.ptr<f32>>
    %a = tt.addptr %pS, %bc : tensor<8x!tt.ptr<f32>>, tensor<8xi32>
    %v = tt.load %a : tensor<8x!tt.ptr<f32>>
    %idx = tt.make_range {start = 0 : i32, end = 8 : i32} : tensor<8xi32>
    %qS = tt.splat %dst : !tt.ptr<f32> -> tensor<8x!tt.ptr<f32>>
    %q = tt.addptr %qS, %idx : tensor<8x!tt.ptr<f32>>, tensor<8xi32>
    tt.store %q, %v : tensor<8x!tt.ptr<f32>>
    tt.return
  }

  tt.func @loop_acc(%src: !tt.ptr<f32>, %dst: !tt.ptr<f32>, %n: i32) {
    %c0 = arith.constant 0 : i32
    %c32 = arith.constant 32 : i32
    %cst = arith.constant 0.000000e+00 : f32
    %r = tt.make_range {start = 0 : i32, end = 128 : i32} : tensor<128xi32>
    %pS = tt.splat %src : !tt.ptr<f32> -> tensor<128x!tt.ptr<f32>>
    %init = tt.addptr %pS, %r : tensor<128x!tt.ptr<f32>>, tensor<128xi32>
    %acc0 = tt.splat %cst : f32 -> tensor<128xf32>
    %stepS = tt.splat %c32 : i32 -> tensor<128xi32>
    %res:2 = scf.for %iv = %c0 to %n step %c32 iter_args(%ptr = %init, %acc = %acc0) -> (tensor<128x!tt.ptr<f32>>, tensor<128xf32>) : i32 {
      %v = tt.load %ptr : tensor<128x!tt.ptr<f32>>
      %sum = arith.addf %acc, %v : tensor<128xf32>
      %next = tt.addptr %ptr, %stepS : tensor<128x!tt.ptr<f32>>, tensor<128xi32>
      scf.yield %next, %sum : tensor<128x!tt.ptr<f32>>, tensor<128xf32>
    }
    %qS = tt.splat %dst : !tt.ptr<f32> -> tensor<128x!tt.ptr<f32>>
    %q = tt.addptr %qS, %r : tensor<128x!tt.ptr<f32>>, tensor<128xi32>
    tt.store %q, %res#1 : tensor<128x!tt.ptr<f32>>
    tt.return
  }
}

// CHECK-LABEL: tt.func @copy(
// CHECK-SAME: %arg0: memref<?xf32>, %arg1: memref<?xf32>
// CHECK: memref.reinterpret_cast %arg0 to offset: [{{.*}}], sizes: [128], strides: [1] : memref<?xf32> to memref<128xf32, strided<[1], offset: ?>>
// CHECK: %[[BUF:.*]] = memref.alloc() {{.*}} : memref<128xf32, 128>
// CHECK: memref.copy {{.*}}, %[[BUF]] : memref<128xf32, strided<[1], offset: ?>> to memref<128xf32, 128>
// CHECK: memref.reinterpret_cast %arg1 to offset: [{{.*}}], sizes: [128], strides: [1] : memref<?xf32> to memref<128xf32, strided<[1], offset: ?>>
// CHECK: memref.copy %[[BUF]], {{.*}} : memref<128xf32, 128> to memref<128xf32, strided<[1], offset: ?>>
// CHECK: tt.return

// CHECK-LABEL: tt.func @scale(
// CHECK-SAME: %arg0: memref<?xf32>, %arg1: memref<?xf32>
// CHECK: memref.reinterpret_cast %arg0 to offset: [0], sizes: [128], strides: [1]
// CHECK: memref.alloc() {{.*}} : memref<128xf32, 128>
// CHECK: memref.copy
// CHECK: linalg.generic
// CHECK: arith.mulf
// CHECK: memref.reinterpret_cast %arg1 to offset: [0], sizes: [128], strides: [1]
// CHECK: memref.copy
// CHECK: tt.return

// CHECK-LABEL: tt.func @two_dim(
// CHECK-SAME: %arg0: memref<?xf32>, %arg1: memref<?xf32>
// CHECK: memref.reinterpret_cast %arg0 to offset: [{{.*}}], sizes: [64, 64], strides: [64, 1] : memref<?xf32> to memref<64x64xf32, strided<[64, 1], offset: ?>>
// CHECK: memref.alloc() {{.*}} : memref<64x64xf32, 128>
// CHECK: memref.copy {{.*}}, {{.*}} : memref<64x64xf32, strided<[64, 1], offset: ?>> to memref<64x64xf32, 128>
// CHECK: memref.reinterpret_cast %arg1 to offset: [{{.*}}], sizes: [64, 64], strides: [64, 1] : memref<?xf32> to memref<64x64xf32, strided<[64, 1], offset: ?>>
// CHECK: memref.copy {{.*}}, {{.*}} : memref<64x64xf32, 128> to memref<64x64xf32, strided<[64, 1], offset: ?>>
// CHECK: tt.return

// CHECK-LABEL: tt.func @broadcast_load(
// CHECK-SAME: %arg0: memref<?xf32>, %arg1: memref<?xf32>
// CHECK: memref.reinterpret_cast %arg0 to offset: [5], sizes: [8], strides: [0] : memref<?xf32> to memref<8xf32, strided<[0], offset: 5>>
// CHECK: memref.alloc() {{.*}} : memref<8xf32, 128>
// CHECK: memref.copy
// CHECK: memref.reinterpret_cast %arg1 to offset: [0], sizes: [8], strides: [1]
// CHECK: memref.copy
// CHECK: tt.return

// CHECK-LABEL: tt.func @loop_acc(
// CHECK-SAME: %arg0: memref<?xf32>, %arg1: memref<?xf32>, %arg2: i32
// CHECK: %[[ACC:.*]] = memref.alloc{{.*}}: memref<128xf32>
// CHECK: memref.copy {{.*}}, %[[ACC]] : memref<128xf32> to memref<128xf32>
// CHECK: scf.for
// CHECK: %[[TILE:.*]] = memref.alloc() {{.*}} : memref<128xf32, 128>
// CHECK: memref.copy {{.*}}, %[[TILE]] : memref<128xf32, strided<[1], offset: ?>> to memref<128xf32, 128>
// CHECK: linalg.generic
// CHECK: ins(%[[ACC]], %[[TILE]] : memref<128xf32>, memref<128xf32, 128>)
// CHECK: memref.reinterpret_cast %arg1
// CHECK: %[[OUT:.*]] = memref.alloc() {{.*}} : memref<128xf32, 128>
// CHECK: memref.copy %[[ACC]], %[[OUT]] : memref<128xf32> to memref<128xf32, 128>
// CHECK: memref.copy %[[OUT]], {{.*}} : memref<128xf32, 128> to memref<128xf32, strided<[1]>>
// CHECK: tt.return
