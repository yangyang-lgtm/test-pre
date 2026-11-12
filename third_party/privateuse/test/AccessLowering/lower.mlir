// RUN: privateuse-opt %s -privateuse-lower-structured-access -privateuse-ptr-to-memref -canonicalize \
// RUN:   -convert-elementwise-to-linalg -one-shot-bufferize="allow-unknown-ops=1" \
// RUN:   -canonicalize | FileCheck %s \
// RUN:   --implicit-check-not=tt.load --implicit-check-not=tt.store \
// RUN:   --implicit-check-not=tt.splat --implicit-check-not=tensor \
// RUN:   --implicit-check-not=bufferization. --implicit-check-not=pumem. \
// RUN:   --implicit-check-not=tt.bitcast

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

  tt.func @bool_store(%src: !tt.ptr<i8>, %dst: !tt.ptr<i1>) {
    %c8 = arith.constant 8 : i32
    %r = tt.make_range {start = 0 : i32, end = 8 : i32} : tensor<8xi32>
    %boundS = tt.splat %c8 : i32 -> tensor<8xi32>
    %full = arith.cmpi slt, %r, %boundS : tensor<8xi32>
    %sS = tt.splat %src : !tt.ptr<i8> -> tensor<8x!tt.ptr<i8>>
    %s = tt.addptr %sS, %r : tensor<8x!tt.ptr<i8>>, tensor<8xi32>
    %v = tt.load %s, %full : tensor<8x!tt.ptr<i8>>
    %dS = tt.splat %dst : !tt.ptr<i1> -> tensor<8x!tt.ptr<i1>>
    %d = tt.addptr %dS, %r : tensor<8x!tt.ptr<i1>>, tensor<8xi32>
    %db = tt.bitcast %d : tensor<8x!tt.ptr<i1>> -> tensor<8x!tt.ptr<i8>>
    tt.store %db, %v, %full : tensor<8x!tt.ptr<i8>>
    tt.return
  }

  tt.func @scalar_access(%w: !tt.ptr<f32>, %c: !tt.ptr<i32>, %n: i32) {
    %pid = tt.get_program_id x : i32
    %c18 = arith.constant 18 : i32
    %base = arith.muli %pid, %c18 : i32
    %wp = tt.addptr %w, %base : !tt.ptr<f32>, i32
    %v = tt.load %wp : !tt.ptr<f32>
    %cp = tt.addptr %c, %base : !tt.ptr<i32>, i32
    tt.store %cp, %c18 : !tt.ptr<i32>
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %res = scf.for %iv = %c0 to %n step %c1 iter_args(%acc = %v) -> f32 : i32 {
      %ip = tt.addptr %w, %iv : !tt.ptr<f32>, i32
      %e = tt.load %ip : !tt.ptr<f32>
      %sum = arith.addf %acc, %e : f32
      scf.yield %sum : f32
    }
    %op = tt.addptr %w, %c0 : !tt.ptr<f32>, i32
    tt.store %op, %res : !tt.ptr<f32>
    tt.return
  }

  tt.func @tail_load(%src: !tt.ptr<f32>, %dst: !tt.ptr<f32>) {
    %pid = tt.get_program_id x : i32
    %c16 = arith.constant 16 : i32
    %bound = arith.addi %pid, %c16 : i32
    %r = tt.make_range {start = 0 : i32, end = 128 : i32} : tensor<128xi32>
    %boundS = tt.splat %bound : i32 -> tensor<128xi32>
    %m = arith.cmpi slt, %r, %boundS : tensor<128xi32>
    %cst = arith.constant -1.000000e+00 : f32
    %otherS = tt.splat %cst : f32 -> tensor<128xf32>
    %pS = tt.splat %src : !tt.ptr<f32> -> tensor<128x!tt.ptr<f32>>
    %p = tt.addptr %pS, %r : tensor<128x!tt.ptr<f32>>, tensor<128xi32>
    %v = tt.load %p, %m, %otherS : tensor<128x!tt.ptr<f32>>
    %qS = tt.splat %dst : !tt.ptr<f32> -> tensor<128x!tt.ptr<f32>>
    %q = tt.addptr %qS, %r : tensor<128x!tt.ptr<f32>>, tensor<128xi32>
    tt.store %q, %v : tensor<128x!tt.ptr<f32>>
    tt.return
  }

  tt.func @box_2d(%src: !tt.ptr<f32>, %dst: !tt.ptr<f32>) {
    %pid = tt.get_program_id x : i32
    %c4 = arith.constant 4 : i32
    %m = arith.addi %pid, %c4 : i32
    %n = arith.addi %pid, %c4 : i32
    %c2 = arith.constant 2 : i32
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
    %mS = tt.splat %m : i32 -> tensor<8x8xi32>
    %nS = tt.splat %n : i32 -> tensor<8x8xi32>
    %mr = arith.cmpi slt, %rowB, %mS : tensor<8x8xi32>
    %c2S = tt.splat %c2 : i32 -> tensor<8x8xi32>
    %col2 = arith.muli %colB, %c2S : tensor<8x8xi32>
    %nc = arith.cmpi sle, %col2, %nS : tensor<8x8xi32>
    %mask = arith.andi %mr, %nc : tensor<8x8xi1>
    %pS = tt.splat %src : !tt.ptr<f32> -> tensor<8x8x!tt.ptr<f32>>
    %p = tt.addptr %pS, %idx : tensor<8x8x!tt.ptr<f32>>, tensor<8x8xi32>
    %v = tt.load %p, %mask : tensor<8x8x!tt.ptr<f32>>
    %qS = tt.splat %dst : !tt.ptr<f32> -> tensor<8x8x!tt.ptr<f32>>
    %q = tt.addptr %qS, %idx : tensor<8x8x!tt.ptr<f32>>, tensor<8x8xi32>
    tt.store %q, %v, %mask : tensor<8x8x!tt.ptr<f32>>
    tt.return
  }

  // 整块 view 写地址重叠（行距 14 < 16），但 box mask 写出的 14x14 前缀窗口仍单射，可以降级。
  tt.func @overlap_box_store(%src: !tt.ptr<f32>, %dst: !tt.ptr<f32>) {
    %rows = tt.make_range {start = 0 : i32, end = 16 : i32} : tensor<16xi32>
    %cols = tt.make_range {start = 0 : i32, end = 16 : i32} : tensor<16xi32>
    %rowsE = tt.expand_dims %rows {axis = 1 : i32} : tensor<16xi32> -> tensor<16x1xi32>
    %c16 = arith.constant 16 : i32
    %c16S = tt.splat %c16 : i32 -> tensor<16x1xi32>
    %rowOff = arith.muli %rowsE, %c16S : tensor<16x1xi32>
    %rowB = tt.broadcast %rowOff : tensor<16x1xi32> -> tensor<16x16xi32>
    %colsE = tt.expand_dims %cols {axis = 0 : i32} : tensor<16xi32> -> tensor<1x16xi32>
    %colB = tt.broadcast %colsE : tensor<1x16xi32> -> tensor<16x16xi32>
    %idx = arith.addi %rowB, %colB : tensor<16x16xi32>
    %pS = tt.splat %src : !tt.ptr<f32> -> tensor<16x16x!tt.ptr<f32>>
    %p = tt.addptr %pS, %idx : tensor<16x16x!tt.ptr<f32>>, tensor<16x16xi32>
    %v = tt.load %p : tensor<16x16x!tt.ptr<f32>>
    %c14 = arith.constant 14 : i32
    %c14S = tt.splat %c14 : i32 -> tensor<16x16xi32>
    %rowsB = tt.broadcast %rowsE : tensor<16x1xi32> -> tensor<16x16xi32>
    %mr = arith.cmpi slt, %rowsB, %c14S : tensor<16x16xi32>
    %mc = arith.cmpi slt, %colB, %c14S : tensor<16x16xi32>
    %mask = arith.andi %mr, %mc : tensor<16x16xi1>
    %c14S1 = tt.splat %c14 : i32 -> tensor<16x1xi32>
    %rowOff14 = arith.muli %rowsE, %c14S1 : tensor<16x1xi32>
    %rowB14 = tt.broadcast %rowOff14 : tensor<16x1xi32> -> tensor<16x16xi32>
    %idx14 = arith.addi %rowB14, %colB : tensor<16x16xi32>
    %qS = tt.splat %dst : !tt.ptr<f32> -> tensor<16x16x!tt.ptr<f32>>
    %q = tt.addptr %qS, %idx14 : tensor<16x16x!tt.ptr<f32>>, tensor<16x16xi32>
    tt.store %q, %v, %mask : tensor<16x16x!tt.ptr<f32>>
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

// CHECK-LABEL: tt.func @bool_store(
// CHECK-SAME: %arg0: memref<?xi8>, %arg1: memref<?xi8>
// CHECK: memref.reinterpret_cast %arg0 to offset: [0], sizes: [8], strides: [1] : memref<?xi8> to memref<8xi8, strided<[1]>>
// CHECK: memref.alloc() {{.*}} : memref<8xi8, 128>
// CHECK: memref.copy
// CHECK: memref.reinterpret_cast %arg1 to offset: [0], sizes: [8], strides: [1] : memref<?xi8> to memref<8xi8, strided<[1]>>
// CHECK: memref.copy
// CHECK: tt.return

// CHECK-LABEL: tt.func @scalar_access(
// CHECK-SAME: %arg0: memref<?xf32>, %arg1: memref<?xi32>, %arg2: i32
// CHECK: memref.load %arg0[{{.*}}] : memref<?xf32>
// CHECK: memref.store %{{.*}}, %arg1[{{.*}}] : memref<?xi32>
// CHECK: scf.for
// CHECK: arith.index_cast
// CHECK: memref.load %arg0[{{.*}}] : memref<?xf32>
// CHECK: memref.store %{{.*}}, %arg0[{{.*}}] : memref<?xf32>
// CHECK: tt.return

// CHECK-LABEL: tt.func @tail_load(
// CHECK-SAME: %arg0: memref<?xf32>, %arg1: memref<?xf32>
// CHECK-DAG: %[[OTHER:.*]] = arith.constant -1.000000e+00 : f32
// CHECK-DAG: %[[N:.*]] = arith.minsi {{.*}}, %c128 : index
// CHECK: %[[VIEW:.*]] = memref.reinterpret_cast %arg0 to offset: [0], sizes: [%[[N]]], strides: [1] : memref<?xf32> to memref<?xf32, strided<[1]>>
// CHECK: %[[BUF:.*]] = memref.alloc() {{.*}} : memref<128xf32, 128>
// CHECK: linalg.fill ins(%[[OTHER]] : f32) outs(%[[BUF]] : memref<128xf32, 128>)
// CHECK: %[[WIN:.*]] = memref.subview %[[BUF]][0] [%[[N]]] [1] : memref<128xf32, 128> to memref<?xf32, strided<[1]>, 128>
// CHECK: memref.copy %[[VIEW]], %[[WIN]] : memref<?xf32, strided<[1]>> to memref<?xf32, strided<[1]>, 128>
// CHECK: memref.reinterpret_cast %arg1 to offset: [0], sizes: [128], strides: [1] : memref<?xf32> to memref<128xf32, strided<[1]>>
// CHECK: memref.copy %[[BUF]], {{.*}} : memref<128xf32, 128> to memref<128xf32, strided<[1]>>
// CHECK: tt.return

// CHECK-LABEL: tt.func @box_2d(
// CHECK-SAME: %arg0: memref<?xf32>, %arg1: memref<?xf32>
// CHECK-DAG: %[[ZERO:.*]] = arith.constant 0.000000e+00 : f32
// CHECK-DAG: arith.ceildivsi
// CHECK-DAG: arith.floordivsi
// CHECK: %[[VIEW2:.*]] = memref.reinterpret_cast %arg0 to offset: [0], sizes: [%{{.*}}, %{{.*}}], strides: [8, 1] : memref<?xf32> to memref<?x?xf32, strided<[8, 1]>>
// CHECK: %[[BUF2:.*]] = memref.alloc() {{.*}} : memref<8x8xf32, 128>
// CHECK: linalg.fill ins(%[[ZERO]] : f32) outs(%[[BUF2]] : memref<8x8xf32, 128>)
// CHECK: %[[WIN2:.*]] = memref.subview %[[BUF2]][0, 0] [%{{.*}}, %{{.*}}] [1, 1] : memref<8x8xf32, 128> to memref<?x?xf32, strided<[8, 1]>, 128>
// CHECK: memref.copy %[[VIEW2]], %[[WIN2]] : memref<?x?xf32, strided<[8, 1]>> to memref<?x?xf32, strided<[8, 1]>, 128>
// CHECK: memref.reinterpret_cast %arg1 to offset: [0], sizes: [%{{.*}}, %{{.*}}], strides: [8, 1] : memref<?xf32> to memref<?x?xf32, strided<[8, 1]>>
// CHECK: %[[WIN3:.*]] = memref.subview %[[BUF2]][0, 0] [%{{.*}}, %{{.*}}] [1, 1] : memref<8x8xf32, 128> to memref<?x?xf32, strided<[8, 1]>, 128>
// CHECK: memref.copy %[[WIN3]], {{.*}} : memref<?x?xf32, strided<[8, 1]>, 128> to memref<?x?xf32, strided<[8, 1]>>
// CHECK: tt.return

// CHECK-LABEL: tt.func @overlap_box_store(
// CHECK-SAME: %arg0: memref<?xf32>, %arg1: memref<?xf32>
// CHECK: %[[OTILE:.*]] = memref.alloc() {{.*}} : memref<16x16xf32, 128>
// CHECK: memref.copy {{.*}}, %[[OTILE]] : memref<16x16xf32, strided<[16, 1]>> to memref<16x16xf32, 128>
// CHECK: memref.reinterpret_cast %arg1 to offset: [0], sizes: [14, 14], strides: [14, 1] : memref<?xf32> to memref<14x14xf32, strided<[14, 1]>>
// CHECK: %[[OWIN:.*]] = memref.subview %[[OTILE]][0, 0] [14, 14] [1, 1] : memref<16x16xf32, 128> to memref<14x14xf32, strided<[16, 1]>, 128>
// CHECK: memref.copy %[[OWIN]], {{.*}} : memref<14x14xf32, strided<[16, 1]>, 128> to memref<14x14xf32, strided<[14, 1]>>
// CHECK: tt.return
