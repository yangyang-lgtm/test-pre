// RUN: privateuse-opt %s -privateuse-lower-structured-access -privateuse-lower-unstructured-access \
// RUN:   -privateuse-ptr-to-memref -canonicalize | FileCheck %s

module attributes {privateuse.grid = array<i64: 4, 3, 1>} {
  tt.func @mixed(%src: !tt.ptr<f32>, %dst: !tt.ptr<f32>) {
    %pid = tt.get_program_id x : i32
    %c4 = arith.constant 4 : i32
    %r = tt.make_range {start = 0 : i32, end = 8 : i32} : tensor<8xi32>
    %cond = arith.cmpi slt, %pid, %c4 : i32
    %half = tt.splat %cond : i1 -> tensor<8xi1>
    %pS = tt.splat %src : !tt.ptr<f32> -> tensor<8x!tt.ptr<f32>>
    %p = tt.addptr %pS, %r : tensor<8x!tt.ptr<f32>>, tensor<8xi32>
    %v = tt.load %p : tensor<8x!tt.ptr<f32>>
    %qS = tt.splat %dst : !tt.ptr<f32> -> tensor<8x!tt.ptr<f32>>
    %q = tt.addptr %qS, %r : tensor<8x!tt.ptr<f32>>, tensor<8xi32>
    tt.store %q, %v : tensor<8x!tt.ptr<f32>>
    %m = tt.load %p, %half : tensor<8x!tt.ptr<f32>>
    tt.store %q, %m, %half : tensor<8x!tt.ptr<f32>>
    tt.return
  }

  tt.func @bool_masked(%src: !tt.ptr<i8>, %dst: !tt.ptr<i1>) {
    %pid = tt.get_program_id x : i32
    %c4 = arith.constant 4 : i32
    %r = tt.make_range {start = 0 : i32, end = 8 : i32} : tensor<8xi32>
    %cond = arith.cmpi slt, %pid, %c4 : i32
    %half = tt.splat %cond : i1 -> tensor<8xi1>
    %sS = tt.splat %src : !tt.ptr<i8> -> tensor<8x!tt.ptr<i8>>
    %s = tt.addptr %sS, %r : tensor<8x!tt.ptr<i8>>, tensor<8xi32>
    %v = tt.load %s : tensor<8x!tt.ptr<i8>>
    %dS = tt.splat %dst : !tt.ptr<i1> -> tensor<8x!tt.ptr<i1>>
    %d = tt.addptr %dS, %r : tensor<8x!tt.ptr<i1>>, tensor<8xi32>
    %db = tt.bitcast %d : tensor<8x!tt.ptr<i1>> -> tensor<8x!tt.ptr<i8>>
    tt.store %db, %v, %half : tensor<8x!tt.ptr<i8>>
    tt.return
  }
}

// CHECK-LABEL: tt.func @mixed(
// CHECK-SAME: %arg0: memref<?xf32>, %arg1: memref<?xf32>
// CHECK-DAG: memref.reinterpret_cast %arg0 to offset: [0], sizes: [8], strides: [1] : memref<?xf32> to memref<8xf32, strided<[1]>>
// CHECK-DAG: memref.reinterpret_cast %arg1 to offset: [0], sizes: [8], strides: [1] : memref<?xf32> to memref<8xf32, strided<[1]>>
// CHECK-DAG: memref.copy
// CHECK-DAG: memref.copy
// CHECK: bufferization.to_buffer {{.*}} : tensor<8xi1> to memref<8xi1, 128>
// CHECK: pumem.gather {{.*}} : memref<?xf32>, memref<8xi32, 128>, memref<8xf32, 128>, memref<8xi1, 128>, f32
// CHECK: pumem.scatter {{.*}} : memref<8xf32, 128>, memref<8xi32, 128>, memref<?xf32>, memref<8xi1, 128>
// CHECK: tt.return

// CHECK-LABEL: tt.func @bool_masked(
// CHECK-SAME: %arg0: memref<?xi8>, %arg1: memref<?xi8>
// CHECK: memref.reinterpret_cast %arg0 to offset: [0], sizes: [8], strides: [1] : memref<?xi8> to memref<8xi8, strided<[1]>>
// CHECK: memref.copy
// CHECK: pumem.scatter {{.*}} : memref<8xi8, 128>, memref<8xi32, 128>, memref<?xi8>, memref<8xi1, 128>
// CHECK: tt.return
