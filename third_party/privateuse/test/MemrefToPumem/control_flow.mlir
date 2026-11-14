// RUN: privateuse-opt %s -privateuse-memref-to-pumem | FileCheck %s \
// RUN:   --implicit-check-not=memref.

// Control flow carries the flattened tuple: scf.for region arguments and
// results, scf.yield, scf.if results, function signatures, func.call and
// func.return all agree on (ptr, offset, sizes..., strides...). The tt.func /
// tt.call / tt.return layer is rewritten to the func dialect.

// CHECK-LABEL: func.func @loop(
// CHECK-SAME: %[[PTR:.*]]: !pumem.ptr<f32, 128>
// CHECK-SAME: -> (!pumem.ptr<f32, 128>, index, index, index) {
tt.func @loop(%A: memref<16xf32, 128>, %n: index) -> memref<16xf32, 128> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  // CHECK: %[[RES:.*]]:4 = scf.for
  // CHECK-SAME: iter_args(%[[ITER:.*]] = %[[PTR]]
  // CHECK-SAME: -> (!pumem.ptr<f32, 128>, index, index, index) {
  %res = scf.for %i = %c0 to %n step %c1 iter_args(%buf = %A) -> (memref<16xf32, 128>) {
    // CHECK: pumem.addptr %[[ITER]], %{{.*}} : !pumem.ptr<f32, 128>
    // CHECK: pumem.load
    %v = memref.load %buf[%i] : memref<16xf32, 128>
    // CHECK: pumem.store
    memref.store %v, %buf[%i] : memref<16xf32, 128>
    // CHECK: scf.yield {{.*}} : !pumem.ptr<f32, 128>, index, index, index
    scf.yield %buf : memref<16xf32, 128>
  }
  // CHECK: return %[[RES]]#0, %[[RES]]#1, %[[RES]]#2, %[[RES]]#3
  tt.return %res : memref<16xf32, 128>
}

// CHECK-LABEL: func.func @branch(
tt.func @branch(%cond: i1, %A: memref<16xf32, 128>, %B: memref<16xf32, 128>) -> memref<16xf32, 128> {
  // CHECK: %[[R:.*]]:4 = scf.if %{{.*}} -> (!pumem.ptr<f32, 128>, index, index, index)
  %r = scf.if %cond -> (memref<16xf32, 128>) {
    // CHECK: scf.yield {{.*}} : !pumem.ptr<f32, 128>, index, index, index
    scf.yield %A : memref<16xf32, 128>
  } else {
    // CHECK: scf.yield {{.*}} : !pumem.ptr<f32, 128>, index, index, index
    scf.yield %B : memref<16xf32, 128>
  }
  tt.return %r : memref<16xf32, 128>
}

// CHECK-LABEL: func.func @caller(
tt.func @caller(%A: memref<16xf32, 128>, %n: index) {
  // CHECK: %[[C:.*]]:4 = call @loop({{.*}}) : (!pumem.ptr<f32, 128>, index, index, index, index) -> (!pumem.ptr<f32, 128>, index, index, index)
  %r = tt.call @loop(%A, %n) : (memref<16xf32, 128>, index) -> memref<16xf32, 128>
  // CHECK: return
  tt.return
}
