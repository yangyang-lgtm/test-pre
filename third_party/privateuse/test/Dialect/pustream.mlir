// RUN: privateuse-opt %s | privateuse-opt | FileCheck %s

// The full statement set: allocations, a load and its wait, a view used by a
// dispatch, the compute wait, a store, an SRAM to SRAM copy and a final drain.
// The pipeline runs twice so printing and parsing have to agree.

// CHECK-LABEL: func.func @stream(
func.func @stream(%dram: memref<256xf32>, %dram2: memref<512xf32>, %c256: index) {
  // CHECK: %[[T0:.*]], %[[M0:.*]] = pustream.alloc() : !pustream.token, memref<256xf32, 128>
  %t0, %m0 = pustream.alloc() : !pustream.token, memref<256xf32, 128>
  // CHECK: %[[T1:.*]], %[[M1:.*]] = pustream.alloc() : !pustream.token, memref<512xf32, 128>
  %t1, %m1 = pustream.alloc() : !pustream.token, memref<512xf32, 128>

  // CHECK: %[[L:.*]] = pustream.transfer %[[T0]], %{{.*}}, %[[M0]] : !pustream.token, memref<256xf32>, memref<256xf32, 128> -> !pustream.token
  %l = pustream.transfer %t0, %dram, %m0
     : !pustream.token, memref<256xf32>, memref<256xf32, 128> -> !pustream.token
  // CHECK: %[[L1:.*]] = pustream.await %[[L]] : !pustream.token -> !pustream.token
  %l1 = pustream.await %l : !pustream.token -> !pustream.token

  // CHECK: %{{.*}} = memref.subview %[[M0]]
  %s = memref.subview %m0[0] [%c256] [1] : memref<256xf32, 128> to memref<?xf32, strided<[1]>, 128>

  // CHECK: %[[D:.*]] = pustream.dispatch @body(%[[L1]], %{{.*}})(%[[T1]], %[[M1]]) : (!pustream.token, memref<?xf32, strided<[1]>, 128>, !pustream.token, memref<512xf32, 128>) -> !pustream.token
  %d = pustream.dispatch @body(%l1, %s)(%t1, %m1)
     : (!pustream.token, memref<?xf32, strided<[1]>, 128>, !pustream.token, memref<512xf32, 128>)
       -> !pustream.token
  // CHECK: %[[D1:.*]] = pustream.await_compute %[[D]] : !pustream.token -> !pustream.token
  %d1 = pustream.await_compute %d : !pustream.token -> !pustream.token

  // CHECK: %[[ST:.*]] = pustream.transfer %[[D1]], %[[M1]], %{{.*}} : !pustream.token, memref<512xf32, 128>, memref<512xf32> -> !pustream.token
  %st = pustream.transfer %d1, %m1, %dram2
      : !pustream.token, memref<512xf32, 128>, memref<512xf32> -> !pustream.token
  // CHECK: %[[ST1:.*]] = pustream.await %[[ST]] : !pustream.token -> !pustream.token
  %st1 = pustream.await %st : !pustream.token -> !pustream.token

  // CHECK: %[[C:.*]] = pustream.copy %[[L1]], %{{.*}}, %[[D1]], %[[M1]] : !pustream.token, memref<?xf32, strided<[1]>, 128>, !pustream.token, memref<512xf32, 128> -> !pustream.token
  %c = pustream.copy %l1, %s, %d1, %m1
    : !pustream.token, memref<?xf32, strided<[1]>, 128>, !pustream.token, memref<512xf32, 128>
      -> !pustream.token
  // CHECK: %[[C1:.*]] = pustream.await %[[C]] : !pustream.token -> !pustream.token
  %c1 = pustream.await %c : !pustream.token -> !pustream.token

  // CHECK: pustream.sync
  pustream.sync
  return
}
