// RUN: privateuse-opt %s | FileCheck %s

// CHECK-LABEL: func.func @single_buffer(
func.func @single_buffer(%dram: memref<8xf32>, %out: memref<8xf32>) {
  %c0 = arith.constant 0 : index
  %sram = memref.alloc() : memref<8xf32, 128>
  // CHECK: %[[S:.*]] = pustream.slot %{{.*}}[%{{.*}}] : memref<8xf32, 128> -> !pustream.slot<memref<8xf32, 128>>
  %s = pustream.slot %sram[%c0] : memref<8xf32, 128> -> !pustream.slot<memref<8xf32, 128>>
  // CHECK: pustream.async.copy %{{.*}}, %[[S]] : memref<8xf32>, !pustream.slot<memref<8xf32, 128>>
  pustream.async.copy %dram, %s : memref<8xf32>, !pustream.slot<memref<8xf32, 128>>
  // CHECK: pustream.wait %[[S]] : !pustream.slot<memref<8xf32, 128>>
  pustream.wait %s : !pustream.slot<memref<8xf32, 128>>
  // CHECK: pustream.async.dispatch @k(%[[S]]) {reads = array<i64: 0>, writes = array<i64: 0>}
  pustream.async.dispatch @k(%s) {reads = array<i64: 0>, writes = array<i64: 0>} : !pustream.slot<memref<8xf32, 128>>
  // CHECK: pustream.wait_dispatch
  pustream.wait_dispatch
  // CHECK: pustream.async.copy %[[S]], %{{.*}} : !pustream.slot<memref<8xf32, 128>>, memref<8xf32>
  pustream.async.copy %s, %out : !pustream.slot<memref<8xf32, 128>>, memref<8xf32>
  // CHECK: pustream.sync
  pustream.sync
  return
}

// CHECK-LABEL: func.func @staged_dynamic_index(
func.func @staged_dynamic_index(%dram: memref<256xf32>, %i: index) {
  // CHECK: %[[IN:.*]] = memref.alloc() {pustream.stages = 4 : i32} : memref<4x256xf32, 128>
  %in = memref.alloc() {pustream.stages = 4 : i32} : memref<4x256xf32, 128>
  // CHECK: %[[S:.*]] = pustream.slot %[[IN]][%{{.*}}] : memref<4x256xf32, 128> -> !pustream.slot<memref<256xf32, 128>>
  %s = pustream.slot %in[%i] : memref<4x256xf32, 128> -> !pustream.slot<memref<256xf32, 128>>
  // CHECK: pustream.async.copy %{{.*}}, %[[S]] : memref<256xf32>, !pustream.slot<memref<256xf32, 128>>
  pustream.async.copy %dram, %s : memref<256xf32>, !pustream.slot<memref<256xf32, 128>>
  pustream.wait %s : !pustream.slot<memref<256xf32, 128>>
  pustream.sync
  return
}

// A partial tile: the DRAM side has a dynamic extent and dynamic strides, the
// slot only contributes capacity. This must verify -- the box comes from the
// DRAM operand, not from the slot type.
// CHECK-LABEL: func.func @partial_tile(
func.func @partial_tile(%dram: memref<?xf32, strided<[1], offset: ?>>) {
  %c0 = arith.constant 0 : index
  %tile = memref.alloc() : memref<256xf32, 128>
  %s = pustream.slot %tile[%c0] : memref<256xf32, 128> -> !pustream.slot<memref<256xf32, 128>>
  // CHECK: pustream.async.copy %{{.*}}, %{{.*}} : memref<?xf32, strided<[1], offset: ?>>, !pustream.slot<memref<256xf32, 128>>
  pustream.async.copy %dram, %s : memref<?xf32, strided<[1], offset: ?>>, !pustream.slot<memref<256xf32, 128>>
  pustream.wait %s : !pustream.slot<memref<256xf32, 128>>
  pustream.sync
  return
}
