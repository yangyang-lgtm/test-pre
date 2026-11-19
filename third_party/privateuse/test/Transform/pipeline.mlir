// RUN: privateuse-opt %s -privateuse-pipeline=verbose -o /dev/null 2>&1 | FileCheck %s
// RUN: privateuse-opt %s -privateuse-schedule-timepoints | FileCheck %s --check-prefix=SCHED

// A tile loop with one load, one consumer and one store is a candidate.
// CHECK: remark: pipeline candidate: 1 consumer(s), streaming, stages=2
func.func @streaming(%dram: memref<?xf32>, %out: memref<?xf32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %n = arith.constant 16 : index
  %tile_size = arith.constant 256 : index
  scf.for %i = %c0 to %n step %c1 {
    %off = arith.muli %i, %tile_size : index
    %src = memref.subview %dram[%off] [256] [1] : memref<?xf32> to memref<256xf32, strided<[1], offset: ?>>
    %tile = memref.alloc() : memref<256xf32, 128>
    %res = memref.alloc() : memref<256xf32, 128>
    %s = pustream.slot %tile[%c0] : memref<256xf32, 128> -> !pustream.slot<memref<256xf32, 128>>
    pustream.async.copy %src, %s : memref<256xf32, strided<[1], offset: ?>>, !pustream.slot<memref<256xf32, 128>>
    pustream.wait %s : !pustream.slot<memref<256xf32, 128>>
    pustream.async.dispatch @body(%tile, %res) {reads = array<i64: 0>, writes = array<i64: 1>} : memref<256xf32, 128>, memref<256xf32, 128>
    pustream.wait_dispatch
    %r = pustream.slot %res[%c0] : memref<256xf32, 128> -> !pustream.slot<memref<256xf32, 128>>
    %dst = memref.subview %out[%off] [256] [1] : memref<?xf32> to memref<256xf32, strided<[1], offset: ?>>
    pustream.async.copy %r, %dst : !pustream.slot<memref<256xf32, 128>>, memref<256xf32, strided<[1], offset: ?>>
  }
  return
}

// A loop that pads the input tile before loading it cannot be pipelined yet: the
// prefetch would race with the fill. That is the masked-load tail case.
// CHECK: remark: not pipelined: the body writes the loaded tile (masked fill); needs the tail-slot transform
func.func @masked_fill(%dram: memref<?xf32>, %out: memref<?xf32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %n = arith.constant 16 : index
  %tile_size = arith.constant 256 : index
  %zero = arith.constant 0.0 : f32
  scf.for %i = %c0 to %n step %c1 {
    %off = arith.muli %i, %tile_size : index
    %src = memref.subview %dram[%off] [256] [1] : memref<?xf32> to memref<256xf32, strided<[1], offset: ?>>
    %tile = memref.alloc() : memref<256xf32, 128>
    %s = pustream.slot %tile[%c0] : memref<256xf32, 128> -> !pustream.slot<memref<256xf32, 128>>
    pustream.async.dispatch @fill(%zero, %tile) {reads = array<i64>, writes = array<i64: 1>} : f32, memref<256xf32, 128>
    pustream.wait_dispatch
    pustream.async.copy %src, %s : memref<256xf32, strided<[1], offset: ?>>, !pustream.slot<memref<256xf32, 128>>
    pustream.wait %s : !pustream.slot<memref<256xf32, 128>>
    pustream.async.dispatch @body(%tile) {reads = array<i64: 0>, writes = array<i64>} : memref<256xf32, 128>
    pustream.wait_dispatch
    %dst = memref.subview %out[%off] [256] [1] : memref<?xf32> to memref<256xf32, strided<[1], offset: ?>>
    pustream.async.copy %s, %dst : !pustream.slot<memref<256xf32, 128>>, memref<256xf32, strided<[1], offset: ?>>
  }
  return
}

// A loop that runs at most once has nothing to overlap.
// CHECK: remark: not pipelined: trip count is below 2, there is nothing to overlap
func.func @single_tile(%dram: memref<?xf32>, %out: memref<?xf32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %tile_size = arith.constant 256 : index
  scf.for %i = %c0 to %c1 step %c1 {
    %off = arith.muli %i, %tile_size : index
    %src = memref.subview %dram[%off] [256] [1] : memref<?xf32> to memref<256xf32, strided<[1], offset: ?>>
    %tile = memref.alloc() : memref<256xf32, 128>
    %s = pustream.slot %tile[%c0] : memref<256xf32, 128> -> !pustream.slot<memref<256xf32, 128>>
    pustream.async.copy %src, %s : memref<256xf32, strided<[1], offset: ?>>, !pustream.slot<memref<256xf32, 128>>
    pustream.wait %s : !pustream.slot<memref<256xf32, 128>>
    pustream.async.dispatch @body(%tile) {reads = array<i64: 0>, writes = array<i64>} : memref<256xf32, 128>
    pustream.wait_dispatch
    %dst = memref.subview %out[%off] [256] [1] : memref<?xf32> to memref<256xf32, strided<[1], offset: ?>>
    pustream.async.copy %s, %dst : !pustream.slot<memref<256xf32, 128>>, memref<256xf32, strided<[1], offset: ?>>
  }
  return
}

// A loop tagged `pustream.pipelined` belongs to the pipeliner: the scheduler
// must not scan or drain its body, but it still has to drain the kernel at exit.
// SCHED-LABEL: func.func @marked_loop(
// SCHED: pustream.async.copy
// SCHED-NOT: pustream.wait
// SCHED: pustream.sync
func.func @marked_loop(%dram: memref<?xf32>, %out: memref<?xf32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %n = arith.constant 16 : index
  %tile_size = arith.constant 256 : index
  scf.for %i = %c0 to %n step %c1 {
    %off = arith.muli %i, %tile_size : index
    %src = memref.subview %dram[%off] [256] [1] : memref<?xf32> to memref<256xf32, strided<[1], offset: ?>>
    %tile = memref.alloc() : memref<256xf32, 128>
    %s = pustream.slot %tile[%c0] : memref<256xf32, 128> -> !pustream.slot<memref<256xf32, 128>>
    pustream.async.copy %src, %s : memref<256xf32, strided<[1], offset: ?>>, !pustream.slot<memref<256xf32, 128>>
    pustream.async.dispatch @body(%tile) {reads = array<i64: 0>, writes = array<i64>} : memref<256xf32, 128>
    %dst = memref.subview %out[%off] [256] [1] : memref<?xf32> to memref<256xf32, strided<[1], offset: ?>>
    pustream.async.copy %s, %dst : !pustream.slot<memref<256xf32, 128>>, memref<256xf32, strided<[1], offset: ?>>
  } {pustream.pipelined}
  return
}
