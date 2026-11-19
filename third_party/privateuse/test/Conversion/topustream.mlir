// RUN: privateuse-opt %s -privateuse-to-pustream --split-input-file | FileCheck %s

// A cross-space copy becomes an async.copy on a slot.
// CHECK-LABEL: func.func @cross_space(
// CHECK: %[[SLOT:.*]] = pustream.slot %{{.*}}[%{{.*}}] : memref<8xf32, 128> -> !pustream.slot<memref<8xf32, 128>>
// CHECK: pustream.async.copy %{{.*}}, %[[SLOT]] : memref<8xf32>, !pustream.slot<memref<8xf32, 128>>
func.func @cross_space(%dram: memref<8xf32>) {
  %sram = memref.alloc() : memref<8xf32, 128>
  memref.copy %dram, %sram : memref<8xf32> to memref<8xf32, 128>
  return
}

// -----

// An SRAM-to-SRAM copy stays a memref.copy: copy elimination must still be able
// to see it as a copy, so it is not turned into a dispatch here.
// CHECK-LABEL: func.func @sram_to_sram(
// CHECK: memref.copy %{{.*}}, %{{.*}} : memref<8xf32, 128> to memref<8xf32, 128>
// CHECK-NOT: pustream.
func.func @sram_to_sram(%a: memref<8xf32, 128>, %b: memref<8xf32, 128>) {
  memref.copy %a, %b : memref<8xf32, 128> to memref<8xf32, 128>
  return
}
