// RUN: privateuse-opt %s -privateuse-analyze-dataflow -o /dev/null 2>&1 | FileCheck %s

module attributes {privateuse.grid = array<i64: 1, 1, 1>} {
  tt.func public @kernel(%arg0: memref<?xf32>, %arg1: memref<?xf32>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    %pid = pumem.get_physical_id : index
    %ub = arith.addi %pid, %c4 : index
    scf.for %p = %c0 to %ub step %c1 {
      %alloc = memref.alloc() : memref<16xf32, 128>
      %src = memref.reinterpret_cast %arg0 to offset: [%c0], sizes: [16], strides: [1] : memref<?xf32> to memref<16xf32, strided<[1], offset: ?>>
      memref.copy %src, %alloc : memref<16xf32, strided<[1], offset: ?>> to memref<16xf32, 128>
      %dst = memref.reinterpret_cast %arg1 to offset: [%c0], sizes: [16], strides: [1] : memref<?xf32> to memref<16xf32, strided<[1], offset: ?>>
      memref.copy %alloc, %dst : memref<16xf32, 128> to memref<16xf32, strided<[1], offset: ?>>
    }
    tt.return
  }
}

// CHECK: func @kernel deferred=symbolic_trip
// CHECK: summary funcs=1 {{.*}} deferred=1 reasons={symbolic_trip=1}
