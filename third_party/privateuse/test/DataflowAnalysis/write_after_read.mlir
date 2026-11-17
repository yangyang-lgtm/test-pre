// RUN: privateuse-opt %s -privateuse-analyze-dataflow -o /dev/null 2>&1 | FileCheck %s

module attributes {privateuse.grid = array<i64: 1, 1, 1>} {
  tt.func public @kernel(%arg0: memref<?xf32>) {
    %c0 = arith.constant 0 : index
    %c4 = arith.constant 4 : index
    %c8 = arith.constant 8 : index
    %zero = arith.constant 0.0 : f32
    %pid = pumem.get_physical_id : index
    scf.for %p = %pid to %c4 step %c8 {
      %alloc = memref.alloc() : memref<16xf32, 128>
      %src = memref.reinterpret_cast %arg0 to offset: [%c0], sizes: [16], strides: [1] : memref<?xf32> to memref<16xf32, strided<[1], offset: ?>>
      memref.copy %src, %alloc : memref<16xf32, strided<[1], offset: ?>> to memref<16xf32, 128>
      %v = memref.load %alloc[%c0] : memref<16xf32, 128>
      memref.store %zero, %alloc[%c0] : memref<16xf32, 128>
    }
    tt.return
  }
}

// CHECK: func @kernel deferred=scalar_write_to_sram
// CHECK: summary funcs=1 {{.*}} deferred=1 reasons={scalar_write_to_sram=1}
