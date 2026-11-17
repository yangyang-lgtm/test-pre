// RUN: privateuse-opt %s -privateuse-analyze-dataflow -o - 2>/dev/null > %t.out
// RUN: tail -n +3 %s | diff - %t.out
module attributes {privateuse.grid = array<i64: 1, 1, 1>} {
  tt.func public @kernel(%arg0: memref<?xf32>, %arg1: memref<?xf32>) {
    %c0 = arith.constant 0 : index
    %c4 = arith.constant 4 : index
    %c8 = arith.constant 8 : index
    %0 = pumem.get_physical_id : index
    scf.for %arg2 = %0 to %c4 step %c8 {
      %reinterpret_cast = memref.reinterpret_cast %arg0 to offset: [%c0], sizes: [256], strides: [1] : memref<?xf32> to memref<256xf32, strided<[1], offset: ?>>
      %alloc = memref.alloc() : memref<256xf32, 128>
      memref.copy %reinterpret_cast, %alloc : memref<256xf32, strided<[1], offset: ?>> to memref<256xf32, 128>
      %reinterpret_cast_0 = memref.reinterpret_cast %arg1 to offset: [%c0], sizes: [256], strides: [1] : memref<?xf32> to memref<256xf32, strided<[1], offset: ?>>
      memref.copy %alloc, %reinterpret_cast_0 : memref<256xf32, 128> to memref<256xf32, strided<[1], offset: ?>>
    }
    tt.return
  }
}

