// RUN: privateuse-opt %s -privateuse-analyze-dataflow -o /dev/null 2>&1 | FileCheck %s

module attributes {privateuse.grid = array<i64: 1, 1, 1>} {
  tt.func public @kernel(%arg0: memref<?xf32>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    %c8 = arith.constant 8 : index
    %pid = pumem.get_physical_id : index
    scf.for %p = %pid to %c4 step %c8 {
      %alloc = memref.alloc() : memref<16xf32, 128>
      %res = scf.while (%i = %c0) : (index) -> index {
        %cond = arith.cmpi slt, %i, %c4 : index
        scf.condition(%cond) %i : index
      } do {
      ^bb0(%i: index):
        %src = memref.reinterpret_cast %arg0 to offset: [%i], sizes: [16], strides: [1] : memref<?xf32> to memref<16xf32, strided<[1], offset: ?>>
        memref.copy %src, %alloc : memref<16xf32, strided<[1], offset: ?>> to memref<16xf32, 128>
        %next = arith.addi %i, %c1 : index
        scf.yield %next : index
      }
    }
    tt.return
  }
}

// CHECK: func @kernel deferred=unmodeled_region
// CHECK: summary funcs=1 {{.*}} deferred=1 reasons={unmodeled_region=1}
