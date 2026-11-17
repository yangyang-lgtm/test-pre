// RUN: privateuse-opt %s -privateuse-analyze-dataflow -o /dev/null 2>&1 | FileCheck %s

module attributes {privateuse.grid = array<i64: 1, 1, 1>} {
  tt.func public @kernel(%arg0: memref<?xf32>, %arg1: memref<?xf32>, %arg2: memref<?xi32>, %arg3: memref<?xi1>) {
    %c0 = arith.constant 0 : index
    %c4 = arith.constant 4 : index
    %c8 = arith.constant 8 : index
    %pid = pumem.get_physical_id : index
    scf.for %p = %pid to %c8 step %c8 {
      %idx = memref.alloc() : memref<16xi32, 128>
      %tile = memref.alloc() : memref<16xf32, 128>
      %mask = memref.alloc() : memref<16xi1, 128>
      %isrc = memref.reinterpret_cast %arg2 to offset: [%c0], sizes: [16], strides: [1] : memref<?xi32> to memref<16xi32, strided<[1], offset: ?>>
      memref.copy %isrc, %idx : memref<16xi32, strided<[1], offset: ?>> to memref<16xi32, 128>
      %msrc = memref.reinterpret_cast %arg3 to offset: [%c0], sizes: [16], strides: [1] : memref<?xi1> to memref<16xi1, strided<[1], offset: ?>>
      memref.copy %msrc, %mask : memref<16xi1, strided<[1], offset: ?>> to memref<16xi1, 128>
      pumem.tile_gather %arg0, %idx, %tile, %mask : memref<?xf32>, memref<16xi32, 128>, memref<16xf32, 128>, memref<16xi1, 128>
      %dst = memref.reinterpret_cast %arg1 to offset: [%c0], sizes: [16], strides: [1] : memref<?xf32> to memref<16xf32, strided<[1], offset: ?>>
      pumem.tile_scatter %tile, %idx, %dst : memref<16xf32, 128>, memref<16xi32, 128>, memref<16xf32, strided<[1], offset: ?>>
    }
    tt.return
  }
}

// CHECK: transfer t2 dir=D2S src=p0 dst=p5 opKind=gather masked=1 dynamic=0 at=pid[4,7]
// CHECK: transfer t3 dir=S2D src=p5 dst=p1 opKind=scatter masked=0 dynamic=0 at=pid[4,9]
// CHECK: station p4 start=pid[4,4] retire=pid[4,9] carried=0 dead=0 through=0 indirect=1
// CHECK: access p4 role=read at=pid[4,7] delta=0
// CHECK: access p4 role=read at=pid[4,9] delta=0
// CHECK: station p5 start=pid[4,7] retire=pid[4,9] carried=0 dead=0 through=0 indirect=1
// CHECK: summary funcs=1 places=7 stations=3 transfers={D2S=3,S2D=1,S2S=0} carriers=0 dead=0 tailWrites=0 unresolved=0
