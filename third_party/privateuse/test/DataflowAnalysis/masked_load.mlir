// RUN: privateuse-opt %s -privateuse-analyze-dataflow -o /dev/null 2>&1 | FileCheck %s

module attributes {privateuse.grid = array<i64: 1, 1, 1>} {
  tt.func public @kernel(%arg0: memref<?xf32>, %arg1: memref<?xf32>, %size: index) {
    %c0 = arith.constant 0 : index
    %c4 = arith.constant 4 : index
    %c1 = arith.constant 1 : index
    %zero = arith.constant 0.0 : f32
    %pid = pumem.get_physical_id : index
    scf.for %i = %pid to %c4 step %c1 {
      %src = memref.subview %arg0[%c0] [%size] [1] : memref<?xf32> to memref<?xf32, strided<[1], offset: ?>>
      %alloc = memref.alloc() : memref<128xf32, 128>
      linalg.fill ins(%zero : f32) outs(%alloc : memref<128xf32, 128>)
      %dst = memref.subview %alloc[%c0] [%size] [1] : memref<128xf32, 128> to memref<?xf32, strided<[1], offset: ?>, 128>
      memref.copy %src, %dst : memref<?xf32, strided<[1], offset: ?>> to memref<?xf32, strided<[1], offset: ?>, 128>
      %out = memref.subview %arg1[%c0] [%size] [1] : memref<?xf32> to memref<?xf32, strided<[1], offset: ?>>
      memref.copy %alloc, %out : memref<128xf32, 128> to memref<?xf32, strided<[1], offset: ?>>
    }
    tt.return
  }
}

// CHECK: place p2 kind=sram_alloc space=sram anchor=%alloc path=pid[5,1] producers=2 consumers=1
// CHECK: transfer t0 dir=D2S src=p0 dst=p2 opKind=copy masked=0 dynamic=1 group=0 at=pid[5,4]
// CHECK: station p2 start=pid[5,2] retire=pid[5,6] carried=0 dead=0 through=0 indirect=0
// CHECK: access p2 role=write at=pid[5,2] delta=0
// CHECK: summary funcs=1 places=3 stations=1 transfers={D2S=1,S2D=1,S2S=0} carriers=0 dead=0 tailWrites=0 unresolved=0
