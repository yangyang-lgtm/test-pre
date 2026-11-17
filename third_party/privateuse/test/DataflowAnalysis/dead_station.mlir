// RUN: privateuse-opt %s -privateuse-analyze-dataflow -o /dev/null 2>&1 | FileCheck %s

module attributes {privateuse.grid = array<i64: 1, 1, 1>} {
  tt.func public @kernel(%arg0: memref<?xf32>, %size: index) {
    %c0 = arith.constant 0 : index
    %c4 = arith.constant 4 : index
    %c8 = arith.constant 8 : index
    %zero = arith.constant 0.0 : f32
    %pid = pumem.get_physical_id : index
    scf.for %p = %pid to %c4 step %c8 {
      %alloc = memref.alloc() : memref<128xf32, 128>
      linalg.fill ins(%zero : f32) outs(%alloc : memref<128xf32, 128>)
      %src = memref.subview %arg0[%c0] [%size] [1] : memref<?xf32> to memref<?xf32, strided<[1], offset: ?>>
      %dst = memref.subview %alloc[%c0] [%size] [1] : memref<128xf32, 128> to memref<?xf32, strided<[1], offset: ?>, 128>
      memref.copy %src, %dst : memref<?xf32, strided<[1], offset: ?>> to memref<?xf32, strided<[1], offset: ?>, 128>
    }
    tt.return
  }
}

// CHECK: place p1 kind=sram_alloc space=sram anchor=%alloc path=pid[5,0] producers=2 consumers=0
// CHECK: station p1 start=pid[5,1] retire=pid[5,4] carried=0 dead=1 through=0 indirect=0
// CHECK: summary funcs=1 places=2 stations=1 transfers={D2S=1,S2D=0,S2S=0} carriers=0 dead=1 tailWrites=1 unresolved=0
