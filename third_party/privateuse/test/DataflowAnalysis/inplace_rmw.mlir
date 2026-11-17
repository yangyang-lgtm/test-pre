// RUN: privateuse-opt %s -privateuse-analyze-dataflow -o /dev/null 2>&1 | FileCheck %s

#map = affine_map<(d0) -> (d0)>
module attributes {privateuse.grid = array<i64: 1, 1, 1>} {
  tt.func public @kernel(%arg0: memref<?xf32>, %arg1: memref<?xf32>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    %c8 = arith.constant 8 : index
    %pid = pumem.get_physical_id : index
    scf.for %p = %pid to %c4 step %c8 {
      %a = memref.alloc() : memref<16xf32, 128>
      %src = memref.reinterpret_cast %arg0 to offset: [%c0], sizes: [16], strides: [1] : memref<?xf32> to memref<16xf32, strided<[1], offset: ?>>
      memref.copy %src, %a : memref<16xf32, strided<[1], offset: ?>> to memref<16xf32, 128>
      %r = scf.for %k = %c0 to %c4 step %c1 iter_args(%car = %a) -> (memref<16xf32, 128>) {
        linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%car : memref<16xf32, 128>) outs(%car : memref<16xf32, 128>) {
        ^bb0(%in: f32, %out: f32):
          linalg.yield %in : f32
        }
        scf.yield %car : memref<16xf32, 128>
      }
      %dst = memref.reinterpret_cast %arg1 to offset: [%c0], sizes: [16], strides: [1] : memref<?xf32> to memref<16xf32, strided<[1], offset: ?>>
      memref.copy %r, %dst : memref<16xf32, 128> to memref<16xf32, strided<[1], offset: ?>>
    }
    tt.return
  }
}

// CHECK: transfer t1 dir=S2D src=p2 dst=p1 opKind=copy masked=0 dynamic=0 at=pid[5,5]
// CHECK: carrier c0 loop=1 arg=%arg4 init=p2 yield=p2 rotation=0 inPlace=1 wroteBody=1
// CHECK: station p2 start=pid[5,2] retire=pid[5,5] carried=1 dead=0 through=0 indirect=0
// CHECK: access p2 role=read at=pid.arg3[5,3,0] delta=+1 carrier=c0 branch=yield guard=body
// CHECK: summary funcs=1 places=4 stations=1 transfers={D2S=1,S2D=1,S2S=0} carriers=1 dead=0 tailWrites=0 unresolved=0
