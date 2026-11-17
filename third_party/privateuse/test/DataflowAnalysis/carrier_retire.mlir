// RUN: privateuse-opt %s -privateuse-analyze-dataflow -o /dev/null 2>&1 | FileCheck %s

#map = affine_map<(d0) -> (d0)>
module attributes {privateuse.grid = array<i64: 1, 1, 1>} {
  tt.func public @kernel(%arg0: memref<?xf32>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    %c8 = arith.constant 8 : index
    %pid = pumem.get_physical_id : index
    scf.for %p = %pid to %c8 step %c8 {
      %a = memref.alloc() : memref<16xf32, 128>
      %b = memref.alloc() : memref<16xf32, 128>
      %src = memref.reinterpret_cast %arg0 to offset: [%c0], sizes: [16], strides: [1] : memref<?xf32> to memref<16xf32, strided<[1], offset: ?>>
      memref.copy %src, %a : memref<16xf32, strided<[1], offset: ?>> to memref<16xf32, 128>
      %r = scf.for %k = %c0 to %c4 step %c1 iter_args(%car = %a) -> (memref<16xf32, 128>) {
        linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%car : memref<16xf32, 128>) outs(%b : memref<16xf32, 128>) {
        ^bb0(%in: f32, %out: f32):
          linalg.yield %in : f32
        }
        scf.yield %b : memref<16xf32, 128>
      }
    }
    tt.return
  }
}

// CHECK: carrier c0 loop=1 arg=%arg3 init=p1 yield=p2 rotation=1 inPlace=0 wroteBody=0
// CHECK: station p2 start=pid.arg2[5,4,0] retire=pid.arg2[5,4,0] delta=+1 carried=1 dead=0 through=0 indirect=0
// CHECK: access p2 role=read at=pid.arg2[5,4,0] delta=+1 carrier=c0 branch=yield guard=body
// CHECK: summary funcs=1 places=4 stations=2 transfers={D2S=1,S2D=0,S2S=0} carriers=1 dead=0 tailWrites=0 unresolved=0
