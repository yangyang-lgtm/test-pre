// RUN: privateuse-opt %s -privateuse-analyze-dataflow -o /dev/null 2>&1 | FileCheck %s

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
        memref.copy %car, %b : memref<16xf32, 128> to memref<16xf32, 128>
        scf.yield %b : memref<16xf32, 128>
      }
    }
    tt.return
  }
}

// CHECK: func @kernel loops={0:pid,1:static(trip=4)}
// CHECK: place p1 kind=sram_alloc space=sram anchor=%alloc path=pid[5,0] producers=1 consumers=1
// CHECK: place p2 kind=sram_alloc space=sram anchor=%alloc_0 path=pid[5,1] producers=2 consumers=1
// CHECK: place p3 kind=loop_carried space=unknown anchor=%arg3 path=pid[5,4]
// CHECK: transfer t1 dir=S2S src=p1 dst=p2 opKind=copy masked=0 dynamic=0 at=pid.arg2[5,4,0] carrier=c0 branch=init guard=first
// CHECK: transfer t2 dir=S2S src=p2 dst=p2 opKind=copy masked=0 dynamic=0 at=pid.arg2[5,4,0] delta=+1 carrier=c0 branch=yield guard=body
// CHECK: carrier c0 loop=1 arg=%arg3 init=p1 yield=p2 rotation=1 inPlace=0 wroteBody=0
// CHECK: station p2 start=pid.arg2[5,4,0] retire=pid.arg2[5,4,0] delta=+1 carried=1 dead=0 through=1 indirect=0
// CHECK: summary funcs=1 places=4 stations=2 transfers={D2S=1,S2D=0,S2S=2} carriers=1 dead=0 tailWrites=0 unresolved=0
