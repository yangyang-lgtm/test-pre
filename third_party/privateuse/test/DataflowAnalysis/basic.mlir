// RUN: privateuse-opt %s -privateuse-analyze-dataflow -split-input-file -o /dev/null 2>&1 | FileCheck %s

module attributes {privateuse.grid = array<i64: 1, 1, 1>} {
  tt.func public @kernel(%arg0: memref<?xf32>, %arg1: memref<?xf32>) {
    %c0 = arith.constant 0 : index
    %c4 = arith.constant 4 : index
    %c8 = arith.constant 8 : index
    %pid = pumem.get_physical_id : index
    scf.for %i = %pid to %c4 step %c8 {
      %src = memref.reinterpret_cast %arg0 to offset: [%c0], sizes: [256], strides: [1] : memref<?xf32> to memref<256xf32, strided<[1], offset: ?>>
      %alloc = memref.alloc() : memref<256xf32, 128>
      memref.copy %src, %alloc : memref<256xf32, strided<[1], offset: ?>> to memref<256xf32, 128>
      %dst = memref.reinterpret_cast %arg1 to offset: [%c0], sizes: [256], strides: [1] : memref<?xf32> to memref<256xf32, strided<[1], offset: ?>>
      memref.copy %alloc, %dst : memref<256xf32, 128> to memref<256xf32, strided<[1], offset: ?>>
    }
    tt.return
  }
}

// CHECK: func @kernel loops={0:pid}
// CHECK: place p0 kind=kernel_arg space=dram anchor=%arg0
// CHECK: place p1 kind=kernel_arg space=dram anchor=%arg1
// CHECK: place p2 kind=sram_alloc space=sram anchor=%alloc path=pid[4,1] producers=1 consumers=1
// CHECK: transfer t0 dir=D2S src=p0 dst=p2 opKind=copy masked=0 dynamic=0 at=pid[4,2]
// CHECK: transfer t1 dir=S2D src=p2 dst=p1 opKind=copy masked=0 dynamic=0 at=pid[4,4]
// CHECK: station p2 start=pid[4,2] retire=pid[4,4] carried=0 dead=0 through=1 indirect=0
// CHECK: check correctness ok
// CHECK: summary funcs=1 places=3 stations=1 transfers={D2S=1,S2D=1,S2S=0} carriers=0 dead=0 tailWrites=0 unresolved=0

// -----

module attributes {privateuse.grid = array<i64: 1, 1, 1>} {
  tt.func public @if_kernel(%arg0: memref<?xf32>, %arg1: memref<?xf32>, %flag: i1) {
    %c0 = arith.constant 0 : index
    %c4 = arith.constant 4 : index
    %c8 = arith.constant 8 : index
    %c16 = arith.constant 16 : index
    %pid = pumem.get_physical_id : index
    scf.for %p = %pid to %c4 step %c8 {
      %src0 = memref.reinterpret_cast %arg0 to offset: [%c0], sizes: [16], strides: [1] : memref<?xf32> to memref<16xf32, strided<[1], offset: ?>>
      %src1 = memref.reinterpret_cast %arg0 to offset: [%c16], sizes: [16], strides: [1] : memref<?xf32> to memref<16xf32, strided<[1], offset: ?>>
      %dst0 = memref.reinterpret_cast %arg1 to offset: [%c0], sizes: [16], strides: [1] : memref<?xf32> to memref<16xf32, strided<[1], offset: ?>>
      %dst1 = memref.reinterpret_cast %arg1 to offset: [%c16], sizes: [16], strides: [1] : memref<?xf32> to memref<16xf32, strided<[1], offset: ?>>
      %buf0 = memref.alloc() : memref<16xf32, 128>
      %buf1 = memref.alloc() : memref<16xf32, 128>
      scf.if %flag {
        memref.copy %src0, %buf0 : memref<16xf32, strided<[1], offset: ?>> to memref<16xf32, 128>
      } else {
        memref.copy %src1, %buf1 : memref<16xf32, strided<[1], offset: ?>> to memref<16xf32, 128>
      }
      memref.copy %buf0, %dst0 : memref<16xf32, 128> to memref<16xf32, strided<[1], offset: ?>>
      memref.copy %buf1, %dst1 : memref<16xf32, 128> to memref<16xf32, strided<[1], offset: ?>>
    }
    tt.return
  }
}

// CHECK: func @if_kernel loops={0:pid} places=4 transfers=4 carriers=0
// CHECK: place p2 kind=sram_alloc space=sram anchor=%alloc path=pid[5,4] producers=1 consumers=1
// CHECK: place p3 kind=sram_alloc space=sram anchor=%alloc_3 path=pid[5,5] producers=1 consumers=1
// CHECK: transfer t0 dir=D2S src=p0 dst=p2 opKind=copy masked=0 dynamic=0 at=pid.if0.then[5,6,0]
// CHECK: transfer t1 dir=D2S src=p0 dst=p3 opKind=copy masked=0 dynamic=0 at=pid.if0.else[5,6,0]
// CHECK: transfer t2 dir=S2D src=p2 dst=p1 opKind=copy masked=0 dynamic=0 at=pid[5,7]
// CHECK: transfer t3 dir=S2D src=p3 dst=p1 opKind=copy masked=0 dynamic=0 at=pid[5,8]
// CHECK: station p2 start=pid.if0.then[5,6,0] retire=pid[5,7] carried=0 dead=0 through=1 indirect=0
// CHECK: station p3 start=pid.if0.else[5,6,0] retire=pid[5,8] carried=0 dead=0 through=1 indirect=0
// CHECK: check correctness ok
// CHECK: summary funcs=1 places=4 stations=2 transfers={D2S=2,S2D=2,S2S=0} carriers=0 dead=0 tailWrites=0 unresolved=0