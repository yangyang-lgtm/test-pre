// RUN: privateuse-opt %s -privateuse-memref-to-pumem | FileCheck %s \
// RUN:   --implicit-check-not=memref. --implicit-check-not=pumem.tile_gather \
// RUN:   --implicit-check-not=pumem.tile_scatter

// Every memref becomes the scalar tuple
//   (ptr, offset, size_0 .. size_{r-1}, stride_0 .. stride_{r-1})
// and every buffer op becomes its pumem counterpart.

// CHECK-LABEL: func.func @basic()
tt.func @basic() {
  // CHECK: pumem.alloc {{.*}}, {{.*}} : !pumem.ptr<void, 128>
  // CHECK: pumem.point_cast {{.*}} : !pumem.ptr<void, 128> -> !pumem.ptr<f32, 128>
  %buf = memref.alloc() : memref<8xf32, 128>
  %c0 = arith.constant 0 : index
  // CHECK: pumem.load
  %v = memref.load %buf[%c0] : memref<8xf32, 128>
  %w = arith.addf %v, %v : f32
  // CHECK: pumem.store
  memref.store %w, %buf[%c0] : memref<8xf32, 128>
  // memref.dim folds to the size tuple element, memref.rank to a constant.
  %d = memref.dim %buf, %c0 : memref<8xf32, 128>
  %r = memref.rank %buf : memref<8xf32, 128>
  // CHECK: pumem.dealloc
  memref.dealloc %buf : memref<8xf32, 128>
  tt.return
}

// An explicit allocation alignment wins over the default one.

// CHECK-LABEL: func.func @aligned()
tt.func @aligned() {
  // CHECK: pumem.alloc {{.*}}, {{.*}} : !pumem.ptr<void, 128>
  %b = memref.alloc() {alignment = 256} : memref<8xf32, 128>
  memref.dealloc %b : memref<8xf32, 128>
  tt.return
}

// A copy between a DRAM and an SRAM buffer keeps both spaces and forwards the
// sizes and both stride sets.

// CHECK-LABEL: func.func @copy(
// CHECK-SAME: %[[DST:.*]]: !pumem.ptr<f32, 0>
// CHECK-SAME: %[[SRC:.*]]: !pumem.ptr<f32, 128>
tt.func @copy(%dst: memref<4x8xf32, 0>, %src: memref<4x8xf32, 128>) {
  // CHECK: pumem.copy %{{.*}}, %{{.*}}, sizes = ({{.*}}, {{.*}}), dst_strides = ({{.*}}, {{.*}}), src_strides = ({{.*}}, {{.*}}), byte_size %{{.*}} {dir = "sram_to_dram"} : !pumem.ptr<f32, 0>, !pumem.ptr<f32, 128>
  memref.copy %src, %dst : memref<4x8xf32, 128> to memref<4x8xf32, 0>
  tt.return
}

// A memref without a memory space lives in DRAM, an allocation without an
// alignment takes kDefaultAlignment.

// CHECK-LABEL: func.func @default_space()
tt.func @default_space() {
  // CHECK: pumem.alloc {{.*}}, {{.*}} : !pumem.ptr<void, 0>
  %b = memref.alloc() : memref<8xf32>
  memref.dealloc %b : memref<8xf32>
  tt.return
}

// The memref-form tile gather / scatter ops become the flat ops: every memref
// operand is unpacked into its tuple, its offset is folded into the pointer
// with pumem.addptr, the tile buffers share the sizes of the tile and the flat
// buffer contributes only its stride.

// CHECK-LABEL: func.func @gather(
tt.func @gather(%src: memref<?xf32, 0>, %idx: memref<4xi32, 128>, %dst: memref<4xf32, 128>) {
  // CHECK-DAG: %[[SRC:.*]] = pumem.addptr %{{.*}}, %{{.*}} : !pumem.ptr<f32, 0>
  // CHECK-DAG: %[[IDX:.*]] = pumem.addptr %{{.*}}, %{{.*}} : !pumem.ptr<i32, 128>
  // CHECK-DAG: %[[DST:.*]] = pumem.addptr %{{.*}}, %{{.*}} : !pumem.ptr<f32, 128>
  // CHECK: pumem.gather %[[SRC]], %{{.*}}, %[[IDX]], %[[DST]], sizes = ({{.*}}), indices_strides = ({{.*}}), destination_strides = ({{.*}}) : !pumem.ptr<f32, 0>, !pumem.ptr<i32, 128>, !pumem.ptr<f32, 128>
  pumem.tile_gather %src, %idx, %dst : memref<?xf32, 0>, memref<4xi32, 128>, memref<4xf32, 128>
  tt.return
}

// The mask tuple and the scalar other operand keep their positions.

// CHECK-LABEL: func.func @masked_gather(
tt.func @masked_gather(
  %src: memref<?xf32, 0>, %idx: memref<4xi32, 128>, %dst: memref<4xf32, 128>,
  %mask: memref<4xi1, 128>, %other: f32
) {
  // CHECK: pumem.gather {{.*}}mask = {{.*}}, mask_strides = ({{.*}}), other {{.*}} : !pumem.ptr<f32, 0>, !pumem.ptr<i32, 128>, !pumem.ptr<f32, 128>, !pumem.ptr<i1, 128>, f32
  pumem.tile_gather %src, %idx, %dst, %mask, %other
    : memref<?xf32, 0>, memref<4xi32, 128>, memref<4xf32, 128>, memref<4xi1, 128>, f32
  tt.return
}

// CHECK-LABEL: func.func @scatter(
tt.func @scatter(%src: memref<4xf32, 128>, %idx: memref<4xi32, 128>, %dst: memref<?xf32, 0>) {
  // CHECK: pumem.scatter {{.*}}sizes = ({{.*}}), source_strides = ({{.*}}), indices_strides = ({{.*}}) : !pumem.ptr<f32, 128>, !pumem.ptr<i32, 128>, !pumem.ptr<f32, 0>
  pumem.tile_scatter %src, %idx, %dst : memref<4xf32, 128>, memref<4xi32, 128>, memref<?xf32, 0>
  tt.return
}

// CHECK-LABEL: func.func @masked_scatter(
tt.func @masked_scatter(
  %src: memref<4xf32, 128>, %idx: memref<4xi32, 128>, %dst: memref<?xf32, 0>, %mask: memref<4xi1, 128>
) {
  // CHECK: pumem.scatter {{.*}}mask = {{.*}}, mask_strides = ({{.*}}) : !pumem.ptr<f32, 128>, !pumem.ptr<i32, 128>, !pumem.ptr<f32, 0>, !pumem.ptr<i1, 128>
  pumem.tile_scatter %src, %idx, %dst, %mask
    : memref<4xf32, 128>, memref<4xi32, 128>, memref<?xf32, 0>, memref<4xi1, 128>
  tt.return
}
