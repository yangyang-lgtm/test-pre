// RUN: privateuse-opt %s --expand-strided-metadata -privateuse-memref-to-pumem | FileCheck %s

// A dispatch that passes a slot hands the outlined callee the tile descriptor
// only. The handle and the metadata slot are the caller's bookkeeping -- the
// caller is the side that waits on them -- so they must not appear in the call.
// CHECK-LABEL: func.func private @body(
// CHECK-SAME: !pumem.ptr<f32, 128>
func.func private @body(%a: memref<256xf32, 128>) {
  return
}

// CHECK-LABEL: func.func @kernel(
// CHECK: pumem.runtime_call @body (%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}) : (!pumem.ptr<f32, 128>, index, index, index) -> ()
func.func @kernel(%i: index) {
  %buf = memref.alloc() {pustream.stages = 2 : i32} : memref<2x256xf32, 128>
  %s = pustream.slot %buf[%i] : memref<2x256xf32, 128> -> !pustream.slot<memref<256xf32, 128>>
  pustream.async.dispatch @body(%s) {reads = array<i64: 0>, writes = array<i64>} : !pustream.slot<memref<256xf32, 128>>
  return
}

// -----

// A runtime function takes one address per buffer, so a non-zero offset is
// folded into the pointer instead of being dropped.
// CHECK-LABEL: func.func @runtime_callee(
// CHECK: %[[P:.*]] = pumem.addptr
// CHECK: pumem.runtime_call @runtime_foo (%[[P]]) : (!pumem.ptr<f32, 0>) -> ()
func.func @runtime_callee(%p: memref<256xf32>, %i: index) {
  %view = memref.subview %p[%i] [128] [1] : memref<256xf32> to memref<128xf32, strided<[1], offset: ?>>
  pustream.async.dispatch @runtime_foo(%view) {reads = array<i64: 0>, writes = array<i64>} : memref<128xf32, strided<[1], offset: ?>>
  return
}
