// RUN: privateuse-opt %s -puc-void-ptr-args | FileCheck %s

// Pointer arguments become void pointers and are cast back at the start of
// the function; scalar arguments, arguments that are already void pointers and
// external declarations stay untouched.

// CHECK-LABEL: puc.func private @helper(!pumem.ptr<f32, 0>, index) -> index

puc.func private @helper(!pumem.ptr<f32, 0>, index) -> index

// CHECK-LABEL: puc.func @kernel(
// CHECK-SAME:    %[[DATA:.*]]: !pumem.ptr<void, 128>, %[[IDX:.*]]: !pumem.ptr<void, 128>, %[[N:.*]]: index
// CHECK:         %[[DATA_CAST:.*]] = puc.point_cast %[[DATA]] : !pumem.ptr<void, 128> -> !pumem.ptr<f32, 128>
// CHECK:         %[[IDX_CAST:.*]] = puc.point_cast %[[IDX]] : !pumem.ptr<void, 128> -> !pumem.ptr<i32, 128>
// CHECK:         %{{.*}} = puc.load %[[DATA_CAST]]
// CHECK:         puc.store %{{.*}}, %[[IDX_CAST]]
// CHECK:         puc.return

puc.func @kernel(%data: !pumem.ptr<f32, 128>, %idx: !pumem.ptr<i32, 128>, %n: index) {
  %v = puc.load %data : !pumem.ptr<f32, 128> -> f32
  %i = puc.constant 1 : i32
  puc.store %i, %idx : i32, !pumem.ptr<i32, 128>
  puc.return
}

// CHECK-LABEL: puc.func @already_void(%{{.*}}: !pumem.ptr<void, 128>, %{{.*}}: index) {
// CHECK-NOT:     puc.point_cast
// CHECK:         puc.return

puc.func @already_void(%p: !pumem.ptr<void, 128>, %n: index) {
  puc.return
}
