// RUN: privateuse-opt %s -puc-simplify-assign | FileCheck %s

// A constant local whose assignments always store the same constant is
// replaced by the constant and erased together with its assignments.

// CHECK-LABEL: puc.func @constant(
// CHECK-NOT: puc.local
// CHECK-NOT: puc.assign
// CHECK: puc.return %[[C7:.*]] : index
puc.func @constant() -> index {
  %c7 = puc.constant 7 : index
  %v = puc.local %c7 : index
  %c0 = puc.constant 0 : index
  %c1 = puc.constant 1 : index
  puc.for %i = %c0 to %c1 step %c1 {
    puc.assign %v, %c7 : index, index
  }
  puc.return %v : index
}

// Self assignments are dropped.

// CHECK-LABEL: puc.func @self(
// CHECK: %[[V:.*]] = puc.local %{{.*}} : index
// CHECK-NOT: puc.assign
// CHECK: puc.return %[[V]] : index
puc.func @self(%init: index) -> index {
  %v = puc.local %init : index
  puc.assign %v, %v : index, index
  puc.return %v : index
}

// A never-written staging local that only feeds one assignment is inlined.

// CHECK-LABEL: puc.func @staging(
// CHECK: %[[V:.*]] = puc.local %{{.*}} : index
// CHECK: puc.assign %[[V]], %[[OTHER:.*]] : index, index
// CHECK-NOT: puc.local
// CHECK: puc.return %[[V]] : index
puc.func @staging(%init: index, %other: index) -> index {
  %v = puc.local %init : index
  %staging = puc.local %other : index
  puc.assign %v, %staging : index, index
  puc.return %v : index
}

// A staging copy protecting a parallel assignment stays.

// CHECK-LABEL: puc.func @swap(
// CHECK: %[[A:.*]] = puc.local %{{.*}} : index
// CHECK: %[[B:.*]] = puc.local %{{.*}} : index
// CHECK: %[[TMP:.*]] = puc.local %[[B]] : index
// CHECK: puc.assign %[[B]], %[[A]] : index, index
// CHECK: puc.assign %[[A]], %[[TMP]] : index, index
// CHECK: puc.return %[[A]], %[[B]] : index, index
puc.func @swap(%a0: index, %b0: index) -> (index, index) {
  %a = puc.local %a0 : index
  %b = puc.local %b0 : index
  %tmp = puc.local %b : index
  puc.assign %b, %a : index, index
  puc.assign %a, %tmp : index, index
  puc.return %a, %b : index, index
}
