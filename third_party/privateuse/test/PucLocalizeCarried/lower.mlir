// RUN: privateuse-opt %s -puc-localize-carried | FileCheck %s

// A carried loop value becomes a mutable local: the loop drops its result,
// the yield its operand, and the carried update turns into an assignment.

// CHECK-LABEL: puc.func @loop(
// CHECK-SAME:    %[[N:.*]]: index, %[[INIT:.*]]: f32) -> f32 {
// CHECK: %[[ACC:.*]] = puc.local %[[INIT]] : f32
// CHECK: puc.for %{{.*}} = %{{.*}} to %[[N]] step %{{.*}} {
// CHECK:   %[[NEXT:.*]] = puc.call @scalar_addf(%[[ACC]], %[[ACC]]) : (f32, f32) -> f32
// CHECK:   %[[STAGED:.*]] = puc.local %[[NEXT]] : f32
// CHECK:   puc.assign %[[ACC]], %[[STAGED]] : f32, f32
// CHECK: }
// CHECK: puc.return %[[ACC]] : f32
puc.func @loop(%n: index, %init: f32) -> f32 {
  %c0 = puc.constant 0 : index
  %c1 = puc.constant 1 : index
  %r = puc.for %i = %c0 to %n step %c1 iter_args(%acc = %init) -> (f32) {
    %next = puc.call @scalar_addf(%acc, %acc) : (f32, f32) -> f32
    puc.yield %next : f32
  }
  puc.return %r : f32
}

// A value carried through unchanged needs no assignment.

// CHECK-LABEL: puc.func @identity(
// CHECK: %[[ACC:.*]] = puc.local %{{.*}} : index
// CHECK: puc.for
// CHECK-NOT: puc.assign
// CHECK: puc.return %[[ACC]] : index
puc.func @identity(%n: index, %init: index) -> index {
  %c0 = puc.constant 0 : index
  %c1 = puc.constant 1 : index
  %r = puc.for %i = %c0 to %n step %c1 iter_args(%acc = %init) -> (index) {
    puc.yield %acc : index
  }
  puc.return %r : index
}

// An if result gets one assignment per branch.

// CHECK-LABEL: puc.func @branch(
// CHECK: %[[V:.*]] = puc.local : f32
// CHECK: puc.if %{{.*}} {
// CHECK:   puc.assign %[[V]], %{{.*}} : f32, f32
// CHECK: } else {
// CHECK:   puc.assign %[[V]], %{{.*}} : f32, f32
// CHECK: }
// CHECK: puc.return %[[V]] : f32
puc.func @branch(%cond: i8, %a: f32, %b: f32) -> f32 {
  %r = puc.if %cond -> (f32) {
    puc.yield %a : f32
  } else {
    puc.yield %b : f32
  }
  puc.return %r : f32
}

// A carried value nobody reads leaves no local behind.

// CHECK-LABEL: puc.func @dead(
// CHECK-NOT: puc.local
// CHECK: puc.return
puc.func @dead(%n: index, %init: index) {
  %c0 = puc.constant 0 : index
  %c1 = puc.constant 1 : index
  %r = puc.for %i = %c0 to %n step %c1 iter_args(%acc = %init) -> (index) {
    puc.yield %acc : index
  }
  puc.return
}
