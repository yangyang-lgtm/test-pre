// RUN: privateuse-opt %s -scf-to-pumem | FileCheck %s \
// RUN:   --implicit-check-not="scf."

// scf.for becomes pumem.for: the bounds, step and induction variable are
// index-typed in pumem, loop-carried values pass through unchanged.

// CHECK-LABEL: func.func @for_index(
// CHECK-SAME:    %[[LB:.*]]: index, %[[UB:.*]]: index, %[[STEP:.*]]: index) -> f32 {
// CHECK:         %[[INIT:.*]] = arith.constant 0.000000e+00 : f32
// CHECK:         %[[RES:.*]] = pumem.for %[[IV:.*]] = %[[LB]] to %[[UB]] step %[[STEP]]
// CHECK-SAME:      iter_args(%[[ACC:.*]] = %[[INIT]]) -> (f32) {
// CHECK:           %[[NEXT:.*]] = arith.addf %[[ACC]], %[[ACC]] : f32
// CHECK:           pumem.yield %[[NEXT]] : f32
// CHECK:         }
// CHECK:         return %[[RES]] : f32
func.func @for_index(%lb: index, %ub: index, %step: index) -> f32 {
  %c0 = arith.constant 0.0 : f32
  %r = scf.for %i = %lb to %ub step %step iter_args(%acc = %c0) -> (f32) {
    %next = arith.addf %acc, %acc : f32
    scf.yield %next : f32
  }
  return %r : f32
}

// An integer loop keeps its i32 semantics: the bounds are widened to index
// for pumem.for and the induction variable is narrowed back inside the body.

// CHECK-LABEL: func.func @for_int(
// CHECK-SAME:    %[[LB:.*]]: i32, %[[UB:.*]]: i32) -> i32 {
// CHECK-DAG:     %[[UBI:.*]] = arith.index_cast %[[UB]] : i32 to index
// CHECK-DAG:     %[[LBI:.*]] = arith.index_cast %[[LB]] : i32 to index
// CHECK:         %[[RES:.*]] = pumem.for %[[IV:.*]] = %[[LBI]] to %[[UBI]] step %{{[^ ]*}}
// CHECK-SAME:      iter_args(%[[ACC:.*]] = %[[LB]]) -> (i32) {
// CHECK:           %[[IVI:.*]] = arith.index_cast %[[IV]] : index to i32
// CHECK:           %[[NEXT:.*]] = arith.addi %[[ACC]], %[[IVI]] : i32
// CHECK:           pumem.yield %[[NEXT]] : i32
// CHECK:         }
// CHECK:         return %[[RES]] : i32
func.func @for_int(%lb: i32, %ub: i32) -> i32 {
  %c1 = arith.constant 1 : i32
  %r = scf.for %i = %lb to %ub step %c1 iter_args(%acc = %lb) -> (i32) : i32 {
    %next = arith.addi %acc, %i : i32
    scf.yield %next : i32
  }
  return %r : i32
}

// scf.if becomes pumem.if with an i8 condition (the pumem boolean); the i1
// function argument is promoted to i8 as well.

// CHECK-LABEL: func.func @if(
// CHECK-SAME:    %[[COND:.*]]: i8, %[[B:.*]]: i32, %[[C:.*]]: i32) -> i32 {
// CHECK:         %[[RES:.*]] = pumem.if %[[COND]] -> (i32) {
// CHECK:           pumem.yield %[[B]] : i32
// CHECK:         } else {
// CHECK:           pumem.yield %[[C]] : i32
// CHECK:         }
// CHECK:         return %[[RES]] : i32
func.func @if(%a: i1, %b: i32, %c: i32) -> i32 {
  %r = scf.if %a -> (i32) {
    scf.yield %b : i32
  } else {
    scf.yield %c : i32
  }
  return %r : i32
}
