// RUN: privateuse-opt %s -arith-to-puarith | FileCheck %s \
// RUN:   --implicit-check-not="{{(^| )arith\\.}}"
// RUN: privateuse-opt %s -arith-to-puarith | FileCheck %s \
// RUN:   --check-prefix=I8 --implicit-check-not="i1"

// Every arith op is rewritten 1:1 into puarith; i1 becomes i8 everywhere
// (signatures included), so the comparison result is an i8.

// CHECK-LABEL: func.func @scalars(
// CHECK-SAME:    %[[A:.*]]: i32, %[[B:.*]]: i32, %[[C:.*]]: f32, %[[D:.*]]: f32) -> (i32, f32, i8) {
// CHECK:         %[[S:.*]] = puarith.addi %[[A]], %[[B]] : i32
// CHECK:         %[[M:.*]] = puarith.mulf %[[C]], %[[D]] : f32
// CHECK:         %[[CMP:.*]] = puarith.cmpi slt, %[[A]], %[[B]] : i32 -> i8
// CHECK:         return %[[S]], %[[M]], %[[CMP]] : i32, f32, i8
func.func @scalars(%a: i32, %b: i32, %c: f32, %d: f32) -> (i32, f32, i1) {
  %s = arith.addi %a, %b : i32
  %m = arith.mulf %c, %d : f32
  %cmp = arith.cmpi slt, %a, %b : i32
  return %s, %m, %cmp : i32, f32, i1
}

// The i1 corner cases:
// I8-LABEL: func.func @i1_ops(
// I8-SAME:    %[[X:.*]]: i32) -> (i8, i32) {
// I8:         %[[ONE8:.*]] = puarith.constant 1 : i8
// I8:         %[[ONE32:.*]] = puarith.constant 1 : i32
// I8:         %[[LOW:.*]] = puarith.andi %[[X]], %[[ONE32]] : i32
// I8:         %[[TRUNC:.*]] = puarith.trunci %[[LOW]] : i32 to i8
// I8:         %[[EXT:.*]] = puarith.extui %[[TRUNC]] : i8 to i32
// I8:         return %[[ONE8]], %[[EXT]] : i8, i32
func.func @i1_ops(%x: i32) -> (i1, i32) {
  %t = arith.constant true
  %low = arith.trunci %x : i32 to i1
  %back = arith.extui %low : i1 to i32
  return %t, %back : i1, i32
}

// A vector comparison computes the mask in the wider element type and then
// narrows it to the promoted i8 mask type.

// CHECK-LABEL: func.func @vectors(
// CHECK-SAME:    %[[A:.*]]: vector<4xi32>, %[[B:.*]]: vector<4xi32>) -> vector<4xi8> {
// CHECK:         %[[CMP:.*]] = puarith.cmpi slt, %[[A]], %[[B]] : vector<4xi32> -> vector<4xi32>
// CHECK:         %[[MASK:.*]] = puarith.trunci %[[CMP]] : vector<4xi32> to vector<4xi8>
// CHECK:         return %[[MASK]] : vector<4xi8>
func.func @vectors(%a: vector<4xi32>, %b: vector<4xi32>) -> vector<4xi1> {
  %cmp = arith.cmpi slt, %a, %b : vector<4xi32>
  return %cmp : vector<4xi1>
}
