// RUN: privateuse-opt %s -reconcile-unrealized-casts -privateuse-to-puc | FileCheck %s \
// RUN:   --implicit-check-not="{{(^| )(func|pumem|puarith|pumath|arith)\\.}}"

// The whole program collapses into the puc emission dialect: the function
// layer, pointer arithmetic and scalar memory access are converted in place,
// the structured control flow becomes puc.for / puc.if / puc.yield, and
// everything else becomes a bare-named runtime call.

// CHECK-LABEL: puc.func @memory(
// CHECK-SAME:    %[[SRC:.*]]: !pumem.ptr<f32, 0>, %[[N:.*]]: index) {
func.func @memory(%src: !pumem.ptr<f32, 0>, %n: index) {
  %c4 = puarith.constant 4 : index
  %c128 = puarith.constant 128 : index
  // CHECK: %[[TID:.*]] = puc.call @get_physical_id() : () -> index
  %tid = pumem.get_physical_id : index
  // CHECK: %[[BUF:.*]] = puc.call @alloc(%{{.*}}, %{{.*}}) : (index, index) -> !pumem.ptr<void, 128>
  %buf = pumem.alloc %c4, %c128 : !pumem.ptr<void, 128>
  // CHECK: %[[P:.*]] = puc.point_cast %[[BUF]] : !pumem.ptr<void, 128> -> !pumem.ptr<f32, 128>
  %p = pumem.point_cast %buf : !pumem.ptr<void, 128> -> !pumem.ptr<f32, 128>
  // CHECK: %[[Q:.*]] = puc.addptr %[[P]], %[[TID]] : !pumem.ptr<f32, 128>
  %q = pumem.addptr %p, %tid : !pumem.ptr<f32, 128>
  // CHECK: %[[V:.*]] = puc.load %[[Q]] : !pumem.ptr<f32, 128> -> f32
  %v = pumem.load %q : !pumem.ptr<f32, 128> -> f32
  // CHECK: puc.store %[[V]], %[[Q]] : f32, !pumem.ptr<f32, 128>
  pumem.store %v, %q : f32, !pumem.ptr<f32, 128>
  // CHECK: puc.call @copy_dram_to_sram<["1"]>(%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}) : (!pumem.ptr<f32, 128>, !pumem.ptr<f32, 0>, index, index, index, index) -> ()
  pumem.copy %q, %src, sizes = (%n), dst_strides = (%n), src_strides = (%n), byte_size %c4
    : !pumem.ptr<f32, 128>, !pumem.ptr<f32, 0>
  // CHECK: puc.call @dealloc(%[[BUF]]) : (!pumem.ptr<void, 128>) -> ()
  pumem.dealloc %buf : !pumem.ptr<void, 128>
  // CHECK: puc.return
  func.return
}

// The gather operand order is the pumem operand order, the rank and the
// presence of the mask / other operand travel as template arguments.

// CHECK-LABEL: puc.func @gather(
// CHECK-SAME:    %[[SRC:.*]]: !pumem.ptr<f32, 0>, %[[STRIDE:.*]]: index, %[[IDX:.*]]: !pumem.ptr<i32, 128>, %[[DST:.*]]: !pumem.ptr<f32, 128>) {
func.func @gather(%src: !pumem.ptr<f32, 0>, %stride: index, %idx: !pumem.ptr<i32, 128>, %dst: !pumem.ptr<f32, 128>) {
  %n = puarith.constant 8 : index
  %c1 = puarith.constant 1 : index
  // CHECK: puc.call @gather<["1", "false", "false"]>(%[[SRC]], %[[STRIDE]], %[[IDX]], %[[DST]], %{{.*}}, %{{.*}}, %{{.*}}) : (!pumem.ptr<f32, 0>, index, !pumem.ptr<i32, 128>, !pumem.ptr<f32, 128>, index, index, index) -> ()
  pumem.gather %src, %stride, %idx, %dst, sizes = (%n), indices_strides = (%c1), destination_strides = (%c1)
    : !pumem.ptr<f32, 0>, !pumem.ptr<i32, 128>, !pumem.ptr<f32, 128>
  func.return
}

// CHECK-LABEL: puc.func @runtime_call(
func.func @runtime_call(%a: !pumem.ptr<f32, 128>, %b: !pumem.ptr<f32, 128>, %c: !pumem.ptr<f32, 128>) {
  // CHECK: puc.call @runtime_matmul<["float", "4", "16", "8"]>(%{{.*}}, %{{.*}}, %{{.*}}) : (!pumem.ptr<f32, 128>, !pumem.ptr<f32, 128>, !pumem.ptr<f32, 128>) -> ()
  pumem.runtime_call @runtime_matmul<[f32, 4 : index, 16 : index, 8 : index]>(%a, %b, %c)
    : (!pumem.ptr<f32, 128>, !pumem.ptr<f32, 128>, !pumem.ptr<f32, 128>) -> ()
  func.return
}

// Value computation becomes scalar_<mnemonic> calls, casts append the source
// and destination type suffixes, comparisons append the predicate, and index
// casts keep the sentinel callee.

// CHECK-LABEL: puc.func @scalars(
// CHECK-SAME:    %[[A:.*]]: i32, %[[B:.*]]: i32, %[[F:.*]]: f32) -> (i32, i8, i64, index, f32, f32) {
func.func @scalars(%a: i32, %b: i32, %f: f32) -> (i32, i8, i64, index, f32, f32) {
  // CHECK: %[[SUM:.*]] = puc.call @scalar_addi(%[[A]], %[[B]]) : (i32, i32) -> i32
  %sum = puarith.addi %a, %b : i32
  // CHECK: %[[CMP:.*]] = puc.call @scalar_cmpi_slt(%[[SUM]], %[[A]]) : (i32, i32) -> i8
  %cmp = puarith.cmpi slt, %sum, %a : i32 -> i8
  // CHECK: %[[EXT:.*]] = puc.call @scalar_extsi_i32_i64(%[[SUM]]) : (i32) -> i64
  %ext = puarith.extsi %sum : i32 to i64
  // CHECK: %[[IDX:.*]] = puc.call @"/*index cast*/"(%[[EXT]]) : (i64) -> index
  %idx = puarith.index_cast %ext : i64 to index
  // CHECK: %[[ROOT:.*]] = puc.call @scalar_sqrt(%[[F]]) : (f32) -> f32
  %root = pumath.sqrt %f : f32
  // CHECK: %[[ABS:.*]] = puc.call @scalar_absf(%[[ROOT]]) : (f32) -> f32
  %abs = pumath.absf %root : f32
  // CHECK: puc.return %[[SUM]], %[[CMP]], %[[EXT]], %[[IDX]], %[[ROOT]], %[[ABS]]
  func.return %sum, %cmp, %ext, %idx, %root, %abs : i32, i8, i64, index, f32, f32
}

// The structured control flow is moved over to puc together with its body.

// CHECK-LABEL: puc.func @control_flow(
// CHECK-SAME:    %[[N:.*]]: index, %[[ACC:.*]]: index, %[[COND:.*]]: i8) -> index {
func.func @control_flow(%n: index, %acc: index, %cond: i8) -> index {
  %c0 = puarith.constant 0 : index
  %c1 = puarith.constant 1 : index
  // CHECK: %[[LOOP:.*]] = puc.for %{{.*}} = %{{.*}} to %[[N]] step %{{.*}} iter_args(%[[ITER:.*]] = %[[ACC]]) -> (index) {
  %loop = pumem.for %i = %c0 to %n step %c1 iter_args(%iter = %acc) -> (index) {
    // CHECK: %[[NEXT:.*]] = puc.call @scalar_addi(%[[ITER]], %{{.*}}) : (index, index) -> index
    %next = puarith.addi %iter, %i : index
    // CHECK: puc.yield %[[NEXT]] : index
    pumem.yield %next : index
  }
  // CHECK: %[[SEL:.*]] = puc.if %[[COND]] -> (index) {
  %sel = pumem.if %cond -> (index) {
    pumem.yield %loop : index
  } else {
    pumem.yield %c0 : index
  }
  // CHECK: puc.return %[[SEL]] : index
  func.return %sel : index
}

// A call to a function in the module stays a direct puc.call.

func.func private @callee()

// CHECK-LABEL: puc.func @caller(
func.func @caller() {
  // CHECK: puc.call @callee() : () -> ()
  func.call @callee() : () -> ()
  func.return
}

// Unrealized casts are conversion-boundary glue: a chain that bridges back to
// the original type folds away before the conversion instead of leaking into
// puc.

// CHECK-LABEL: puc.func @reconciled(
// CHECK-SAME:    %[[ARG:.*]]: f32) -> f32 {
func.func @reconciled(%arg0: f32) -> f32 {
  %0 = builtin.unrealized_conversion_cast %arg0 : f32 to f64
  %1 = builtin.unrealized_conversion_cast %0 : f64 to f32
  // CHECK: puc.return %[[ARG]] : f32
  func.return %1 : f32
}
