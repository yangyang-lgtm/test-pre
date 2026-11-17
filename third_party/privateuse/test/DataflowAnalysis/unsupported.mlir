// RUN: not privateuse-opt %s -privateuse-analyze-dataflow -split-input-file -o /dev/null 2>&1 | FileCheck %s

module attributes {privateuse.grid = array<i64: 1, 1, 1>} {
  tt.func public @bad_space() {
    %c4 = arith.constant 4 : index
    %c8 = arith.constant 8 : index
    %pid = pumem.get_physical_id : index
    scf.for %p = %pid to %c4 step %c8 {
      %bad = memref.alloc() : memref<16xf32, 3>
    }
    tt.return
  }
}

// CHECK: func @bad_space deferred=unsupported_memory_space

// -----

module attributes {privateuse.grid = array<i64: 1, 1, 1>} {
  func.func private @sink(memref<16xf32, 128>)

  tt.func public @escape() {
    %c4 = arith.constant 4 : index
    %c8 = arith.constant 8 : index
    %pid = pumem.get_physical_id : index
    scf.for %p = %pid to %c4 step %c8 {
      %alloc = memref.alloc() : memref<16xf32, 128>
      func.call @sink(%alloc) : (memref<16xf32, 128>) -> ()
    }
    tt.return
  }
}

// CHECK: func @escape deferred=memref_escape

// -----

#map = affine_map<(d0) -> (d0)>
module attributes {privateuse.grid = array<i64: 1, 1, 1>} {
  memref.global "private" @g : memref<16xf32>

  tt.func public @unresolved() {
    %c4 = arith.constant 4 : index
    %c8 = arith.constant 8 : index
    %pid = pumem.get_physical_id : index
    scf.for %p = %pid to %c4 step %c8 {
      %g = memref.get_global @g : memref<16xf32>
      linalg.generic {indexing_maps = [#map], iterator_types = ["parallel"]} outs(%g : memref<16xf32>) {
      ^bb0(%out: f32):
        linalg.yield %out : f32
      }
    }
    tt.return
  }
}

// CHECK: func @unresolved deferred=unresolved_memref

// -----

module attributes {privateuse.grid = array<i64: 1, 1, 1>} {
  memref.global "private" @bad : memref<16xf32, 3>

  tt.func public @bad_space_value() {
    %c4 = arith.constant 4 : index
    %c8 = arith.constant 8 : index
    %pid = pumem.get_physical_id : index
    scf.for %p = %pid to %c4 step %c8 {
      %b = memref.get_global @bad : memref<16xf32, 3>
    }
    tt.return
  }
}

// CHECK: func @bad_space_value deferred=unsupported_memory_space

// -----

module attributes {privateuse.grid = array<i64: 1, 1, 1>} {
  tt.func public @load_memref(%buf: memref<8xmemref<?xf32>>) {
    %c0 = arith.constant 0 : index
    %c4 = arith.constant 4 : index
    %c8 = arith.constant 8 : index
    %pid = pumem.get_physical_id : index
    scf.for %p = %pid to %c4 step %c8 {
      %m = memref.load %buf[%c0] : memref<8xmemref<?xf32>>
    }
    tt.return
  }
}

// CHECK: func @load_memref deferred=memref_escape

// -----

module {
  tt.func public @no_grid() {
    tt.return
  }
}

// CHECK: error: missing privateuse.grid module attribute