// RUN: privateuse-opt %s -privateuse-memref-to-pumem -verify-diagnostics -split-input-file

// The preconditions reject the cases the tuple representation cannot express.
// Each chunk runs on its own because the pass stops at the first error.

// expected-error @+1 {{unranked memref types are not supported}}
tt.func @unranked(%A: memref<*xf32>) {
  tt.return
}

// -----

// expected-error @+1 {{memref memory space must be an integer attribute}}
tt.func @string_space(%A: memref<16xf32, "gpu">) {
  tt.return
}

// -----

// Allocations must have a static, non-empty shape.

tt.func @dynamic_alloc(%n: index) {
  // expected-error @+2 {{cannot lower an allocation with a dynamic shape}}
  // expected-error @+1 {{failed to legalize operation 'memref.alloc'}}
  %A = memref.alloc(%n) : memref<?xf32, 128>
  memref.dealloc %A : memref<?xf32, 128>
  tt.return
}

// -----

tt.func @empty_alloc() {
  // expected-error @+2 {{cannot lower an allocation with a zero-sized dimension}}
  // expected-error @+1 {{failed to legalize operation 'memref.alloc'}}
  %A = memref.alloc() : memref<0xf32, 128>
  memref.dealloc %A : memref<0xf32, 128>
  tt.return
}

// -----

// Ownership cannot be proven through block arguments, so freeing a memref that
// comes in as a parameter is a diagnostic, not a silent free.

tt.func @free_arg(%A: memref<16xf32, 128>) {
  // expected-error @+2 {{cannot free a memref that does not come straight out of a pumem.alloc}}
  // expected-error @+1 {{failed to legalize operation 'memref.dealloc'}}
  memref.dealloc %A : memref<16xf32, 128>
  tt.return
}
