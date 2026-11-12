#ifndef PRIVATEUSE_CONVERSION_PASSES_TD
#define PRIVATEUSE_CONVERSION_PASSES_TD

include "mlir/Pass/PassBase.td"

def AnalyzeAccess : Pass<"privateuse-analyze-access", "mlir::ModuleOp"> {
  let summary = "Report canonical memory access views and masks";
  let description = [{
    Runs the access analysis over every tt.load / tt.store in the module and
    prints one report line per operation plus a module summary:

    * the canonical strided view (base pointer, scalar offset, per-dimension
      start / stride / size);
    * the mask classification (none, all-true, all-false, scalar, box);
    * the deferred reason when the access cannot be lowered yet.

    The pass never modifies the IR; it is a diagnostic and regression tool for
    the memory lowering pipeline.
  }];
  let constructor = "mlir::triton::createAnalyzeAccessPass()";
}

def LowerStructuredAccess : Pass<"privateuse-lower-structured-access", "mlir::ModuleOp"> {
  let summary = "Lower structured tt.load / tt.store to memref views and copies";
  let description = [{
    Rewrites tensor accesses whose canonical view is available and whose mask is
    absent or statically all-true:

    * each access becomes a memref.reinterpret_cast view followed by memref.copy
      through a local tile buffer; the base pointer is bridged with an
      unrealized_conversion_cast so that the address chain stays untouched;
    * tt.splat becomes tensor.splat and dead value / mask chains are removed;
    * functions with at least one lowered access are tagged so that
      -privateuse-ptr-to-memref can rewrite their signatures afterwards.

    Address chains and loop-carried pointers are intentionally preserved:
    accesses that are not lowered here (gather / scatter candidates) can still
    be rewritten by a later pass.

    Coverage is limited: accesses with a deferred canonical view (descriptor /
    atomic / gather / scatter, negative strides, data-dependent offsets), box or
    scalar masks, stores whose view is not injective over the shape box
    (overlapping addresses, including zero-stride stores), functions with tensor
    arguments or results, and functions whose access base is not an entry block
    argument are left untouched. Modules containing tt.call are skipped.
  }];
  let constructor = "mlir::triton::createLowerStructuredAccessPass()";
}

def LowerUnstructuredAccess : Pass<"privateuse-lower-unstructured-access", "mlir::ModuleOp"> {
  let summary = "Lower non-structured tt.load / tt.store to pumem gather / scatter";
  let description = [{
    Rewrites tensor accesses whose address chain decomposes into an entry-block
    pointer argument plus a per-element index tensor:

    * the base pointer is bridged with an unrealized_conversion_cast so that the
      address chain stays untouched;
    * the accumulated element offsets are materialized into an offset buffer
      that keeps the integer element type of the offsets;
    * loads become pumem.gather through a local tile buffer, stores become
      pumem.scatter from a local tile buffer;
    * functions with at least one lowered access are tagged so that
      -privateuse-ptr-to-memref can rewrite their signatures afterwards.

    Coverage is limited: masked, volatile / ignore_cta accesses, atomics,
    descriptors, accesses whose base is not an entry block pointer argument
    (e.g. loop-carried pointers), functions with tensor signatures, and
    modules containing tt.call are left untouched.
  }];
  let constructor = "mlir::triton::createLowerUnstructuredAccessPass()";
}

def PtrToMemref : Pass<"privateuse-ptr-to-memref", "mlir::ModuleOp"> {
  let summary = "Rewrite pointer signatures and address chains of lowered functions to memrefs";
  let description = [{
    Converts the functions tagged by -privateuse-lower-structured-access:

    * pointer arguments become ranked memrefs;
    * in functions where every pointer use was lowered, address chains
      (tt.splat / tt.addptr / tt.broadcast / tt.expand_dims and scf region
      carried pointers) become opaque memref tokens with the same type, so the
      conversion is a pure value substitution;
    * in functions that still contain pointer-consuming operations, address
      chains and scf regions are left untouched (tokenizing a chain that feeds
      a remaining access would drop its offset); only the signature is
      converted and bridged with unrealized casts;
    * unrealized casts left by the access lowering are reconciled, and argument
      attributes on converted memref parameters are cleared.

    Untagged functions are left untouched.
  }];
  let constructor = "mlir::triton::createPtrToMemrefPass()";
}

#endif
