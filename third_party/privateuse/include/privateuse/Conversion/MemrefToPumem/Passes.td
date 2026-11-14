#ifndef PRIVATEUSE_CONVERSION_MEMREFTOPUMEM_PASSES_TD
#define PRIVATEUSE_CONVERSION_MEMREFTOPUMEM_PASSES_TD

include "mlir/Pass/PassBase.td"

def MemrefToPumem : Pass<"privateuse-memref-to-pumem", "mlir::ModuleOp"> {
  let summary = "Lower every memref to a scalar (pointer, offset, sizes, strides) tuple";
  let description = [{
    Removes memrefs from the IR by replacing each memref value with a flat
    tuple of scalars, in this order:

      (ptr, offset, size_0 .. size_{r-1}, stride_0 .. stride_{r-1})

    The pass performs a single 1->N dialect conversion: there is no
    intermediate descriptor IR type, no structs are ever materialized, and no
    unrealized_conversion_casts are left behind. Function signatures use the
    same tuple ABI, and scf.for / scf.if carry the flattened values.

    The conversion is illegal-by-default: every operation whose operands,
    results, block arguments or function signature mention a memref must have
    a lowering, otherwise the pass fails and names the operation. Shapes and
    layouts must already be normalized to strided form (run
    -expand-strided-metadata and -memref-expand first); residual
    memref.subview / memref.expand_shape / memref.collapse_shape operations
    are rejected.

    Supported lowerings: memref.alloc, memref.dealloc, memref.load,
    memref.store, memref.copy, memref.cast, memref.dim, memref.rank,
    memref.extract_strided_metadata, memref.reinterpret_cast, scf.for,
    scf.yield, scf.if, tt.func, tt.call, tt.return, pumem.tile_gather and
    pumem.tile_scatter. tt.func / tt.call / tt.return are replaced by their
    func dialect counterparts. pumem.tile_gather and pumem.tile_scatter are
    replaced by their flat counterparts pumem.gather and pumem.scatter, with
    every memref operand unpacked into its (pointer, offset, sizes, strides)
    tuple; the offset is folded into the pointer with pumem.addptr, so the
    flat ops only take pointers. pumem.runtime_call keeps its callee and
    template arguments and lowers every memref operand to the adjusted
    pointer. pumem.alloc uses SRAM, i.e. the memory space
    carried by the memref,
    and the alignment of the memref allocation when present, otherwise
    kDefaultAlignment; memrefs without a memory space live in DRAM
    (kDramSpace). memref.dealloc is only lowered when the pointer can be
    traced back to a pumem.alloc and the offset is statically zero; anything
    else is a diagnostic, not a silent free.
  }];
  let dependentDialects = ["::mlir::pumem::PUMemDialect", "::mlir::func::FuncDialect"];
  let constructor = "mlir::triton::createMemrefToPumemPass()";
}

#endif
