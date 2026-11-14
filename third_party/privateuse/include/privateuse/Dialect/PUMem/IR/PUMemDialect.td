#ifndef PUMEM_DIALECT_TD
#define PUMEM_DIALECT_TD

include "mlir/IR/OpBase.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

def PUMem_Dialect : Dialect {
  let name = "pumem";

  let cppNamespace = "::mlir::pumem";

  let summary = "PUM memory dialect";

  let useDefaultTypePrinterParser = 1;
}

class PUMEM_Op<string mnemonic, list<Trait> traits = []> :
    Op<PUMem_Dialect, mnemonic, traits> {
}

class PUMEM_Type<string name, string typeMnemonic, list<Trait> traits = []> :
    TypeDef<PUMem_Dialect, name, traits> {
  let mnemonic = typeMnemonic;
}

def PUMem_PtrType : PUMEM_Type<"Ptr", "ptr"> {
  let summary = "A pointer into a PUM memory space";

  let description = [{
    A typed pointer to an element of the given memory space. Pointers are
    scalars: all memory that the backend can address, including buffers that
    used to be memrefs, is represented as a (pointer, offset) pair.

    Example: !pumem.ptr<f32, 128>
  }];
  let parameters = (ins "Type":$elemType, "int64_t":$space);
  let assemblyFormat = "`<` $elemType `,` $space `>`";
}

def PUMem_Ptr : Type<CPred<"::llvm::isa<::mlir::pumem::PtrType>($_self)">,
                     "a pumem pointer", "::mlir::pumem::PtrType">;

def PUMem_GetPhysicalIdOp : PUMEM_Op<"get_physical_id", [Pure]> {
  let summary = "Get the id of the current physical thread";

  let description = [{
    Returns the linear physical thread id set by the launch loop, in
    [0, physical_threads_num). The program loop starts from it and strides by
    the number of physical threads to cover the whole grid.

    Example: %tid = pumem.get_physical_id : index
  }];
  let results = (outs Index:$id);
  let assemblyFormat = "attr-dict `:` type($id)";
}

def PUMem_AllocOp : PUMEM_Op<"alloc"> {
  let summary = "Allocate a runtime buffer";

  let description = [{
    Allocates `bytes` bytes in the memory space encoded in the result pointer
    type and returns a pointer to the first byte. `alignment` is a power of
    two. Every allocation is owned by the op that performs it: `pumem.dealloc`
    must only be handed a pointer that comes straight from `pumem.alloc`.

    Example: %buf = pumem.alloc %bytes {alignment = 64 : i64} : !pumem.ptr<f32, 128>
  }];
  let arguments = (ins Index:$bytes, I64Attr:$alignment);
  let results = (outs PUMem_Ptr:$ptr);
  let assemblyFormat = "$bytes attr-dict `:` qualified(type($ptr))";
  let builders = [OpBuilder<(ins "Type":$resultType, "Value":$bytes, "int64_t":$alignment), [{
    $_state.addTypes(resultType);
    $_state.addOperands(bytes);
    $_state.addAttribute("alignment", $_builder.getI64IntegerAttr(alignment));
  }]>];
  let hasVerifier = 1;
}

def PUMem_DeallocOp : PUMEM_Op<"dealloc"> {
  let summary = "Release a buffer allocated by pumem.alloc";

  let description = [{
    Releases the buffer rooted at `ptr`, which must be the result of a
    `pumem.alloc` (possibly carried through block arguments untouched, but not
    through `pumem.addptr`).

    Example: pumem.dealloc %buf : !pumem.ptr<f32, 128>
  }];
  let arguments = (ins PUMem_Ptr:$ptr);
  let assemblyFormat = "$ptr attr-dict `:` qualified(type($ptr))";
}

def PUMem_AddPtrOp : PUMEM_Op<"addptr", [Pure, AllTypesMatch<["ptr", "result"]>]> {
  let summary = "Offset a pointer by a number of elements";

  let description = [{
    Returns a pointer to the element `offset` elements after `ptr`. The
    arithmetic is done in elements of the pointee type, so `offset` is scaled
    by the element size by the consumer of the op.

    Example: %next = pumem.addptr %buf, %i : !pumem.ptr<f32, 128>
  }];
  let arguments = (ins PUMem_Ptr:$ptr, Index:$offset);
  let results = (outs PUMem_Ptr:$result);
  let assemblyFormat = "$ptr `,` $offset attr-dict `:` qualified(type($ptr))";
  let hasFolder = 1;
}

def PUMem_LoadOp : PUMEM_Op<"load"> {
  let summary = "Load the element a pointer points to";

  let description = [{
    Reads one element of the pointer's pointee type from memory.

    Example: %v = pumem.load %addr : !pumem.ptr<f32, 128> -> f32
  }];
  let arguments = (ins Arg<PUMem_Ptr, "pointer to load from", [MemRead]>:$ptr);
  let results = (outs AnyType:$value);
  let assemblyFormat = "$ptr attr-dict `:` qualified(type($ptr)) `->` type($value)";
  let hasVerifier = 1;
}

def PUMem_StoreOp : PUMEM_Op<"store"> {
  let summary = "Store an element through a pointer";

  let description = [{
    Writes one element of the pointer's pointee type to memory.

    Example: pumem.store %v, %addr : f32, !pumem.ptr<f32, 128>
  }];
  let arguments = (ins AnyType:$value,
                       Arg<PUMem_Ptr, "pointer to store to", [MemWrite]>:$ptr);
  let assemblyFormat = "$value `,` $ptr attr-dict `:` type($value) `,` qualified(type($ptr))";
  let hasVerifier = 1;
}

def PUMem_CopyOp : PUMEM_Op<"copy", [AttrSizedOperandSegments]> {
  let summary = "Copy a strided block between two buffers";

  let description = [{
    Copies `sizes` elements (one size per dimension) from the strided block at
    `src` to the strided block at `dst`. `dst_strides` and `src_strides` are
    per-dimension element strides. `byte_size` is the size of one element in
    bytes.

    Example:
      pumem.copy %dst, %src, sizes = (%n, %m), dst_strides = (%m, %c1),
                 src_strides = (%m, %c1), byte_size %c4
                 : !pumem.ptr<f32, 128>, !pumem.ptr<f32, 0>
  }];
  let arguments = (ins Arg<PUMem_Ptr, "destination buffer", [MemWrite]>:$dst,
                       Arg<PUMem_Ptr, "source buffer", [MemRead]>:$src,
                       Variadic<Index>:$sizes,
                       Variadic<Index>:$dst_strides,
                       Variadic<Index>:$src_strides,
                       Index:$byte_size);
  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1;
}

def PUMem_TileGatherOp : PUMEM_Op<"tile_gather", [AttrSizedOperandSegments]> {
  let summary = "Gather elements from a flat buffer into a tile";

  let description = [{
    Writes one element into the destination tile for every element of the
    offset buffer, both shaped alike:

      destination[i] = mask[i] ? source[indices[i]] : other

    Without a mask every element is gathered, as if the mask were all true.
    Masked-off lanes never read source and take `other`, a scalar of the source
    element type; without `other` they are left untouched. `source` is a flat
    runtime buffer addressed in elements; `indices` is a runtime buffer of
    element offsets. The op has no results and mutates the destination tile in
    place, like memref.copy.

    This is the buffer-level form: `-privateuse-memref-to-pumem` replaces it
    with `pumem.gather`, whose operands are the scalar tuples of these buffers.
  }];
  let arguments = (ins Arg<AnyMemRef, "flat buffer read at the given element offsets", [MemRead]>:$source,
                       Arg<AnyMemRef, "per-element offsets", [MemRead]>:$indices,
                       Arg<AnyMemRef, "tile written with the gathered values", [MemWrite]>:$destination,
                       Optional<AnyMemRef>:$mask,
                       Optional<AnyType>:$other);
  let assemblyFormat = "$source `,` $indices `,` $destination (`,` $mask^)? (`,` $other^)? attr-dict `:` type($source) `,` type($indices) `,` type($destination) (`,` type($mask)^)? (`,` type($other)^)?";
  let hasVerifier = 1;
}

def PUMem_TileScatterOp : PUMEM_Op<"tile_scatter"> {
  let summary = "Scatter elements from a tile into a flat buffer";

  let description = [{
    Reads one element from the source tile for every element of the offset
    buffer, both shaped alike, and writes it into the flat destination buffer:

      destination[indices[i]] = source[i]

    With a mask, masked-off lanes are neither read nor written. `destination` is
    a flat runtime buffer addressed in elements; `indices` is a runtime buffer of
    element offsets. The op has no results and mutates the destination buffer in
    place, like memref.copy.

    This is the buffer-level form: `-privateuse-memref-to-pumem` replaces it
    with `pumem.scatter`, whose operands are the scalar tuples of these buffers.
  }];
  let arguments = (ins Arg<AnyMemRef, "tile read in order", [MemRead]>:$source,
                       Arg<AnyMemRef, "per-element offsets", [MemRead]>:$indices,
                       Arg<AnyMemRef, "flat buffer written at the given element offsets", [MemWrite]>:$destination,
                       Optional<AnyMemRef>:$mask);
  let assemblyFormat = "$source `,` $indices `,` $destination (`,` $mask^)? attr-dict `:` type($source) `,` type($indices) `,` type($destination) (`,` type($mask)^)?";
  let hasVerifier = 1;
}

def PUMem_GatherOp : PUMEM_Op<"gather", [AttrSizedOperandSegments]> {
  let summary = "Gather elements from a flat buffer into a tile, over scalar tuples";

  let description = [{
    Target-level form of `pumem.tile_gather`. Every buffer is decomposed into
    its scalar tuple and its offset is folded into the pointer with
    `pumem.addptr`; `source` is a flat buffer, so it contributes only its
    pointer and stride, while the tile buffers share `sizes` (the
    indices/destination shape) and carry one stride per dimension:

      destination[i] = mask[i] ? source[indices[i]] : other

    With a mask, `mask` and `mask_strides` describe the mask buffer. `other`, a
    scalar of the source element type, requires a mask; without it, masked-off
    lanes are left untouched.

    Example:
      pumem.gather %src, %src_stride, %idx, %dst,
                   sizes = (%n, %m), indices_strides = (%m, %c1),
                   destination_strides = (%m, %c1), mask = %mask,
                   mask_strides = (%m, %c1), other %fill
                   : !pumem.ptr<f32, 0>, !pumem.ptr<i32, 128>, !pumem.ptr<f32, 128>, !pumem.ptr<i1, 128>, f32
  }];
  let arguments = (ins Arg<PUMem_Ptr, "flat buffer read at the given element offsets", [MemRead]>:$source,
                       Index:$source_stride,
                       Arg<PUMem_Ptr, "per-element offsets", [MemRead]>:$indices,
                       Arg<PUMem_Ptr, "tile written with the gathered values", [MemWrite]>:$destination,
                       Variadic<Index>:$sizes,
                       Variadic<Index>:$indices_strides,
                       Variadic<Index>:$destination_strides,
                       Optional<PUMem_Ptr>:$mask,
                       Variadic<Index>:$mask_strides,
                       Optional<AnyType>:$other);
  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1;
}

def PUMem_ScatterOp : PUMEM_Op<"scatter", [AttrSizedOperandSegments]> {
  let summary = "Scatter elements from a tile into a flat buffer, over scalar tuples";

  let description = [{
    Target-level form of `pumem.tile_scatter`. Every buffer is decomposed into
    its scalar tuple and its offset is folded into the pointer with
    `pumem.addptr`; `destination` is a flat buffer, so it contributes only its
    pointer and stride, while the tile buffers share `sizes` (the
    source/indices shape) and carry one stride per dimension:

      destination[indices[i]] = source[i]

    With a mask, `mask` and `mask_strides` describe the mask buffer;
    masked-off lanes are neither read nor written.

    Example:
      pumem.scatter %src, %idx, %dst, %dst_stride,
                    sizes = (%n, %m), source_strides = (%m, %c1),
                    indices_strides = (%m, %c1)
                    : !pumem.ptr<f32, 128>, !pumem.ptr<i32, 128>, !pumem.ptr<f32, 0>
  }];
  let arguments = (ins Arg<PUMem_Ptr, "tile read in order", [MemRead]>:$source,
                       Arg<PUMem_Ptr, "per-element offsets", [MemRead]>:$indices,
                       Arg<PUMem_Ptr, "flat buffer written at the given element offsets", [MemWrite]>:$destination,
                       Index:$destination_stride,
                       Variadic<Index>:$sizes,
                       Variadic<Index>:$source_strides,
                       Variadic<Index>:$indices_strides,
                       Optional<PUMem_Ptr>:$mask,
                       Variadic<Index>:$mask_strides);
  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1;
}

def PUMem_RuntimeCallOp : PUMEM_Op<"runtime_call", [MemoryEffects<[MemRead, MemWrite]>]> {
  let summary = "Direct call into the C++ runtime";

  let description = [{
    Calls a runtime function by name, optionally binding explicit template
    arguments. The call is printed verbatim by the C++ emitter, so the callee
    (plus template arguments, if any) must exist in the runtime header. Each
    template argument is either a type or a compile-time value.

    Example:
      pumem.runtime_call @runtime_matmul<[f32, 4 : index, 16 : index, 8 : index]>(%a, %b, %c)
        : (!pumem.ptr<f32, 128>, !pumem.ptr<f32, 128>, !pumem.ptr<f32, 128>) -> ()
  }];
  let arguments = (ins FlatSymbolRefAttr:$callee,
                       OptionalAttr<ArrayAttr>:$templateArgs,
                       Variadic<AnyType>:$operands);
  let results = (outs Variadic<AnyType>:$results);
  let assemblyFormat = "$callee (`<` $templateArgs^ `>`)? `(` $operands `)` attr-dict `:` functional-type($operands, results)";
}

#endif
