#ifndef PUMEM_DIALECT_TD
#define PUMEM_DIALECT_TD

include "mlir/IR/OpBase.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/LoopLikeInterface.td"
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

def PUMem_VoidType : PUMEM_Type<"Void", "void"> {
  let summary = "PUM void (untyped) type";

  let description = [{
    The element type of a `void` pointer, e.g. the result of
    `pumem.alloc` is `!pumem.ptr<void, 128>`.
  }];
}

def PUMem_PtrType : PUMEM_Type<"Ptr", "ptr"> {
  let summary = "A pointer into a PUM memory space";

  let description = [{
    A typed pointer to an element of the given memory space. Pointers are
    scalars: all memory that the backend can address, including buffers that
    used to be memrefs, is represented as a (pointer, offset) pair.

    A `void` pointer (untyped pointer) uses `!pumem.void` as the element
    type and is spelled with the `void` keyword, e.g.
    `!pumem.ptr<void, 128>`.

    Example: !pumem.ptr<f32, 128>
  }];
  let parameters = (ins "Type":$elemType, "int64_t":$space);

  let hasCustomAssemblyFormat = 1;

  code extraClassDefinition = [{
    ::mlir::Type $cppClass::parse(::mlir::AsmParser &parser) {
      if (parser.parseLess())
        return {};
      ::mlir::Type elemType;
      if (::mlir::succeeded(parser.parseOptionalKeyword("void"))) {
        elemType = VoidType::get(parser.getContext());
      } else if (parser.parseType(elemType)) {
        return {};
      }
      int64_t space = 0;
      if (parser.parseComma() || parser.parseInteger(space) ||
          parser.parseGreater())
        return {};
      return $cppClass::get(parser.getContext(), elemType, space);
    }

    void $cppClass::print(::mlir::AsmPrinter &printer) const {
      printer << "<";
      if (::mlir::isa<VoidType>(getElemType())) {
        printer << "void";
      } else {
        printer.printStrippedAttrOrType(getElemType());
      }
      printer << ", " << getSpace() << ">";
    }
  }];
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
    Allocates `bytes` bytes, aligned to `align` (a power of two), in the
    memory space encoded in the result pointer type and returns a `void`
    pointer to the first byte. Use `pumem.point_cast` to turn it into a typed
    pointer. Every allocation is owned by the op that performs it:
    `pumem.dealloc` must only be handed a pointer that comes straight from
    `pumem.alloc`.

    Example: %buf = pumem.alloc %bytes, %align : !pumem.ptr<void, 128>
  }];
  let arguments = (ins Index:$bytes, Index:$align);
  let results = (outs PUMem_Ptr:$ptr);
  let assemblyFormat = "$bytes `,` $align attr-dict `:` qualified(type($ptr))";
  let hasVerifier = 1;
}

def PUMem_PointCastOp : PUMEM_Op<"point_cast", [Pure]> {
  let summary = "Cast a PUM void pointer to a typed pointer";

  let description = [{
    Reinterprets a `void` pointer produced by `pumem.alloc` as a pointer to
    elements of the target element type in the target memory space.

    Example: %f = pumem.point_cast %p : !pumem.ptr<void, 128> -> !pumem.ptr<f32, 128>
  }];
  let arguments = (ins PUMem_Ptr:$ptr);
  let results = (outs PUMem_Ptr:$result);
  let assemblyFormat = "$ptr attr-dict `:` qualified(type($ptr)) `->` qualified(type($result))";
  let hasFolder = 1;
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

    The `dir` attribute records the copy direction relative to the memory
    spaces ("dram_to_sram" / "sram_to_dram" / "sram_to_sram" /
    "dram_to_dram"; space 0 is DRAM, any other space is SRAM). It is derived
    automatically from the operand spaces, so callers never pass it.
  }];
  let arguments = (ins Arg<PUMem_Ptr, "destination buffer", [MemWrite]>:$dst,
                       Arg<PUMem_Ptr, "source buffer", [MemRead]>:$src,
                       Variadic<Index>:$sizes,
                       Variadic<Index>:$dst_strides,
                       Variadic<Index>:$src_strides,
                       Index:$byte_size,
                       StrAttr:$dir);
  let builders = [
    OpBuilder<(ins "Value":$dst, "Value":$src, "ValueRange":$sizes,
                   "ValueRange":$dstStrides, "ValueRange":$srcStrides, "Value":$byteSize),
    [{
      auto dstDram = ::mlir::cast<::mlir::pumem::PtrType>(dst.getType()).getSpace() == ::mlir::triton::kDramSpace;
      auto srcDram = ::mlir::cast<::mlir::pumem::PtrType>(src.getType()).getSpace() == ::mlir::triton::kDramSpace;
      const char* dir =
        dstDram ? (srcDram ? "dram_to_dram" : "sram_to_dram") : (srcDram ? "dram_to_sram" : "sram_to_sram");
      build($_builder, $_state, dst, src, sizes, dstStrides, srcStrides, byteSize, dir);
    }]>
  ];
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
                   : !pumem.ptr<f32, 0>, !pumem.ptr<i32, 128>, !pumem.ptr<f32, 128>, !pumem.ptr<i8, 128>, f32
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

def PUMem_YieldOp : PUMEM_Op<"yield",
    [Pure, ReturnLike, Terminator,
     ParentOneOf<["pumem::ForOp", "pumem::IfOp"]>]> {
  let summary = "loop yield and termination operation";

  let description = [{
    The `pumem.yield` operation yields an SSA value from a pumem region and
    terminates it. If the parent operation defines no values, the
    `pumem.yield` may be left out in the custom syntax and the builders will
    insert one implicitly.
  }];

  let arguments = (ins Variadic<AnyType>:$results);

  let builders = [OpBuilder<(ins), [{ /* nothing to do */ }]>];

  let assemblyFormat = "attr-dict ($results^ `:` type($results))?";
}

//===----------------------------------------------------------------------===//
// ForOp (mirrors scf.for)
//===----------------------------------------------------------------------===//

def PUMem_ForOp : PUMEM_Op<"for",
    [SingleBlock, SingleBlockImplicitTerminator<"pumem::YieldOp">,
     RecursiveMemoryEffects,
     DeclareOpInterfaceMethods<LoopLikeOpInterface,
      ["getLoopRegions", "getLoopInductionVars", "getLoopLowerBounds",
       "getLoopUpperBounds", "getLoopSteps", "setLoopLowerBounds",
       "setLoopUpperBounds", "setLoopSteps", "getInitsMutable",
       "getYieldedValuesMutable", "getLoopResults",
       "getStaticTripCount", "isValidInductionVarType",
       "replaceWithAdditionalYields"]>]> {
  let summary = "for operation";

  let description = [{
    The `pumem.for` operation represents a loop whose first three operands
    are the lower bound, upper bound and step respectively. The operation has
    one region capturing the loop body. The induction variable is represented
    as an argument of this region.

    Lower bound, upper bound, and step are `index` values (like everywhere
    else in the pumem dialect: alloc/addptr/copy all use `index`). The lower
    and upper bounds specify a half-open range, including the lower bound but
    excluding the upper bound.

    `pumem.for` can also operate on loop-carried variables and returns the
    final values after loop termination. The initial values of the variables
    are passed as additional SSA operands following the three loop control
    operands. The region has an argument for the induction variable, followed
    by one argument for each loop-carried variable.

    Example:
    ```mlir
    %sum = pumem.for %i = %lb to %ub step %c1
           iter_args(%acc = %init) -> (f32) {
      pumem.yield %next : f32
    }
    ```
  }];

  let arguments = (ins Index:$lowerBound,
                       Index:$upperBound,
                       Index:$step,
                       Variadic<AnyType>:$initArgs);
  let results = (outs Variadic<AnyType>:$results);
  let regions = (region SizedRegion<1>:$region);

  let skipDefaultBuilders = 1;
  let builders = [OpBuilder<(ins "Value":$lowerBound, "Value":$upperBound,
      "Value":$step, CArg<"ValueRange", "{}">:$initArgs,
      CArg<"function_ref<void(OpBuilder &, Location, Value, ValueRange)>",
           "nullptr">)>];

  let extraClassDeclaration = [{
    using BodyBuilderFn =
        function_ref<void(OpBuilder &, Location, Value, ValueRange)>;

    Value getInductionVar() { return getBody()->getArgument(0); }

    /// Number of induction variables, always 1 for pumem::ForOp.
    unsigned getNumInductionVars() { return 1; }

    /// Number of region arguments for loop-carried values.
    unsigned getNumRegionIterArgs() {
      return getBody()->getNumArguments() - getNumInductionVars();
    }

    /// Return the `index`-th region iteration argument.
    BlockArgument getRegionIterArg(unsigned index) {
      if (!(index < getNumRegionIterArgs()))
        llvm::report_fatal_error("expected an index less than the number of region iter args");
      return getBody()->getArguments().drop_front(getNumInductionVars())[index];
    }

    Block::BlockArgListType getRegionIterArgs() {
      return getBody()->getArguments().drop_front(getNumInductionVars());
    }

    /// Number of operands controlling the loop: lb, ub, step.
    unsigned getNumControlOperands() { return 3; }

    Region &getBodyRegion() { return getOperation()->getRegion(0); }
    Block *getBody() { return &getBodyRegion().front(); }

    void setLowerBound(Value bound) { setOperand(0, bound); }
    void setUpperBound(Value bound) { setOperand(1, bound); }
    void setStep(Value step) { setOperand(2, step); }
  }];

  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1;
  let hasRegionVerifier = 1;
}

// IfOp (mirrors scf.if)
//===----------------------------------------------------------------------===//

def PUMem_IfOp : PUMEM_Op<"if",
    [SingleBlock, SingleBlockImplicitTerminator<"pumem::YieldOp">,
     RecursiveMemoryEffects, NoRegionArguments]> {
  let summary = "if-then-else operation";

  let description = [{
    The `pumem.if` operation represents an if-then-else construct for
    conditional execution. The operation takes one `i8` condition (matching
    the `i8` boolean produced by integer/pointer comparisons in the pumem
    world) and two regions. Results, when present, must be yielded by both
    regions, which requires the else region to exist.

    Example:
    ```mlir
    %r = pumem.if %cond -> (f32) {
      pumem.yield %a : f32
    } else {
      pumem.yield %b : f32
    }
    ```
  }];

  let arguments = (ins I8:$condition);
  let results = (outs Variadic<AnyType>:$results);
  let regions = (region SizedRegion<1>:$thenRegion,
                        MaxSizedRegion<1>:$elseRegion);

  let skipDefaultBuilders = 1;
  let builders = [
    OpBuilder<(ins "TypeRange":$resultTypes, "Value":$cond)>,
    OpBuilder<(ins "TypeRange":$resultTypes, "Value":$cond,
                   "bool":$addThenBlock, "bool":$addElseBlock)>,
    OpBuilder<(ins "TypeRange":$resultTypes, "Value":$cond,
                   "bool":$withElseRegion)>,
    OpBuilder<(ins "Value":$cond, CArg<"bool", "true">:$withElseRegion)>
  ];

  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1;
}


#endif
