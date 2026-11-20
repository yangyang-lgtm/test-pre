#ifndef PUC_DIALECT_TD
#define PUC_DIALECT_TD

include "mlir/IR/OpBase.td"
include "mlir/IR/BuiltinAttributeInterfaces.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/Interfaces/CallInterfaces.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/FunctionInterfaces.td"
include "mlir/Interfaces/LoopLikeInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

def Puc_Dialect : Dialect {
  let name = "puc";

  let cppNamespace = "::mlir::puc";

  let summary = "PrivateUse C emission dialect";

  let description = [{
    The puc dialect is the thin C emission layer: typed pointers, constants,
    pointer arithmetic, functions and calls, the kernel launcher, and the
    structured control flow (`puc.for` / `puc.if` / `puc.yield`) that the
    C emitter lowers to C loops and branches. All value computation lives in
    other dialects or in `puc.call`s into the runtime.
  }];

  let hasConstantMaterializer = 1;
}

class Puc_Op<string mnemonic, list<Trait> traits = []> :
    Op<Puc_Dialect, mnemonic, traits> {
}

def Puc_AnyPumemPtr : TypeConstraint<CPred<"::llvm::isa<::mlir::pumem::PtrType>($_self)">, "PUM pointer">;

def Puc_LaunchOp : Puc_Op<"launch"> {
  let summary = "Tail-call a kernel entry function";

  let description = [{
    Forwards the launcher arguments to the kernel entry function named by the
    kernel attribute. Emitted as a plain C call, so the launcher keeps the
    extern "C" symbol while the kernel keeps its own calling convention.

    Example: puc.launch @kernel(%arg0, %arg1) : !pumem.ptr<f32, 0>, index
  }];
  let arguments = (ins FlatSymbolRefAttr:$kernel, Variadic<AnyType>:$args);
  let assemblyFormat = "$kernel `(` $args `)` attr-dict `:` type($args)";
}

def Puc_ConstantOp : Puc_Op<"constant",
    [ConstantLike, Pure, AllTypesMatch<["value", "result"]>]> {
  let summary = "integer or floating point constant";

  let description = [{
    The `puc.constant` operation produces an SSA value equal to some integer
    or floating-point constant specified by an attribute.

    Example:
    ```mlir
    %c = puc.constant 42 : i32
    ```
  }];
  let arguments = (ins TypedAttrInterface:$value);
  let results = (outs AnyType:$result);
  let assemblyFormat = "attr-dict $value";

  let extraClassDeclaration = [{
    static bool isBuildableWith(::mlir::Attribute value, ::mlir::Type type);

    static ConstantOp materialize(::mlir::OpBuilder &builder,
                                  ::mlir::Attribute value, ::mlir::Type type,
                                  ::mlir::Location loc);
  }];

  let hasVerifier = 1;
  let hasFolder = 1;
}

def Puc_AddPtrOp : Puc_Op<"addptr",
    [Pure, AllTypesMatch<["ptr", "result"]>]> {
  let summary = "Offset a typed PUC pointer by a number of elements";

  let description = [{
    Returns `ptr` advanced by `offset` elements. The arithmetic is done in
    elements of the pointee type, so `offset` is scaled by the element size by
    the consumer of the op.

    Example: %p = puc.addptr %ptr, %offset : !pumem.ptr<f32, 128>
  }];
  let arguments = (ins Puc_AnyPumemPtr:$ptr, Index:$offset);
  let results = (outs Puc_AnyPumemPtr:$result);
  let assemblyFormat = "$ptr `,` $offset attr-dict `:` qualified(type($ptr))";
  let hasVerifier = 1;
  let hasFolder = 1;
}

def Puc_PointCastOp : Puc_Op<"point_cast", [Pure]> {
  let summary = "Reinterpret a PUC pointer as another pointer type";

  let description = [{
    Reinterprets a pointer as a pointer to a different element type. No
    address computation is performed.

    Example: %f = puc.point_cast %p : !pumem.ptr<void, 128> -> !pumem.ptr<f32, 128>
  }];
  let arguments = (ins Puc_AnyPumemPtr:$ptr);
  let results = (outs Puc_AnyPumemPtr:$result);
  let assemblyFormat = "$ptr attr-dict `:` qualified(type($ptr)) `->` qualified(type($result))";
  let hasFolder = 1;
}

def Puc_LoadOp : Puc_Op<"load", [MemoryEffects<[MemRead]>]> {
  let summary = "Load a scalar from a typed PUM pointer";

  let description = [{
    Loads the element pointed to by `ptr`. The pointer is typically the
    result of `puc.addptr`.

    Example: %v = puc.load %ptr : !pumem.ptr<f32, 128> -> f32
  }];
  let arguments = (ins Puc_AnyPumemPtr:$ptr);
  let results = (outs AnyType:$scalar);
  let assemblyFormat = "$ptr attr-dict `:` qualified(type($ptr)) `->` type($scalar)";
  let hasVerifier = 1;
}

def Puc_StoreOp : Puc_Op<"store", [MemoryEffects<[MemWrite]>]> {
  let summary = "Store a scalar through a typed PUM pointer";

  let description = [{
    Stores `scalar` at the location pointed to by `ptr`.

    Example: puc.store %v, %ptr : f32, !pumem.ptr<f32, 128>
  }];
  let arguments = (ins AnyType:$scalar, Puc_AnyPumemPtr:$ptr);
  let assemblyFormat = "$scalar `,` $ptr attr-dict `:` type($scalar) `,` qualified(type($ptr))";
  let hasVerifier = 1;
}

def Puc_ReturnOp : Puc_Op<"return",
    [Pure, ReturnLike, Terminator, ParentOneOf<["puc::FuncOp"]>]> {
  let summary = "Function return operation";

  let description = [{
    The `puc.return` operation represents a return operation within a
    function. The operation takes a variable number of operands and produces
    no results.
  }];
  let arguments = (ins Variadic<AnyType>:$operands);

  let builders = [OpBuilder<(ins), [{
    build($_builder, $_state, {});
  }]>];

  let assemblyFormat = "attr-dict ($operands^ `:` type($operands))?";
}

def Puc_FuncOp : Puc_Op<"func",
    [AutomaticAllocationScope, IsolatedFromAbove,
     CallableOpInterface, FunctionOpInterface, Symbol]> {
  let summary = "PUC function";

  let description = [{
    Example:
    ```mlir
    puc.func @kernel(%arg0: !pumem.ptr<f32, 128>) {
      ...
      puc.return
    }
    ```
  }];
  let arguments = (ins SymbolNameAttr:$sym_name,
                       TypeAttrOf<FunctionType>:$function_type,
                       OptionalAttr<DictArrayAttr>:$arg_attrs,
                       OptionalAttr<DictArrayAttr>:$res_attrs);
  let regions = (region AnyRegion:$body);

  let skipDefaultBuilders = 1;
  let builders = [OpBuilder<(ins
    "StringRef":$name, "FunctionType":$type,
    CArg<"ArrayRef<NamedAttribute>", "{}">:$attrs,
    CArg<"ArrayRef<DictionaryAttr>", "{}">:$argAttrs)
  >];

  let extraClassDeclaration = [{
    ::mlir::Region *getCallableRegion() { return isExternal() ? nullptr : &getBody(); }

    ::mlir::ArrayRef<::mlir::Type> getArgumentTypes() { return getFunctionType().getInputs(); }

    ::mlir::ArrayRef<::mlir::Type> getResultTypes() { return getFunctionType().getResults(); }
  }];

  let hasCustomAssemblyFormat = 1;
  let hasRegionVerifier = 1;
}

def Puc_CallOp : Puc_Op<"call"> {
  let summary = "call operation";

  let description = [{
    The `puc.call` operation represents a direct call to a function. The
    callee is either a `puc.func` in the same module or a runtime function
    provided by the backend runtime header. `templateArgs`, when present,
    carries the already formatted C++ template arguments of a runtime call.

    Example:
    ```mlir
    %r = puc.call @helper(%arg0) : (!pumem.ptr<f32, 0>) -> f32
    ```
  }];
  let arguments = (ins
    FlatSymbolRefAttr:$callee,
    OptionalAttr<ArrayAttr>:$templateArgs,
    Variadic<AnyType>:$operands
  );

  let results = (outs Variadic<AnyType>);

  let hasCustomAssemblyFormat = 1;
}

def Puc_YieldOp : Puc_Op<"yield",
    [Pure, ReturnLike, Terminator,
     ParentOneOf<["puc::ForOp", "puc::IfOp"]>]> {
  let summary = "loop yield and termination operation";

  let description = [{
    The `puc.yield` operation yields an SSA value from a puc region and
    terminates it. If the parent operation defines no values, the
    `puc.yield` may be left out in the custom syntax and the builders will
    insert one implicitly.
  }];

  let arguments = (ins Variadic<AnyType>:$results);

  let builders = [OpBuilder<(ins), [{ /* nothing to do */ }]>];

  let assemblyFormat = "attr-dict ($results^ `:` type($results))?";
}

def Puc_ForOp : Puc_Op<"for",
    [SingleBlock, SingleBlockImplicitTerminator<"puc::YieldOp">,
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
    The `puc.for` operation represents a loop whose first three operands are
    the lower bound, upper bound and step respectively, all `index` values.
    The operation has one region capturing the loop body, whose first
    argument is the induction variable; `puc.for` can also operate on
    loop-carried variables, which follow the control operands and get one
    region argument each.

    Example:
    ```mlir
    %sum = puc.for %i = %lb to %ub step %c1
           iter_args(%acc = %init) -> (f32) {
      puc.yield %next : f32
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

    /// Number of induction variables, always 1 for puc::ForOp.
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

def Puc_IfOp : Puc_Op<"if",
    [SingleBlock, SingleBlockImplicitTerminator<"puc::YieldOp">,
     RecursiveMemoryEffects, NoRegionArguments]> {
  let summary = "if-then-else operation";

  let description = [{
    The `puc.if` operation represents an if-then-else construct for
    conditional execution. The operation takes one `i8` condition and two
    regions. Results, when present, must be yielded by both regions, which
    requires the else region to exist.

    Example:
    ```mlir
    %r = puc.if %cond -> (f32) {
      puc.yield %a : f32
    } else {
      puc.yield %b : f32
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

def Puc_LocalOp : Puc_Op<"local", [MemoryEffects<[MemAlloc]>]> {
  let summary = "Declare a mutable scalar local";

  let description = [{
    Declares a mutable local variable, optionally initialized from `init`.
    The result denotes the variable: every use reads it, `puc.assign` writes
    it. This is the carrier used after `puc-localize-carried` turns
    loop/branch results into explicit assignments, mirroring the C++ locals
    emitted by puc-translate.
  }];
  let arguments = (ins Optional<AnyType>:$init);
  let results = (outs AnyType:$result);
  let hasCustomAssemblyFormat = 1;
}

def Puc_AddressOfOp : Puc_Op<"address_of", [Pure]> {
  let summary = "Take the address of a slot object";

  let description = [{
    A handle and a metadata slot are C++ objects the runtime mutates; they hold
    atomics, so they are never copied. Everything that hands one out passes its
    address, and this op is where that address is taken -- the difference from
    `puc.local` with an initializer, which is a copy.

    Example: %p = puc.address_of %h : !pumem.dma_handle -> !pumem.dma_handle_ptr
  }];
  let arguments = (ins AnyType:$value);
  let results = (outs AnyType:$pointer);
  let assemblyFormat = "$value attr-dict `:` qualified(type($value)) `->` qualified(type($pointer))";
}

def Puc_AssignOp : Puc_Op<"assign", [MemoryEffects<[MemWrite]>]> {
  let summary = "Assign a value to a mutable local";

  let description = [{
    Stores `value` into the local variable denoted by `var` (a `puc.local`
    result). Later reads of `var` observe the new value.
  }];
  let arguments = (ins AnyType:$var, AnyType:$value);
  let assemblyFormat = "$var `,` $value attr-dict `:` type($var) `,` type($value)";
}

#endif
