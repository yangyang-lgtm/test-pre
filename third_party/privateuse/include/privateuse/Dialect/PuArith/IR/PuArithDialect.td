include "PuArithInterfaces.td"
include "mlir/Interfaces/CastInterfaces.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/VectorInterfaces.td"
include "mlir/IR/BuiltinAttributeInterfaces.td"
include "mlir/IR/OpAsmInterface.td"

class PuArith_Op<string mnemonic, list<Trait> traits = []> :
    Op<PuArith_Dialect, mnemonic,
       traits #
       [DeclareOpInterfaceMethods<VectorUnrollOpInterface>, NoMemoryEffect] #
       ElementwiseMappable.traits>;


def PuArith_SignlessIntegerOrIndexLike : TypeOrValueSemanticsContainer<
    AnyNonZeroBitwidthSignlessIntegerOrIndex,
    "signless-non-zero-bitwidth-integer-like">;

class PuArith_ArithOp<string mnemonic, list<Trait> traits = []> :
    PuArith_Op<mnemonic, traits # [SameOperandsAndResultType]>;

class PuArith_UnaryOp<string mnemonic, list<Trait> traits = []> :
    PuArith_ArithOp<mnemonic, traits # [Pure]> {
  let assemblyFormat = "$operand attr-dict `:` type($result)";
}

class PuArith_BinaryOp<string mnemonic, list<Trait> traits = []> :
    PuArith_ArithOp<mnemonic, traits> {
  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` type($result)";
}

class PuArith_IntBinaryOp<string mnemonic, list<Trait> traits = []> :
    PuArith_BinaryOp<mnemonic, traits #
      []>,
    Arguments<(ins PuArith_SignlessIntegerOrIndexLike:$lhs, PuArith_SignlessIntegerOrIndexLike:$rhs)>,
    Results<(outs PuArith_SignlessIntegerOrIndexLike:$result)>;

class PuArith_TotalIntBinaryOp<string mnemonic, list<Trait> traits = []> :
    PuArith_IntBinaryOp<mnemonic, traits # [Pure]>;

class PuArith_FloatUnaryOp<string mnemonic, list<Trait> traits = []> :
    PuArith_UnaryOp<mnemonic,
      !listconcat([DeclareOpInterfaceMethods<PuArithFastMathInterface>],
                  traits)>,
    Arguments<(ins FloatLike:$operand,
      DefaultValuedAttr<
        PuArith_FastMathAttr, "::mlir::puarith::FastMathFlags::none">:$fastmath)>,
    Results<(outs FloatLike:$result)> {
  let assemblyFormat = [{ $operand (`fastmath` `` $fastmath^)?
                          attr-dict `:` type($result) }];
}

class PuArith_FloatBinaryOp<string mnemonic, list<Trait> traits = []> :
    PuArith_BinaryOp<mnemonic,
      !listconcat([Pure, DeclareOpInterfaceMethods<PuArithFastMathInterface>],
                  traits)> {
  let arguments = (ins FloatLike:$lhs, FloatLike:$rhs,
      DefaultValuedAttr<
        PuArith_FastMathAttr, "::mlir::puarith::FastMathFlags::none">:$fastmath);
  let results = (outs FloatLike:$result);
  let assemblyFormat = [{ $lhs `,` $rhs (`fastmath` `` $fastmath^)?
                          attr-dict `:` type($result) }];
}

class PuArith_FloatBinaryOpWithRoundingMode<string mnemonic,
                                          list<Trait> traits = []> :
    PuArith_FloatBinaryOp<mnemonic,
      !listconcat([DeclareOpInterfaceMethods<PuArithRoundingModeInterface>],
                  traits)> {
  let arguments = (ins FloatLike:$lhs, FloatLike:$rhs,
      DefaultValuedAttr<
        PuArith_FastMathAttr, "::mlir::puarith::FastMathFlags::none">:$fastmath,
      OptionalAttr<PuArith_RoundingModeAttr>:$roundingmode);
  let builders = [
    OpBuilder<(ins "Value":$lhs, "Value":$rhs,
      CArg<"::mlir::puarith::FastMathFlags",
           "::mlir::puarith::FastMathFlags::none">:$fastmath), [{
      build($_builder, $_state, lhs, rhs, fastmath,
            ::mlir::puarith::RoundingModeAttr{});
    }]>,
    OpBuilder<(ins "Value":$lhs, "Value":$rhs,
      "::mlir::puarith::FastMathFlagsAttr":$fastmath), [{
      build($_builder, $_state, lhs, rhs, fastmath,
            ::mlir::puarith::RoundingModeAttr{});
    }]>,
    OpBuilder<(ins "Type":$type, "Value":$lhs, "Value":$rhs,
      CArg<"::mlir::puarith::FastMathFlags",
           "::mlir::puarith::FastMathFlags::none">:$fastmath), [{
      build($_builder, $_state, type, lhs, rhs,
            ::mlir::puarith::FastMathFlagsAttr::get(
                $_builder.getContext(), fastmath),
            ::mlir::puarith::RoundingModeAttr{});
    }]>];
  let assemblyFormat = [{ $lhs `,` $rhs ($roundingmode^)?
                          (`fastmath` `` $fastmath^)?
                          attr-dict `:` type($result) }];
}

def SameInputOutputTensorDims : PredOpTrait<
    "input and output have the same tensor dimensions",
    AllMatchSameOperatorPred<["in", "out"],
      "(::llvm::isa<::mlir::TensorType>($_self.getType()) ?"
      " ::llvm::cast<::mlir::TensorType>($_self.getType()).getShape() :"
      " ::llvm::ArrayRef<int64_t>{})">>;

class PuArith_CastOp<string mnemonic, TypeConstraint From, TypeConstraint To,
                   list<Trait> traits = []> :
    PuArith_Op<mnemonic, traits # [Pure, SameOperandsAndResultShape,
      SameInputOutputTensorDims, DeclareOpInterfaceMethods<CastOpInterface>,
      PuArith_CastInterface]>,
    Arguments<(ins From:$in)>,
    Results<(outs To:$out)> {
  let assemblyFormat = "$in attr-dict `:` type($in) `to` type($out)";
}

def SignlessFixedWidthIntegerLike : TypeConstraint<Or<[
        AnyNonZeroBitwidthSignlessInteger.predicate,
        VectorOfAnyRankOf<[AnyNonZeroBitwidthSignlessInteger]>.predicate,
        TensorOf<[AnyNonZeroBitwidthSignlessInteger]>.predicate]>,
    "signless-fixed-width-integer-like">;

class PuArith_IToICastOp<string mnemonic, list<Trait> traits = []> :
    PuArith_CastOp<mnemonic, SignlessFixedWidthIntegerLike,
                           SignlessFixedWidthIntegerLike,
                           traits #
                           []>;
class PuArith_IToFCastOp<string mnemonic, list<Trait> traits = []> :
    PuArith_CastOp<mnemonic, SignlessFixedWidthIntegerLike, FloatLike, traits>;
class PuArith_FToICastOp<string mnemonic, list<Trait> traits = []> :
    PuArith_CastOp<mnemonic, FloatLike, SignlessFixedWidthIntegerLike, traits>;
class PuArith_FToFCastOp<string mnemonic, list<Trait> traits = []> :
    PuArith_CastOp<mnemonic, FloatLike, FloatLike, traits>;

class PuArith_CompareOp<string mnemonic, list<Trait> traits = []> :
    PuArith_Op<mnemonic, traits # [Pure, SameTypeOperands]> {
  let results = (outs AnyType:$result);

  let assemblyFormat = "$predicate `,` $lhs `,` $rhs attr-dict `:` type($lhs) `->` type($result)";

  let hasVerifier = 1;
}

class PuArith_IntBinaryOpWithOverflowFlags<string mnemonic, list<Trait> traits = []> :
    PuArith_BinaryOp<mnemonic, traits #
      [Pure, DeclareOpInterfaceMethods<PuArithIntegerOverflowFlagsInterface>]>,
    Arguments<(ins PuArith_SignlessIntegerOrIndexLike:$lhs, PuArith_SignlessIntegerOrIndexLike:$rhs,
      DefaultValuedAttr<
        PuArith_IntegerOverflowAttr,
        "::mlir::puarith::IntegerOverflowFlags::none">:$overflowFlags)>,
    Results<(outs PuArith_SignlessIntegerOrIndexLike:$result)> {

  let assemblyFormat = [{ $lhs `,` $rhs (`overflow` `` $overflowFlags^)?
                          attr-dict `:` type($result) }];
}

class PuArith_IntBinaryOpWithExactFlag<string mnemonic, list<Trait> traits = []> :
    PuArith_BinaryOp<mnemonic, traits #
      []>,
    Arguments<(ins PuArith_SignlessIntegerOrIndexLike:$lhs,
               PuArith_SignlessIntegerOrIndexLike:$rhs,
               UnitAttr:$isExact)>,
    Results<(outs PuArith_SignlessIntegerOrIndexLike:$result)> {

  let assemblyFormat = [{ $lhs `,` $rhs (`exact` $isExact^)?
                          attr-dict `:` type($result) }];
}

def PuArith_ConstantOp : Op<PuArith_Dialect, "constant",
    [ConstantLike, Pure,
     DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmResultNames"]>,
     AllTypesMatch<["value", "result"]>]> {
  let summary = "integer or floating point constant";
  let description = [{
    The `constant` operation produces an SSA value equal to some integer or
    floating-point constant specified by an attribute. This is the way MLIR
    forms simple integer and floating point constants.

    Example:

    ```
    %1 = puarith.constant 42 : i32

    %1 = "arith.constant"() {value = 42 : i32} : () -> i32
    ```
  }];

  let arguments = (ins TypedAttrInterface:$value);
  let results = (outs  AnyType:$result);

  let extraClassDeclaration = [{
    static bool isBuildableWith(Attribute value, Type type);

    static ConstantOp materialize(OpBuilder &builder, Attribute value,
                                  Type type, Location loc);
  }];

  let hasFolder = 1;
  let assemblyFormat = "attr-dict $value";
  let hasVerifier = 1;
}

def PuArith_AddIOp : PuArith_IntBinaryOpWithOverflowFlags<"addi",
    [Commutative]> {
  let summary = "integer addition operation";
  let description = [{
    Performs N-bit addition on the operands. The operands are interpreted as
    unsigned bitvectors. The result is represented by a bitvector containing the
    mathematical value of the addition modulo 2^n, where `n` is the bitwidth.
    Because `arith` integers use a two's complement representation, this operation
    is applicable on both signed and unsigned integer operands.

    The `addi` operation takes two operands and returns one result, each of
    these is required to be the same type. This type may be an integer scalar type,
    a vector whose element type is integer, or a tensor of integers.

    This op supports `nuw`/`nsw` overflow flags which stands for
    "No Unsigned Wrap" and "No Signed Wrap", respectively. If the `nuw` and/or
    `nsw` flags are present, and an unsigned/signed overflow occurs
    (respectively), the result is poison.

    Example:

    ```mlir
    // Scalar addition.
    %a = puarith.addi %b, %c : i64

    // Scalar addition with overflow flags.
    %a = puarith.addi %b, %c overflow<nsw, nuw> : i64

    // SIMD vector element-wise addition.
    %f = puarith.addi %g, %h : vector<4xi32>

    // Tensor element-wise addition.
    %x = puarith.addi %y, %z : tensor<4x?xi8>
    ```
  }];
  let hasFolder = 1;
  let hasCanonicalizer = 1;
}

//===----------------------------------------------------------------------===//
// AddUIExtendedOp
//===----------------------------------------------------------------------===//

def PuArith_AddUIExtendedOp : PuArith_Op<"addui_extended", [Pure, Commutative,
    AllTypesMatch<["lhs", "rhs", "sum"]>]> {
  let summary = [{
    extended unsigned integer addition operation returning sum and overflow bit
  }];

  let description = [{
    Performs (N+1)-bit addition on zero-extended operands. Returns two results:
    the N-bit sum (same type as both operands), and the overflow bit
    (boolean-like), where `1` indicates unsigned addition overflow, while `0`
    indicates no overflow.

    Example:

    ```mlir
    // Scalar addition.
    %sum, %overflow = puarith.addui_extended %b, %c : i64, i1

    // Vector element-wise addition.
    %d:2 = puarith.addui_extended %e, %f : vector<4xi32>, vector<4xi1>

    // Tensor element-wise addition.
    %x:2 = puarith.addui_extended %y, %z : tensor<4x?xi8>, tensor<4x?xi1>
    ```
  }];

  let arguments = (ins PuArith_SignlessIntegerOrIndexLike:$lhs, PuArith_SignlessIntegerOrIndexLike:$rhs);
  let results = (outs PuArith_SignlessIntegerOrIndexLike:$sum, PuArith_PredI8Like:$overflow);
  let assemblyFormat = [{
    $lhs `,` $rhs attr-dict `:` type($sum) `,` type($overflow)
  }];

  let builders = [
    OpBuilder<(ins "Value":$lhs, "Value":$rhs), [{
      build($_builder, $_state, lhs.getType(), ::mlir::puarith::getI8SameShape(lhs.getType()),
            lhs, rhs);
    }]>
  ];

  let hasFolder = 1;
  let hasCanonicalizer = 1;

  let extraClassDeclaration = [{
    std::optional<SmallVector<int64_t, 4>> getShapeForUnroll();
  }];
}

//===----------------------------------------------------------------------===//
// SubUIExtendedOp
//===----------------------------------------------------------------------===//

def PuArith_SubUIExtendedOp : PuArith_Op<"subui_extended", [Pure,
    AllTypesMatch<["lhs", "rhs", "diff"]>]> {
  let summary = [{
    extended unsigned integer subtraction operation returning difference and
    borrow bit
  }];

  let description = [{
    Performs (N+1)-bit subtraction on zero-extended operands. Returns two
    results: the N-bit difference (same type as both operands), and the borrow
    bit (boolean-like), where `1` indicates unsigned subtraction underflow
    (i.e. `lhs < rhs` when interpreted as unsigned), while `0` indicates no
    underflow.

    Example:

    ```mlir
    // Scalar subtraction.
    %diff, %borrow = puarith.subui_extended %b, %c : i64, i1

    // Vector element-wise subtraction.
    %d:2 = puarith.subui_extended %e, %f : vector<4xi32>, vector<4xi1>

    // Tensor element-wise subtraction.
    %x:2 = puarith.subui_extended %y, %z : tensor<4x?xi8>, tensor<4x?xi1>
    ```
  }];

  let arguments = (ins PuArith_SignlessIntegerOrIndexLike:$lhs, PuArith_SignlessIntegerOrIndexLike:$rhs);
  let results = (outs PuArith_SignlessIntegerOrIndexLike:$diff, PuArith_PredI8Like:$borrow);
  let assemblyFormat = [{
    $lhs `,` $rhs attr-dict `:` type($diff) `,` type($borrow)
  }];

  let builders = [
    OpBuilder<(ins "Value":$lhs, "Value":$rhs), [{
      build($_builder, $_state, lhs.getType(), ::mlir::puarith::getI8SameShape(lhs.getType()),
            lhs, rhs);
    }]>
  ];

  let hasFolder = 1;
  let hasCanonicalizer = 1;

  let extraClassDeclaration = [{
    std::optional<SmallVector<int64_t, 4>> getShapeForUnroll();
  }];
}

//===----------------------------------------------------------------------===//
// SubIOp
//===----------------------------------------------------------------------===//

def PuArith_SubIOp : PuArith_IntBinaryOpWithOverflowFlags<"subi",
    []> {
  let summary = [{
    Integer subtraction operation.
  }];
  let description = [{
    Performs N-bit subtraction on the operands. The operands are interpreted as unsigned
    bitvectors. The result is represented by a bitvector containing the mathematical
    value of the subtraction modulo 2^n, where `n` is the bitwidth. Because `arith`
    integers use a two's complement representation, this operation is applicable on
    both signed and unsigned integer operands.

    The `subi` operation takes two operands and returns one result, each of
    these is required to be the same type. This type may be an integer scalar type,
    a vector whose element type is integer, or a tensor of integers.

    This op supports `nuw`/`nsw` overflow flags which stands for
    "No Unsigned Wrap" and "No Signed Wrap", respectively. If the `nuw` and/or
    `nsw` flags are present, and an unsigned/signed overflow occurs
    (respectively), the result is poison.

    Example:

    ```mlir
    %a = puarith.subi %b, %c : i64

    %a = puarith.subi %b, %c overflow<nsw, nuw> : i64

    %f = puarith.subi %g, %h : vector<4xi32>

    %x = puarith.subi %y, %z : tensor<4x?xi8>
    ```
  }];
  let hasFolder = 1;
  let hasCanonicalizer = 1;
}

def PuArith_MulIOp : PuArith_IntBinaryOpWithOverflowFlags<"muli",
  [Commutative,
   DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmResultNames"]>]
> {
  let summary = [{
    Integer multiplication operation.
  }];
  let description = [{
    Performs N-bit multiplication on the operands. The operands are interpreted as
    unsigned bitvectors. The result is represented by a bitvector containing the
    mathematical value of the multiplication modulo 2^n, where `n` is the bitwidth.
    Because `arith` integers use a two's complement representation, this operation is
    applicable on both signed and unsigned integer operands.

    The `muli` operation takes two operands and returns one result, each of
    these is required to be the same type. This type may be an integer scalar type,
    a vector whose element type is integer, or a tensor of integers.

    This op supports `nuw`/`nsw` overflow flags which stands for
    "No Unsigned Wrap" and "No Signed Wrap", respectively. If the `nuw` and/or
    `nsw` flags are present, and an unsigned/signed overflow occurs
    (respectively), the result is poison.

    Example:

    ```mlir
    // Scalar multiplication.
    %a = puarith.muli %b, %c : i64

    // Scalar multiplication with overflow flags.
    %a = puarith.muli %b, %c overflow<nsw, nuw> : i64

    // SIMD vector element-wise multiplication.
    %f = puarith.muli %g, %h : vector<4xi32>

    // Tensor element-wise multiplication.
    %x = puarith.muli %y, %z : tensor<4x?xi8>
    ```
  }];
  let hasFolder = 1;
  let hasCanonicalizer = 1;
}

//===----------------------------------------------------------------------===//
// MulSIExtendedOp
//===----------------------------------------------------------------------===//

def PuArith_MulSIExtendedOp : PuArith_Op<"mulsi_extended", [Pure, Commutative,
    AllTypesMatch<["lhs", "rhs", "low", "high"]>]> {
  let summary = [{
    extended signed integer multiplication operation
  }];

  let description = [{
    Performs (2*N)-bit multiplication on sign-extended operands. Returns two
    N-bit results: the low and the high halves of the product. The low half has
    the same value as the result of regular multiplication `puarith.muli` with
    the same operands.

    Example:

    ```mlir
    // Scalar multiplication.
    %low, %high = puarith.mulsi_extended %a, %b : i32

    // Vector element-wise multiplication.
    %c:2 = puarith.mulsi_extended %d, %e : vector<4xi32>

    // Tensor element-wise multiplication.
    %x:2 = puarith.mulsi_extended %y, %z : tensor<4x?xi8>
    ```
  }];

  let arguments = (ins PuArith_SignlessIntegerOrIndexLike:$lhs, PuArith_SignlessIntegerOrIndexLike:$rhs);
  let results = (outs PuArith_SignlessIntegerOrIndexLike:$low, PuArith_SignlessIntegerOrIndexLike:$high);

  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` type($lhs)";

  let hasFolder = 1;
  let hasCanonicalizer = 1;

  let extraClassDeclaration = [{
    std::optional<SmallVector<int64_t, 4>> getShapeForUnroll();
  }];
}

//===----------------------------------------------------------------------===//
// MulUIExtendedOp
//===----------------------------------------------------------------------===//

def PuArith_MulUIExtendedOp : PuArith_Op<"mului_extended", [Pure, Commutative,
    AllTypesMatch<["lhs", "rhs", "low", "high"]>]> {
  let summary = [{
    extended unsigned integer multiplication operation
  }];

  let description = [{
    Performs (2*N)-bit multiplication on zero-extended operands. Returns two
    N-bit results: the low and the high halves of the product. The low half has
    the same value as the result of regular multiplication `puarith.muli` with
    the same operands.

    Example:

    ```mlir
    // Scalar multiplication.
    %low, %high = puarith.mului_extended %a, %b : i32

    // Vector element-wise multiplication.
    %c:2 = puarith.mului_extended %d, %e : vector<4xi32>

    // Tensor element-wise multiplication.
    %x:2 = puarith.mului_extended %y, %z : tensor<4x?xi8>
    ```
  }];

  let arguments = (ins PuArith_SignlessIntegerOrIndexLike:$lhs, PuArith_SignlessIntegerOrIndexLike:$rhs);
  let results = (outs PuArith_SignlessIntegerOrIndexLike:$low, PuArith_SignlessIntegerOrIndexLike:$high);

  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` type($lhs)";

  let hasFolder = 1;
  let hasCanonicalizer = 1;

  let extraClassDeclaration = [{
    std::optional<SmallVector<int64_t, 4>> getShapeForUnroll();
  }];
}

//===----------------------------------------------------------------------===//
// DivUIOp
//===----------------------------------------------------------------------===//

def PuArith_DivUIOp : PuArith_IntBinaryOpWithExactFlag<"divui",
    [ConditionallySpeculatable]> {
  let summary = "unsigned integer division operation";
  let description = [{
    Unsigned integer division. Rounds towards zero. Treats the leading bit as
    the most significant, i.e. for `i16` given two's complement representation,
    `6 / -2 = 6 / (2^16 - 2) = 0`.

    Division by zero is undefined behavior. When applied to `vector` and
    `tensor` values, the behavior is undefined if _any_ elements are divided by
    zero.

    If the `exact` attribute is present, the result value is poison if `lhs` is
    not a multiple of `rhs`.

    Example:

    ```mlir
    %a = puarith.divui %b, %c : i64

    %a = puarith.divui %b, %c exact : i64

    %f = puarith.divui %g, %h : vector<4xi32>

    %x = puarith.divui %y, %z : tensor<4x?xi8>
    ```
  }];

  let extraClassDeclaration = [{
    Speculation::Speculatability getSpeculatability();
  }];

  let hasFolder = 1;
}

def PuArith_DivSIOp : PuArith_IntBinaryOpWithExactFlag<"divsi",
                                                   [ConditionallySpeculatable]> {
  let summary = "signed integer division operation";
  let description = [{
    Signed integer division. Rounds towards zero. Treats the leading bit as
    sign, i.e. `6 / -2 = -3`.

    Divison by zero, or signed division overflow (minimum value divided by -1)
    is undefined behavior. When applied to `vector` and `tensor` values, the
    behavior is undefined if _any_ of its elements are divided by zero or has a
    signed division overflow.

    If the `exact` attribute is present, the result value is poison if `lhs` is
    not a multiple of `rhs`.

    Example:

    ```mlir
    %a = puarith.divsi %b, %c : i64

    %a = puarith.divsi %b, %c exact : i64

    %f = puarith.divsi %g, %h : vector<4xi32>

    %x = puarith.divsi %y, %z : tensor<4x?xi8>
    ```
  }];

  let extraClassDeclaration = [{
    Speculation::Speculatability getSpeculatability();
  }];

  let hasFolder = 1;
}

def PuArith_CeilDivUIOp : PuArith_IntBinaryOp<"ceildivui",
                                          [ConditionallySpeculatable]> {
  let summary = "unsigned ceil integer division operation";
  let description = [{
    Unsigned integer division. Rounds towards positive infinity. Treats the
    leading bit as the most significant, i.e. for `i16` given two's complement
    representation, `6 / -2 = 6 / (2^16 - 2) = 1`.

    Division by zero is undefined behavior. When applied to `vector` and
    `tensor` values, the behavior is undefined if _any_ elements are divided by
    zero.

    Example:

    ```mlir
    // Scalar unsigned integer division.
    %a = puarith.ceildivui %b, %c : i64
    ```
  }];

  let extraClassDeclaration = [{
    /// Interface method for ConditionallySpeculatable.
    Speculation::Speculatability getSpeculatability();
  }];

  let hasFolder = 1;
}

//===----------------------------------------------------------------------===//
// CeilDivSIOp
//===----------------------------------------------------------------------===//

def PuArith_CeilDivSIOp : PuArith_IntBinaryOp<"ceildivsi",
                                          [ConditionallySpeculatable]> {
  let summary = "signed ceil integer division operation";
  let description = [{
    Signed integer division. Rounds towards positive infinity, i.e. `7 / -2 = -3`.

    Divison by zero, or signed division overflow (minimum value divided by -1)
    is undefined behavior. When applied to `vector` and `tensor` values, the
    behavior is undefined if _any_ of its elements are divided by zero or has a
    signed division overflow.

    Example:

    ```mlir
    // Scalar signed integer division.
    %a = puarith.ceildivsi %b, %c : i64
    ```
  }];

  let extraClassDeclaration = [{
    /// Interface method for ConditionallySpeculatable.
    Speculation::Speculatability getSpeculatability();
  }];

  let hasFolder = 1;
}

//===----------------------------------------------------------------------===//
// FloorDivSIOp
//===----------------------------------------------------------------------===//

def PuArith_FloorDivSIOp : PuArith_TotalIntBinaryOp<"floordivsi"> {
  let summary = "signed floor integer division operation";
  let description = [{
    Signed integer division. Rounds towards negative infinity, i.e. `5 / -2 = -3`.

    Divison by zero, or signed division overflow (minimum value divided by -1)
    is undefined behavior. When applied to `vector` and `tensor` values, the
    behavior is undefined if _any_ of its elements are divided by zero or has a
    signed division overflow.

    Example:

    ```mlir
    // Scalar signed integer division.
    %a = puarith.floordivsi %b, %c : i64

    ```
  }];
  let hasFolder = 1;
}

//===----------------------------------------------------------------------===//
// RemUIOp
//===----------------------------------------------------------------------===//

def PuArith_RemUIOp : PuArith_IntBinaryOp<"remui",
                                      [ConditionallySpeculatable]> {
  let summary = "unsigned integer division remainder operation";
  let description = [{
    Unsigned integer division remainder. Treats the leading bit as the most
    significant, i.e. for `i16`, `6 % -2 = 6 % (2^16 - 2) = 6`.

    Division by zero is undefined behavior. When applied to `vector` and
    `tensor` values, the behavior is undefined if _any_ elements are divided by
    zero.

    Example:

    ```mlir
    // Scalar unsigned integer division remainder.
    %a = puarith.remui %b, %c : i64

    // SIMD vector element-wise division remainder.
    %f = puarith.remui %g, %h : vector<4xi32>

    // Tensor element-wise integer division remainder.
    %x = puarith.remui %y, %z : tensor<4x?xi8>
    ```
  }];

  let extraClassDeclaration = [{
    /// Interface method for ConditionallySpeculatable.
    Speculation::Speculatability getSpeculatability();
  }];

  let hasFolder = 1;
}

//===----------------------------------------------------------------------===//
// RemSIOp
//===----------------------------------------------------------------------===//

def PuArith_RemSIOp : PuArith_IntBinaryOp<"remsi",
                                      [ConditionallySpeculatable]> {
  let summary = "signed integer division remainder operation";
  let description = [{
    Signed integer division remainder. Treats the leading bit as sign, i.e. `6 %
    -2 = 0`.

    Division by zero is undefined behavior. When applied to `vector` and
    `tensor` values, the behavior is undefined if _any_ elements are divided by
    zero.

    Example:

    ```mlir
    // Scalar signed integer division remainder.
    %a = puarith.remsi %b, %c : i64

    // SIMD vector element-wise division remainder.
    %f = puarith.remsi %g, %h : vector<4xi32>

    // Tensor element-wise integer division remainder.
    %x = puarith.remsi %y, %z : tensor<4x?xi8>
    ```
  }];

  let extraClassDeclaration = [{
    /// Interface method for ConditionallySpeculatable.
    Speculation::Speculatability getSpeculatability();
  }];

  let hasFolder = 1;
}

//===----------------------------------------------------------------------===//
// AndIOp
//===----------------------------------------------------------------------===//

def PuArith_AndIOp : PuArith_TotalIntBinaryOp<"andi", [Commutative, Idempotent]> {
  let summary = "integer binary and";
  let description = [{
    The `andi` operation takes two operands and returns one result, each of
    these is required to be the same type. This type may be an integer scalar
    type, a vector whose element type is integer, or a tensor of integers. It
    has no standard attributes.

    Example:

    ```mlir
    // Scalar integer bitwise and.
    %a = puarith.andi %b, %c : i64

    // SIMD vector element-wise bitwise integer and.
    %f = puarith.andi %g, %h : vector<4xi32>

    // Tensor element-wise bitwise integer and.
    %x = puarith.andi %y, %z : tensor<4x?xi8>
    ```
  }];
  let hasFolder = 1;
  let hasCanonicalizer = 1;
}

//===----------------------------------------------------------------------===//
// OrIOp
//===----------------------------------------------------------------------===//

def PuArith_OrIOp : PuArith_TotalIntBinaryOp<"ori", [Commutative, Idempotent]> {
  let summary = "integer binary or";
  let description = [{
    The `ori` operation takes two operands and returns one result, each of these
    is required to be the same type. This type may be an integer scalar type, a
    vector whose element type is integer, or a tensor of integers. It has no
    standard attributes.

    Example:

    ```mlir
    // Scalar integer bitwise or.
    %a = puarith.ori %b, %c : i64

    // SIMD vector element-wise bitwise integer or.
    %f = puarith.ori %g, %h : vector<4xi32>

    // Tensor element-wise bitwise integer or.
    %x = puarith.ori %y, %z : tensor<4x?xi8>
    ```
  }];
  let hasFolder = 1;
  let hasCanonicalizer = 1;
}

//===----------------------------------------------------------------------===//
// XOrIOp
//===----------------------------------------------------------------------===//

def PuArith_XOrIOp : PuArith_TotalIntBinaryOp<"xori", [Commutative]> {
  let summary = "integer binary xor";
  let description = [{
    The `xori` operation takes two operands and returns one result, each of
    these is required to be the same type. This type may be an integer scalar
    type, a vector whose element type is integer, or a tensor of integers. It
    has no standard attributes.

    Example:

    ```mlir
    // Scalar integer bitwise xor.
    %a = puarith.xori %b, %c : i64

    // SIMD vector element-wise bitwise integer xor.
    %f = puarith.xori %g, %h : vector<4xi32>

    // Tensor element-wise bitwise integer xor.
    %x = puarith.xori %y, %z : tensor<4x?xi8>
    ```
  }];
  let hasFolder = 1;
  let hasCanonicalizer = 1;
}

//===----------------------------------------------------------------------===//
// ShLIOp
//===----------------------------------------------------------------------===//

def PuArith_ShLIOp : PuArith_IntBinaryOpWithOverflowFlags<"shli"> {
  let summary = "integer left-shift";
  let description = [{
    The `shli` operation shifts the integer value of the first operand to the left
    by the integer value of the second operand. The second operand is interpreted as
    unsigned. The low order bits are filled with zeros. If the value of the second
    operand is greater or equal than the bitwidth of the first operand, then the
    operation returns poison.

    This op supports `nuw`/`nsw` overflow flags which stands for
    "No Unsigned Wrap" and "No Signed Wrap", respectively. If the `nuw` and/or
    `nsw` flags are present, and an unsigned/signed overflow occurs
    (respectively), the result is poison.

    Example:

    ```mlir
    %1 = puarith.constant 5 : i8  // %1 is 0b00000101
    %2 = puarith.constant 3 : i8
    %3 = puarith.shli %1, %2 : i8 // %3 is 0b00101000
    %4 = puarith.shli %1, %2 overflow<nsw, nuw> : i8
    ```
  }];
  let hasFolder = 1;
}

//===----------------------------------------------------------------------===//
// ShRUIOp
//===----------------------------------------------------------------------===//

def PuArith_ShRUIOp : PuArith_IntBinaryOpWithExactFlag<"shrui", [Pure]> {
  let summary = "unsigned integer right-shift";
  let description = [{
    The `shrui` operation shifts an integer value of the first operand to the right
    by the value of the second operand. The first operand is interpreted as unsigned,
    and the second operand is interpreted as unsigned. The high order bits are always
    filled with zeros. If the value of the second operand is greater or equal than the
    bitwidth of the first operand, then the operation returns poison.

    If the `exact` attribute is present, the result value of shrui is a poison
    value if any of the bits shifted out are non-zero.

    Example:

    ```mlir
    %1 = puarith.constant 160 : i8        // %1 is 0b10100000
    %2 = puarith.constant 3 : i8
    %3 = puarith.constant 6 : i8
    %4 = puarith.shrui %1, %2 exact : i8  // %4 is 0b00010100
    %5 = puarith.shrui %1, %3 : i8        // %3 is 0b00000010
    ```
  }];
  let hasFolder = 1;
}

//===----------------------------------------------------------------------===//
// ShRSIOp
//===----------------------------------------------------------------------===//

def PuArith_ShRSIOp : PuArith_IntBinaryOpWithExactFlag<"shrsi", [Pure]> {
  let summary = "signed integer right-shift";
  let description = [{
    The `shrsi` operation shifts an integer value of the first operand to the right
    by the value of the second operand. The first operand is interpreted as signed,
    and the second operand is interpreter as unsigned. The high order bits in the
    output are filled with copies of the most-significant bit of the shifted value
    (which means that the sign of the value is preserved). If the value of the second
    operand is greater or equal than bitwidth of the first operand, then the operation
    returns poison.

    If the `exact` attribute is present, the result value of shrsi is a poison
    value if any of the bits shifted out are non-zero.

    Example:

    ```mlir
    %1 = puarith.constant 160 : i8         // %1 is 0b10100000
    %2 = puarith.constant 3 : i8
    %3 = puarith.shrsi %1, %2 exact : i8   // %3 is 0b11110100
    %4 = puarith.constant 98 : i8          // %4 is 0b01100010
    %5 = puarith.shrsi %4, %2 : i8         // %5 is 0b00001100
    ```
  }];
  let hasFolder = 1;
}

//===----------------------------------------------------------------------===//
// NegFOp
//===----------------------------------------------------------------------===//

def PuArith_NegFOp : PuArith_FloatUnaryOp<"negf"> {
  let summary = "floating point negation";
  let description = [{
    The `negf` operation computes the negation of a given value. It takes one
    operand and returns one result of the same type. This type may be a float
    scalar type, a vector whose element type is float, or a tensor of floats.
    It has no standard attributes.

    Example:

    ```mlir
    // Scalar negation value.
    %a = puarith.negf %b : f64

    // SIMD vector element-wise negation value.
    %f = puarith.negf %g : vector<4xf32>

    // Tensor element-wise negation value.
    %x = puarith.negf %y : tensor<4x?xf8>
    ```
  }];
  let hasFolder = 1;
}

//===----------------------------------------------------------------------===//
// FlushDenormalsOp
//===----------------------------------------------------------------------===//

def PuArith_FlushDenormalsOp : PuArith_FloatUnaryOp<"flush_denormals"> {
  let summary = "flush denormal floating-point values to zero";
  let description = [{
    The `flush_denormals` operation takes a floating-point value and returns
    the input value if it is a normal (or zero, infinity, or NaN) value, or
    a zero of the same type if the input is a denormal (subnormal) value.
    The sign of zero is preserved when flushing a denormal: negative
    denormals flush to `-0.0`, positive denormals flush to `+0.0`.

    A denormal number ("subnormal number" in IEEE-754) is a non-zero floating
    point number that is smaller (closer to zero) than the smallest normal
    number. Denormals fill the underflow gap around zero in floating-point
    arithmetics, but may come at a runtime cost on some architectures.

    The input and result are required to be the same type. This type may be
    a floating-point scalar type, a vector whose element type is a
    floating-point type, or a tensor of floats. When operating on vectors
    or tensors, the operation is applied elementwise.

    Example:

    ```mlir
    // Scalar denormal flushing.
    %a = puarith.flush_denormals %b : f32

    // SIMD vector element-wise denormal flushing.
    %f = puarith.flush_denormals %g : vector<4xf32>

    // Tensor element-wise denormal flushing.
    %x = puarith.flush_denormals %y : tensor<4x?xbf16>
    ```
  }];
  let hasFolder = 1;
}

//===----------------------------------------------------------------------===//
// AddFOp
//===----------------------------------------------------------------------===//

def PuArith_AddFOp : PuArith_FloatBinaryOpWithRoundingMode<"addf", [Commutative]> {
  let summary = "floating point addition operation";
  let description = [{
    The `addf` operation takes two operands and returns one result, each of
    these is required to be the same type. This type may be a floating point
    scalar type, a vector whose element type is a floating point type, or a
    floating point tensor.

    Example:

    ```mlir
    // Scalar addition.
    %a = puarith.addf %b, %c : f64

    // SIMD vector addition, e.g. for Intel SSE.
    %f = puarith.addf %g, %h : vector<4xf32>

    // Tensor addition.
    %x = puarith.addf %y, %z : tensor<4x?xbf16>

    // Scalar addition with rounding mode.
    %a = puarith.addf %b, %c to_nearest_even : f64
    ```
  }];
  let hasCanonicalizer = 1;
  let hasFolder = 1;
}

//===----------------------------------------------------------------------===//
// SubFOp
//===----------------------------------------------------------------------===//

def PuArith_SubFOp : PuArith_FloatBinaryOpWithRoundingMode<"subf"> {
  let summary = "floating point subtraction operation";
  let description = [{
    The `subf` operation takes two operands and returns one result, each of
    these is required to be the same type. This type may be a floating point
    scalar type, a vector whose element type is a floating point type, or a
    floating point tensor.

    Example:

    ```mlir
    // Scalar subtraction.
    %a = puarith.subf %b, %c : f64

    // SIMD vector subtraction, e.g. for Intel SSE.
    %f = puarith.subf %g, %h : vector<4xf32>

    // Tensor subtraction.
    %x = puarith.subf %y, %z : tensor<4x?xbf16>

    // Scalar subtraction with rounding mode.
    %a = puarith.subf %b, %c downward : f64
    ```
  }];
  let hasFolder = 1;
  let hasCanonicalizer = 1;
}

//===----------------------------------------------------------------------===//
// MaximumFOp
//===----------------------------------------------------------------------===//

def PuArith_MaximumFOp : PuArith_FloatBinaryOp<"maximumf", [Commutative]> {
  let summary = "floating-point maximum operation";
  let description = [{
    Returns the maximum of the two arguments, treating -0.0 as less than +0.0.
    If one of the arguments is NaN, then the result is also NaN.

    Example:

    ```mlir
    // Scalar floating-point maximum.
    %a = puarith.maximumf %b, %c : f64
    ```
  }];
  let hasFolder = 1;
}

//===----------------------------------------------------------------------===//
// MaxNumFOp
//===----------------------------------------------------------------------===//

def PuArith_MaxNumFOp : PuArith_FloatBinaryOp<"maxnumf", [Commutative]> {
  let summary = "floating-point maximum operation";
  let description = [{
    Returns the maximum of the two arguments.
    If the arguments are -0.0 and +0.0, then the result is either of them.
    If one of the arguments is NaN, then the result is the other argument.

    Example:

    ```mlir
    // Scalar floating-point maximum.
    %a = puarith.maxnumf %b, %c : f64
    ```
  }];
  let hasFolder = 1;
}


//===----------------------------------------------------------------------===//
// MaxSIOp
//===----------------------------------------------------------------------===//

def PuArith_MaxSIOp : PuArith_TotalIntBinaryOp<"maxsi",
    [Commutative]> {
  let summary = "signed integer maximum operation";
  let hasFolder = 1;
}

//===----------------------------------------------------------------------===//
// MaxUIOp
//===----------------------------------------------------------------------===//

def PuArith_MaxUIOp : PuArith_TotalIntBinaryOp<"maxui",
    [Commutative]> {
  let summary = "unsigned integer maximum operation";
  let hasFolder = 1;
}

//===----------------------------------------------------------------------===//
// MinimumFOp
//===----------------------------------------------------------------------===//

def PuArith_MinimumFOp : PuArith_FloatBinaryOp<"minimumf", [Commutative]> {
  let summary = "floating-point minimum operation";
  let description = [{
    Returns the minimum of the two arguments, treating -0.0 as less than +0.0.
    If one of the arguments is NaN, then the result is also NaN.

    Example:

    ```mlir
    // Scalar floating-point minimum.
    %a = puarith.minimumf %b, %c : f64
    ```
  }];
  let hasFolder = 1;
}

//===----------------------------------------------------------------------===//
// MinNumFOp
//===----------------------------------------------------------------------===//

def PuArith_MinNumFOp : PuArith_FloatBinaryOp<"minnumf", [Commutative]> {
  let summary = "floating-point minimum operation";
  let description = [{
    Returns the minimum of the two arguments.
    If the arguments are -0.0 and +0.0, then the result is either of them.
    If one of the arguments is NaN, then the result is the other argument.

    Example:

    ```mlir
    // Scalar floating-point minimum.
    %a = puarith.minnumf %b, %c : f64
    ```
  }];
  let hasFolder = 1;
}

//===----------------------------------------------------------------------===//
// MinSIOp
//===----------------------------------------------------------------------===//

def PuArith_MinSIOp : PuArith_TotalIntBinaryOp<"minsi",
    [Commutative]> {
  let summary = "signed integer minimum operation";
  let hasFolder = 1;
}

//===----------------------------------------------------------------------===//
// MinUIOp
//===----------------------------------------------------------------------===//

def PuArith_MinUIOp : PuArith_TotalIntBinaryOp<"minui",
    [Commutative]> {
  let summary = "unsigned integer minimum operation";
  let hasFolder = 1;
}


//===----------------------------------------------------------------------===//
// MulFOp
//===----------------------------------------------------------------------===//

def PuArith_MulFOp : PuArith_FloatBinaryOpWithRoundingMode<"mulf", [Commutative]> {
  let summary = "floating point multiplication operation";
  let description = [{
    The `mulf` operation takes two operands and returns one result, each of
    these is required to be the same type. This type may be a floating point
    scalar type, a vector whose element type is a floating point type, or a
    floating point tensor.

    Example:

    ```mlir
    // Scalar multiplication.
    %a = puarith.mulf %b, %c : f64

    // SIMD pointwise vector multiplication, e.g. for Intel SSE.
    %f = puarith.mulf %g, %h : vector<4xf32>

    // Tensor pointwise multiplication.
    %x = puarith.mulf %y, %z : tensor<4x?xbf16>

    // Scalar multiplication with rounding mode.
    %a = puarith.mulf %b, %c upward : f64
    ```
  }];
  let hasFolder = 1;
  let hasCanonicalizer = 1;
}

//===----------------------------------------------------------------------===//
// DivFOp
//===----------------------------------------------------------------------===//

def PuArith_DivFOp : PuArith_FloatBinaryOpWithRoundingMode<"divf"> {
  let summary = "floating point division operation";
  let description = [{
    The `divf` operation takes two operands and returns one result, each of
    these is required to be the same type. This type may be a floating point
    scalar type, a vector whose element type is a floating point type, or a
    floating point tensor.

    Example:

    ```mlir
    // Scalar division.
    %a = puarith.divf %b, %c : f64

    // Scalar division with rounding mode.
    %a = puarith.divf %b, %c toward_zero : f64
    ```
  }];
  let hasFolder = 1;
  let hasCanonicalizer = 1;
}

//===----------------------------------------------------------------------===//
// RemFOp
//===----------------------------------------------------------------------===//

def PuArith_RemFOp : PuArith_FloatBinaryOp<"remf"> {
  let summary = "floating point division remainder operation";
  let description = [{
    Returns the floating point division remainder.
    The remainder has the same sign as the dividend (lhs operand).

    TODO: Add support for rounding modes.
  }];
  let hasFolder = 1;
}

//===----------------------------------------------------------------------===//
// ExtUIOp
//===----------------------------------------------------------------------===//

def PuArith_ExtUIOp :
    PuArith_IToICastOp<"extui", [DeclareOpInterfaceMethods<PuArithNonNegFlagInterface>]> {
  let summary = "integer zero extension operation";
  let description = [{
    The integer zero extension operation takes an integer input of
    width M and an integer destination type of width N. The destination
    bit-width must be larger than the input bit-width (N > M).
    The top-most (N - M) bits of the output are filled with zeros.

    When the `nneg` flag is present, the operand is assumed to have
    the most significant bit set to 0. In this case, zero extension is
    equivalent to sign extension. When this assumption is violated, the
    result is poison.

    Example:

    ```mlir
      %1 = puarith.constant 5 : i3      // %1 is 0b101
      %2 = puarith.extui %1 : i3 to i6  // %2 is 0b000101
      %3 = puarith.constant 2 : i3      // %3 is 0b010
      %4 = puarith.extui %3 : i3 to i6  // %4 is 0b000010

      %5 = puarith.extui %0 : vector<2 x i32> to vector<2 x i64>

      // Zero extension with nneg flag.
      %6 = puarith.extui %3 nneg : i3 to i6
    ```
  }];

  let arguments = (ins SignlessFixedWidthIntegerLike:$in, UnitAttr:$nonNeg);
  let results = (outs SignlessFixedWidthIntegerLike:$out);
  let assemblyFormat = [{
    $in (`nneg` $nonNeg^)? attr-dict `:` type($in) `to` type($out)
  }];
  let hasFolder = 1;
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// ExtSIOp
//===----------------------------------------------------------------------===//

def PuArith_ExtSIOp : PuArith_IToICastOp<"extsi"> {
  let summary = "integer sign extension operation";

  let description = [{
    The integer sign extension operation takes an integer input of
    width M and an integer destination type of width N. The destination
    bit-width must be larger than the input bit-width (N > M).
    The top-most (N - M) bits of the output are filled with copies
    of the most-significant bit of the input.

    Example:

    ```mlir
    %1 = puarith.constant 5 : i3      // %1 is 0b101
    %2 = puarith.extsi %1 : i3 to i6  // %2 is 0b111101
    %3 = puarith.constant 2 : i3      // %3 is 0b010
    %4 = puarith.extsi %3 : i3 to i6  // %4 is 0b000010

    %5 = puarith.extsi %0 : vector<2 x i32> to vector<2 x i64>
    ```
  }];

  let hasFolder = 1;
  let hasCanonicalizer = 1;
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// ExtFOp
//===----------------------------------------------------------------------===//

def PuArith_ExtFOp : PuArith_FToFCastOp<"extf", [DeclareOpInterfaceMethods<PuArithFastMathInterface>]> {
  let summary = "cast from floating-point to wider floating-point";
  let description = [{
    Cast a floating-point value to a larger floating-point-typed value.
    The destination type must to be strictly wider than the source type.
    When operating on vectors, casts elementwise.
  }];
  let hasVerifier = 1;
  let hasFolder = 1;

  let arguments = (ins FloatLike:$in,
                       OptionalAttr<PuArith_FastMathAttr>:$fastmath);
  let results = (outs FloatLike:$out);

  let assemblyFormat = [{ $in (`fastmath` `` $fastmath^)?
                          attr-dict `:` type($in) `to` type($out) }];
}

//===----------------------------------------------------------------------===//
// Scaling ExtFOp
//===----------------------------------------------------------------------===//
def PuArith_ScalingExtFOp
    : PuArith_Op<
          "scaling_extf", [Pure, SameInputOutputTensorDims,
                           DeclareOpInterfaceMethods<PuArithFastMathInterface>,
                           DeclareOpInterfaceMethods<CastOpInterface>]>,
      Arguments<(ins FloatLike:$in, FloatLike:$scale,
          OptionalAttr<PuArith_FastMathAttr>:$fastmath)>,
      Results<(outs FloatLike:$out)> {
  let summary = "Upcasts input floats using provided scales values following "
                "OCP MXFP Spec";
  let description = [{
    This operation upcasts input floating-point values using provided scale
    values. It expects both scales and the input operand to be of the same shape,
    making the operation elementwise. Scales are usually calculated per block
    following the OCP MXFP spec as described in https://arxiv.org/abs/2310.10537.

    If scales are calculated per block where blockSize != 1, then scales may
    require broadcasting to make this operation elementwise. For example, let's
    say the input is of shape `<dim1 x dim2 x ... dimN>`. Given blockSize != 1 and
    assuming quantization happens on the last axis, the input can be reshaped to
    `<dim1 x dim2 x ... (dimN/blockSize) x blockSize>`. Scales will be calculated
    per block on the last axis. Therefore, scales will be of shape
    `<dim1 x dim2 x ... (dimN/blockSize) x 1>`. Scales could also be of some other
    shape as long as it is broadcast compatible with the input, e.g.,
    `<1 x 1 x ... (dimN/blockSize) x 1>`.

    In this example, before calling into `puarith.scaling_extf`, scales must be
    broadcasted to `<dim1 x dim2 x dim3 ... (dimN/blockSize) x blockSize>`. Note
    that there could be multiple quantization axes. Internally,
    `puarith.scaling_extf` would perform the following:

    ```mlir
    %0 = puarith.truncf %1 : f32 to f8E8M0FNU
    %1 = puarith.extf %0 : f8E8M0FNU to f16

    %2 = puarith.extf %3 : f4E2M1FN to f16

    %3 = puarith.mulf %2, %1 : f16
    ```
    It propagates NaN values. Therefore, if either scale or the input element
    contains NaN, then the output element value will also be a NaN.

    Example:

    ```mlir
    %a = puarith.scaling_extf %b, %c : f4E2M1FN, f8E8M0FNU to f32

    %f = vector.broadcast %g : vector<1xf8E8M0FNU> to vector<32xf8E8M0FNU>
    %h = puarith.scaling_extf %i, %f : vector<32xf4E2M1FN>, vector<32xf8E8M0FNU> to vector<32xbf16>
    ```
  }];
  let hasVerifier = 1;
  let assemblyFormat =
      [{ $in `,` $scale (`fastmath` `` $fastmath^)? attr-dict `:`
      type($in) `,` type($scale) `to` type($out)}];
}

def PuArith_TruncIOp : PuArith_Op<"trunci",
    [Pure, SameOperandsAndResultShape, SameInputOutputTensorDims,
     DeclareOpInterfaceMethods<CastOpInterface>,
     DeclareOpInterfaceMethods<PuArithIntegerOverflowFlagsInterface>,
     PuArith_CastInterface]> {
  let summary = "integer truncation operation";
  let description = [{
    The integer truncation operation takes an integer input of
    width M and an integer destination type of width N. The destination
    bit-width must be smaller than the input bit-width (N < M).
    The top-most (N - M) bits of the input are discarded.

    This op supports `nuw`/`nsw` overflow flags which stands for "No Unsigned
    Wrap" and "No Signed Wrap", respectively. If the nuw keyword is present,
    and any of the truncated bits are non-zero, the result is a poison value.
    If the nsw keyword is present, and any of the truncated bits are not the
    same as the top bit of the truncation result, the result is a poison value.

    Example:

    ```mlir
      %1 = puarith.constant 21 : i5
      %2 = puarith.trunci %1 : i5 to i4
      %3 = puarith.trunci %1 : i5 to i3

      %4 = puarith.trunci %0 : vector<2 x i32> to vector<2 x i16>

      %5 = puarith.trunci %a overflow<nsw, nuw> : i32 to i16
    ```
  }];

  let arguments = (ins
      SignlessFixedWidthIntegerLike:$in,
      DefaultValuedAttr<PuArith_IntegerOverflowAttr,
          "::mlir::puarith::IntegerOverflowFlags::none">:$overflowFlags);
  let results = (outs SignlessFixedWidthIntegerLike:$out);
  let assemblyFormat = [{
    $in (`overflow` `` $overflowFlags^)? attr-dict
    `:` type($in) `to` type($out)
  }];
  let hasFolder = 1;
  let hasCanonicalizer = 1;
  let hasVerifier = 1;
}

def PuArith_TruncFOp :
    PuArith_Op<"truncf",
      [Pure, SameOperandsAndResultShape, SameInputOutputTensorDims,
       DeclareOpInterfaceMethods<PuArithRoundingModeInterface>,
       DeclareOpInterfaceMethods<PuArithFastMathInterface>,
       DeclareOpInterfaceMethods<CastOpInterface>,
       PuArith_CastInterface]>,
    Arguments<(ins FloatLike:$in,
                   OptionalAttr<PuArith_RoundingModeAttr>:$roundingmode,
                   OptionalAttr<PuArith_FastMathAttr>:$fastmath)>,
    Results<(outs FloatLike:$out)> {
  let summary = "cast from floating-point to narrower floating-point";
  let description = [{
    Truncate a floating-point value to a smaller floating-point-typed value.
    The destination type must be strictly narrower than the source type.
    When operating on vectors, casts elementwise.
  }];
  let builders = [
    OpBuilder<(ins "Type":$out, "Value":$in), [{
      $_state.addOperands(in);
      $_state.addTypes(out);
    }]>
  ];

  let hasFolder = 1;
  let hasCanonicalizer = 1;
  let hasVerifier = 1;
  let assemblyFormat = [{ $in ($roundingmode^)?
                          (`fastmath` `` $fastmath^)?
                          attr-dict `:` type($in) `to` type($out) }];
}

def PuArith_ConvertFOp :
    PuArith_Op<"convertf",
      [Pure, SameOperandsAndResultShape, SameInputOutputTensorDims,
       DeclareOpInterfaceMethods<PuArithRoundingModeInterface>,
       DeclareOpInterfaceMethods<PuArithFastMathInterface>,
       DeclareOpInterfaceMethods<CastOpInterface>,
       PuArith_CastInterface]>,
    Arguments<(ins FloatLike:$in,
                   OptionalAttr<PuArith_RoundingModeAttr>:$roundingmode,
                   OptionalAttr<PuArith_FastMathAttr>:$fastmath)>,
    Results<(outs FloatLike:$out)> {
  let summary = "cast between floating-point types of the same bitwidth";
  let description = [{
    Cast a floating-point value to a different floating-point type of the same
    bitwidth. This operation handles conversions between types that have the
    same bitwidth but different semantics (e.g., f16 to bf16), which cannot
    be represented by `puarith.extf` or `puarith.truncf`.

    The source and destination element types must be different and must have
    the same bitwidth. When operating on vectors, casts elementwise.
  }];

  let hasFolder = 1;
  let hasVerifier = 1;
  let assemblyFormat = [{ $in ($roundingmode^)?
                          (`fastmath` `` $fastmath^)?
                          attr-dict `:` type($in) `to` type($out) }];
}

def PuArith_ScalingTruncFOp
    : PuArith_Op<"scaling_truncf",
               [Pure, SameInputOutputTensorDims,
                DeclareOpInterfaceMethods<PuArithRoundingModeInterface>,
                DeclareOpInterfaceMethods<PuArithFastMathInterface>,
                DeclareOpInterfaceMethods<CastOpInterface>]>,
      Arguments<(ins FloatLike:$in, FloatLike:$scale,
          OptionalAttr<PuArith_RoundingModeAttr>:$roundingmode,
          OptionalAttr<PuArith_FastMathAttr>:$fastmath)>,
      Results<(outs FloatLike:$out)> {
  let summary = "Downcasts input floating point values using provided scales "
                "values following OCP MXFP Spec";
  let description = [{
    This operation downcasts input using the provided scale values. It expects
    both scales and the input operand to be of the same shape and, therefore,
    makes the operation elementwise. Scales are usually calculated per block
    following the OCP MXFP spec as described in https:
    Users are required to normalize and clamp the scales as necessary before calling
    passing them to this operation.  OCP MXFP spec also does the flushing of denorms
    on the input operand, which should be handled during lowering by passing appropriate
    fastMath flag to this operation.

    If scales are calculated per block where blockSize != 1, scales may require
    broadcasting to make this operation elementwise. For example, let's say the
    input is of shape `<dim1 x dim2 x ... dimN>`. Given blockSize != 1 and
    assuming quantization happens on the last axis, the input can be reshaped to
    `<dim1 x dim2 x ... (dimN/blockSize) x blockSize>`. Scales will be calculated
    per block on the last axis. Therefore, scales will be of shape
    `<dim1 x dim2 x ... (dimN/blockSize) x 1>`. Scales could also be of some other
    shape as long as it is broadcast compatible with the input, e.g.,
    `<1 x 1 x ... (dimN/blockSize) x 1>`.

    In this example, before calling into `puarith.scaling_truncf`, scales must be
    broadcasted to `<dim1 x dim2 x dim3 ... (dimN/blockSize) x blockSize>`. Note
    that there could be multiple quantization axes. Internally,
    `puarith.scaling_truncf` would perform the following:

    ```mlir
    // Cast scale to input type.
    %0 = puarith.truncf %1 : f32 to f8E8M0FNU
    %1 = puarith.extf %0 : f8E8M0FNU to f16

    // Perform scaling.
    %3 = puarith.divf %2, %1 : f16

    // Cast to result type.
    %4 = puarith.truncf %3 : f16 to f4E2M1FN
    ```

    Example:

    ```mlir
    // Downcast from f32 to f4E2M1FN.
    %a = puarith.scaling_truncf %b, %c : f32, f8E8M0FNU to f4E2M1FN

    // Element-wise downcast with broadcast (blockSize = 32).
    %f = vector.broadcast %g : vector<1xf8E8M0FNU> to vector<32xf8E8M0FNU>
    %h = puarith.scaling_truncf %i, %f : vector<32xbf16>, vector<32xf8E8M0FNU> to vector<32xf4E2M1FN>
    ```
  }];
  let hasVerifier = 1;
  let assemblyFormat =
      [{ $in `,` $scale ($roundingmode^)? (`fastmath` `` $fastmath^)? attr-dict `:`
      type($in) `,` type($scale) `to` type($out)}];
}

//===----------------------------------------------------------------------===//
// UIToFPOp
//===----------------------------------------------------------------------===//

def PuArith_UIToFPOp :
    PuArith_IToFCastOp<"uitofp", [DeclareOpInterfaceMethods<PuArithNonNegFlagInterface>]> {
  let summary = "cast from unsigned integer type to floating-point";
  let description = [{
    Cast from a value interpreted as unsigned integer to the corresponding
    floating-point value. When operating on vectors, casts elementwise.

    When the `nneg` flag is present, the operand is assumed to have
    the most significant bit set to 0. In this case, zero extension is
    equivalent to sign extension. When this assumption is violated, the
    result is poison.

    Example:

    ```mlir
      // Without nneg flag.
      %0 = puarith.uitofp %a : i32 to f64

      // With nneg flag.
      %1 = puarith.uitofp %a nneg : i32 to f64
    ```
  }];

  let arguments = (ins SignlessFixedWidthIntegerLike:$in, UnitAttr:$nonNeg);
  let results = (outs FloatLike:$out);
  let assemblyFormat = [{
    $in (`nneg` $nonNeg^)? attr-dict `:` type($in) `to` type($out)
  }];
  let hasFolder = 1;
  let hasCanonicalizer = 1;
}

//===----------------------------------------------------------------------===//
// SIToFPOp
//===----------------------------------------------------------------------===//

def PuArith_SIToFPOp : PuArith_IToFCastOp<"sitofp"> {
  let summary = "cast from integer type to floating-point";
  let description = [{
    Cast from a value interpreted as a signed integer to the corresponding
    floating-point value. When operating on vectors, casts elementwise.
  }];
  let hasFolder = 1;
  let hasCanonicalizer = 1;
}

//===----------------------------------------------------------------------===//
// FPToUIOp
//===----------------------------------------------------------------------===//

def PuArith_FPToUIOp : PuArith_FToICastOp<"fptoui"> {
  let summary = "cast from floating-point type to integer type";
  let description = [{
    Cast from a value interpreted as floating-point to the nearest (rounding
    towards zero) unsigned integer value. When operating on vectors, casts
    elementwise.
  }];
  let hasFolder = 1;
}

//===----------------------------------------------------------------------===//
// FPToSIOp
//===----------------------------------------------------------------------===//

def PuArith_FPToSIOp : PuArith_FToICastOp<"fptosi"> {
  let summary = "cast from floating-point type to integer type";
  let description = [{
    Cast from a value interpreted as floating-point to the nearest (rounding
    towards zero) signed integer value. When operating on vectors, casts
    elementwise.
  }];
  let hasFolder = 1;
}

//===----------------------------------------------------------------------===//
// IndexCastOp
//===----------------------------------------------------------------------===//

// Index cast can convert between memrefs of signless integers and indices too.
def IndexCastTypeConstraint : TypeConstraint<Or<[
        PuArith_SignlessIntegerOrIndexLike.predicate,
        MemRefOf<[AnyNonZeroBitwidthSignlessInteger, Index]>.predicate]>,
    "signless-non-zero-bitwidth-integer-like or memref of signless-integer">;

def PuArith_IndexCastOp
  : PuArith_CastOp<"index_cast", IndexCastTypeConstraint, IndexCastTypeConstraint,
                 []> {
  let summary = "cast between index and integer types";
  let description = [{
    Casts between scalar or vector integers and corresponding 'index' scalar or
    vectors. Index is an integer of platform-specific bit width. If casting to
    a wider integer, the value is sign-extended. If casting to a narrower
    integer, the value is truncated.
  }];

  let hasFolder = 1;
  let hasCanonicalizer = 1;
}

//===----------------------------------------------------------------------===//
// IndexCastUIOp
//===----------------------------------------------------------------------===//

def PuArith_IndexCastUIOp
  : PuArith_CastOp<"index_castui", IndexCastTypeConstraint, IndexCastTypeConstraint,
                 [DeclareOpInterfaceMethods<PuArithNonNegFlagInterface>]> {
  let summary = "unsigned cast between index and integer types";
  let description = [{
    Casts between scalar or vector integers and corresponding 'index' scalar or
    vectors. Index is an integer of platform-specific bit width. If casting to
    a wider integer, the value is zero-extended. If casting to a narrower
    integer, the value is truncated.

    When the `nneg` flag is present, the operand is assumed to have
    the most significant bit set to 0. In this case, zero extension
    is equivalent to sign extension. When this assumption is violated,
    the result is poison.

    Example:

    ```mlir
      %0 = puarith.index_castui %a : i32 to index
      %1 = puarith.index_castui %a nneg : i32 to index
      %2 = puarith.index_castui %b nneg : index to i64
    ```
  }];

  let arguments = (ins IndexCastTypeConstraint:$in, UnitAttr:$nonNeg);
  let results = (outs IndexCastTypeConstraint:$out);
  let assemblyFormat = [{
    $in (`nneg` $nonNeg^)? attr-dict `:` type($in) `to` type($out)
  }];
  let hasFolder = 1;
  let hasCanonicalizer = 1;
}

//===----------------------------------------------------------------------===//
// BitcastOp
//===----------------------------------------------------------------------===//

// Bitcast can convert between memrefs of signless integers and floats.
// i0 (zero-bitwidth) integers are excluded: a 0-bit bitcast is meaningless.
def BitcastTypeConstraint : TypeConstraint<Or<[
        SignlessFixedWidthIntegerLike.predicate,
        FloatLike.predicate,
        MemRefOf<[AnyNonZeroBitwidthSignlessInteger, AnyFloat]>.predicate]>,
    "non-zero-bitwidth-signless-integer-or-float-like or memref of non-zero-bitwidth signless integer or float">;

def PuArith_BitcastOp : PuArith_Op<"bitcast",
    [Pure, SameOperandsAndResultShape, SameInputOutputTensorDims,
     DeclareOpInterfaceMethods<CastOpInterface>]>,
    Arguments<(ins BitcastTypeConstraint:$in)>,
    Results<(outs BitcastTypeConstraint:$out)> {
  let assemblyFormat = "$in attr-dict `:` type($in) `to` type($out)";

  let summary = "bitcast between values of equal bit width";
  let description = [{
    Bitcast an integer or floating point value to an integer or floating point
    value of equal bit width. When operating on vectors, casts elementwise.

    Note that this implements a logical bitcast independent of target
    endianness. This allows constant folding without target information and is
    consitent with the bitcast constant folders in LLVM (see
    https://github.com/llvm/llvm-project/blob/18c19414eb/llvm/lib/IR/ConstantFold.cpp#L168)
    For targets where the source and target type have the same endianness (which
    is the standard), this cast will also change no bits at runtime, but it may
    still require an operation, for example if the machine has different
    floating point and integer register files. For targets that have a different
    endianness for the source and target types (e.g. float is big-endian and
    integer is little-endian) a proper lowering would add operations to swap the
    order of words in addition to the bitcast.
  }];

  let hasFolder = 1;
  let hasCanonicalizer = 1;
}

//===----------------------------------------------------------------------===//
// CmpIOp
//===----------------------------------------------------------------------===//

def PuArith_CmpIOp : PuArith_CompareOp<"cmpi",
    []> {
  let summary = "integer comparison operation";
  let description = [{
    The `cmpi` operation is a generic comparison for integer-like types. Its two
    arguments can be integers, vectors or tensors thereof as long as their types
    match. The operation produces an i1 for the former case, a vector or a
    tensor of i1 with the same shape as inputs in the other cases.

    Its first argument is an attribute that defines which type of comparison is
    performed. The following comparisons are supported:

    -   equal (mnemonic: `"eq"`; integer value: `0`)
    -   not equal (mnemonic: `"ne"`; integer value: `1`)
    -   signed less than (mnemonic: `"slt"`; integer value: `2`)
    -   signed less than or equal (mnemonic: `"sle"`; integer value: `3`)
    -   signed greater than (mnemonic: `"sgt"`; integer value: `4`)
    -   signed greater than or equal (mnemonic: `"sge"`; integer value: `5`)
    -   unsigned less than (mnemonic: `"ult"`; integer value: `6`)
    -   unsigned less than or equal (mnemonic: `"ule"`; integer value: `7`)
    -   unsigned greater than (mnemonic: `"ugt"`; integer value: `8`)
    -   unsigned greater than or equal (mnemonic: `"uge"`; integer value: `9`)

    The result is `1` if the comparison is true and `0` otherwise. For vector or
    tensor operands, the comparison is performed elementwise and the element of
    the result indicates whether the comparison is true for the operand elements
    with the same indices as those of the result.

    Note: while the custom assembly form uses strings, the actual underlying
    attribute has integer type (or rather enum class in C++ code) as seen from
    the generic assembly form. String literals are used to improve readability
    of the IR by humans.

    This operation only applies to integer-like operands, but not floats. The
    main reason being that comparison operations have diverging sets of
    attributes: integers require sign specification while floats require various
    floating point-related particularities, e.g., `-ffast-math` behavior,
    IEEE754 compliance, etc
    ([rationale](../Rationale/Rationale.md#splitting-floating-point-vs-integer-operations)).
    The type of comparison is specified as attribute to avoid introducing ten
    similar operations, taking into account that they are often implemented
    using the same operation downstream
    ([rationale](../Rationale/Rationale.md#specifying-comparison-kind-as-attribute)). The
    separation between signed and unsigned order comparisons is necessary
    because of integers being signless. The comparison operation must know how
    to interpret values with the foremost bit being set: negatives in two's
    complement or large positives
    ([rationale](../Rationale/Rationale.md#specifying-sign-in-integer-comparison-operations)).

    Example:

    ```mlir
    %x = puarith.cmpi slt, %lhs, %rhs : i32

    %x = "arith.cmpi"(%lhs, %rhs) {predicate = 2 : i64} : (i32, i32) -> i1

    %x = puarith.cmpi eq, %lhs, %rhs : vector<4xi64>

    %x = "arith.cmpi"(%lhs, %rhs) {predicate = 0 : i64}
        : (vector<4xi64>, vector<4xi64>) -> vector<4xi1>
    ```
  }];

  let arguments = (ins PuArith_CmpIPredicateAttr:$predicate,
                       PuArith_SignlessIntegerOrIndexLike:$lhs,
                       PuArith_SignlessIntegerOrIndexLike:$rhs);

  let skipDefaultBuilders = 1;
  let builders = [
    OpBuilder<(ins "Type":$resultType, "Value":$lhs, "Value":$rhs,
                   "CmpIPredicate":$predicate),
    [{
      $_state.addTypes(resultType);
      $_state.addOperands({lhs, rhs});
      $_state.addAttribute("predicate",
          puarith::CmpIPredicateAttr::get($_builder.getContext(), predicate));
    }]>
  ];

  let extraClassDeclaration = [{
    static CmpIOp create(::mlir::OpBuilder& b, ::mlir::Location loc, ::mlir::TypeRange types,
                         ::mlir::ValueRange operands, Properties& props) {
      ::mlir::OperationState st(loc, getOperationName());
      st.addTypes(types);
      st.addOperands(operands);
      st.getOrAddProperties<Properties>().predicate = props.predicate;
      return ::llvm::cast<CmpIOp>(b.create(st));
    }
    static CmpIOp create(::mlir::OpBuilder& b, ::mlir::Location loc, CmpIPredicateAttr predicate,
                         ::mlir::Value lhs, ::mlir::Value rhs) {
      Properties props;
      props.predicate = predicate;
      return create(b, loc, getI8SameShape(lhs.getType()), ::mlir::ValueRange{lhs, rhs}, props);
    }
  }];

  let hasFolder = 1;
  let hasCanonicalizer = 1;
}

def PuArith_CmpFOp : PuArith_CompareOp<"cmpf",
    [DeclareOpInterfaceMethods<PuArithFastMathInterface>]> {
  let summary = "floating-point comparison operation";
  let description = [{
    The `cmpf` operation compares its two operands according to the float
    comparison rules and the predicate specified by the respective attribute.
    The predicate defines the type of comparison: (un)orderedness, (in)equality
    and signed less/greater than (or equal to) as well as predicates that are
    always true or false.  The operands must have the same type, and this type
    must be a float type, or a vector or tensor thereof.  The result is an i1,
    or a vector/tensor thereof having the same shape as the inputs. Unlike cmpi,
    the operands are always treated as signed. The u prefix indicates
    *unordered* comparison, not unsigned comparison, so "une" means unordered or
    not equal. For the sake of readability by humans, custom assembly form for
    the operation uses a string-typed attribute for the predicate.  The value of
    this attribute corresponds to lower-cased name of the predicate constant,
    e.g., "one" means "ordered not equal".  The string representation of the
    attribute is merely a syntactic sugar and is converted to an integer
    attribute by the parser.

    Example:

    ```mlir
    %r1 = puarith.cmpf oeq, %0, %1 : f32
    %r2 = puarith.cmpf ult, %0, %1 : tensor<42x42xf64>
    %r3 = "arith.cmpf"(%0, %1) {predicate: 0} : (f8, f8) -> i1
    ```
  }];

  let arguments = (ins PuArith_CmpFPredicateAttr:$predicate,
                       FloatLike:$lhs,
                       FloatLike:$rhs,
                       DefaultValuedAttr<
                         PuArith_FastMathAttr, "::mlir::puarith::FastMathFlags::none">:$fastmath);

  let skipDefaultBuilders = 1;
  let builders = [
    OpBuilder<(ins "Type":$resultType, "Value":$lhs, "Value":$rhs,
                   "CmpFPredicate":$predicate),
    [{
      $_state.addTypes(resultType);
      $_state.addOperands({lhs, rhs});
      $_state.addAttribute("predicate",
          puarith::CmpFPredicateAttr::get($_builder.getContext(), predicate));
      $_state.addAttribute("fastmath",
          puarith::FastMathFlagsAttr::get($_builder.getContext(),
              puarith::FastMathFlags::none));
    }]>
  ];

  let hasFolder = 1;
  let hasCanonicalizer = 1;
}

class BooleanConditionOrMatchingShape<string condition, string result> :
    PredOpTrait<
      condition # " is signless i8 or has matching shape",
      Or<[TypeIsPred<condition, I8>,
          And<[SubstLeaves<"$_self", "$" # condition # ".getType()", IsShapedTypePred>,
               SubstLeaves<"$_self", "$" # result # ".getType()", IsShapedTypePred>,
               AllShapesMatch<[condition, result]>.predicate]>]>>;

def SelectOp : PuArith_Op<"select", [Pure,
    AllTypesMatch<["true_value", "false_value", "result"]>,
    BooleanConditionOrMatchingShape<"condition", "result">,
    DeclareOpInterfaceMethods<SelectLikeOpInterface>]> {
  let summary = "select operation";
  let description = [{
    The `puarith.select` operation chooses one value based on a binary condition
    supplied as its first operand.

    If the value of the first operand (the condition) is `1`, then the second
    operand is returned, and the third operand is ignored, even if it was poison.

    If the value of the first operand (the condition) is `0`, then the third
    operand is returned, and the second operand is ignored, even if it was poison.

    If the value of the first operand (the condition) is poison, then the
    operation returns poison.

    The operation applies to vectors and tensors elementwise given the _shape_
    of all operands is identical. The choice is made for each element
    individually based on the value at the same position as the element in the
    condition operand. If an i1 is provided as the condition, the entire vector
    or tensor is chosen.

    Example:

    ```mlir
    %x = puarith.select %cond, %true, %false : i32

    %x = "arith.select"(%cond, %true, %false) : (i1, i32, i32) -> i32

    %vx = puarith.select %vcond, %vtrue, %vfalse : vector<42xi1>, vector<42xf32>

    %vx = puarith.select %cond, %vtrue, %vfalse : vector<42xf32>
    ```
  }];

  let arguments = (ins AnyType:$condition,
                       AnyType:$true_value,
                       AnyType:$false_value);
  let results = (outs AnyType:$result);

  let hasCanonicalizer = 1;
  let hasFolder = 1;
  let hasVerifier = 1;

  let hasCustomAssemblyFormat = 1;
}
