include "privateuse/Dialect/PuArith/IR/PuArithInterfaces.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/VectorInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

def PuMath_Dialect : Dialect {
  let name = "pumath";
  let cppNamespace = "::mlir::pumath";
  let description = [{
    Full clone of the math dialect; classification predicates are carried as
    `i8` (and `vector<Wxi8>`) instead of `i1`.
  }];

  let hasConstantMaterializer = 1;
  let dependentDialects = [
    "::mlir::puarith::PuArithDialect"
  ];
}

class PuMath_Op<string mnemonic, list<Trait> traits = []> :
    Op<PuMath_Dialect, mnemonic, traits # [Pure,
    DeclareOpInterfaceMethods<VectorUnrollOpInterface>] #
    ElementwiseMappable.traits>;



class PuMath_IntegerUnaryOp<string mnemonic, list<Trait> traits = []> :
    PuMath_Op<mnemonic, traits # [SameOperandsAndResultType]> {
  let arguments = (ins SignlessIntegerOrIndexLike:$operand);
  let results = (outs SignlessIntegerOrIndexLike:$result);

  let assemblyFormat = "$operand attr-dict `:` type($result)";
}

class PuMath_FloatClassificationOp<string mnemonic, list<Trait> traits = []> :
    PuMath_Op<mnemonic,
      traits # [DeclareOpInterfaceMethods<PuArithFastMathInterface>,
                TypesMatchWith<
          "result type has i8 element type and same shape as operands",
          "operand", "result", "::mlir::puarith::getI8SameShape($_self)">]> {
  let arguments = (ins FloatLike:$operand,
      DefaultValuedAttr<PuArith_FastMathAttr,
                        "::mlir::puarith::FastMathFlags::none">:$fastmath);
  let results = (outs PuArith_PredI8Like:$result);

  let assemblyFormat = "$operand attr-dict `:` type($operand)";
}

class PuMath_FloatUnaryOp<string mnemonic, list<Trait> traits = []> :
    PuMath_Op<mnemonic,
        traits # [SameOperandsAndResultType,
                  DeclareOpInterfaceMethods<PuArithFastMathInterface>]> {
  let arguments = (ins FloatLike:$operand,
      DefaultValuedAttr<PuArith_FastMathAttr,
                        "::mlir::puarith::FastMathFlags::none">:$fastmath);
  let results = (outs FloatLike:$result);

  let assemblyFormat = [{ $operand (`fastmath` `` $fastmath^)?
                          attr-dict `:` type($result) }];
}

class PuMath_IntegerBinaryOp<string mnemonic, list<Trait> traits = []> :
    PuMath_Op<mnemonic, traits # [SameOperandsAndResultType]> {
  let arguments = (ins SignlessIntegerOrIndexLike:$lhs, SignlessIntegerOrIndexLike:$rhs);
  let results = (outs SignlessIntegerOrIndexLike:$result);

  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` type($result)";
}

class PuMath_FloatBinaryOp<string mnemonic, list<Trait> traits = []> :
    PuMath_Op<mnemonic,
        traits # [SameOperandsAndResultType,
                  DeclareOpInterfaceMethods<PuArithFastMathInterface>]> {
  let arguments = (ins FloatLike:$lhs, FloatLike:$rhs,
      DefaultValuedAttr<PuArith_FastMathAttr,
                        "::mlir::puarith::FastMathFlags::none">:$fastmath);
  let results = (outs FloatLike:$result);

  let assemblyFormat = [{ $lhs `,` $rhs (`fastmath` `` $fastmath^)?
                          attr-dict `:` type($result) }];
}

class PuMath_FloatTernaryOp<string mnemonic, list<Trait> traits = []> :
    PuMath_Op<mnemonic,
        traits # [SameOperandsAndResultType,
                  DeclareOpInterfaceMethods<PuArithFastMathInterface>]> {
  let arguments = (ins FloatLike:$a, FloatLike:$b, FloatLike:$c,
      DefaultValuedAttr<PuArith_FastMathAttr,
                        "::mlir::puarith::FastMathFlags::none">:$fastmath);
  let results = (outs FloatLike:$result);

  let assemblyFormat = [{ $a `,` $b `,` $c (`fastmath` `` $fastmath^)?
                          attr-dict `:` type($result) }];
}

class PuMath_FloatTernaryOpWithRoundingMode<string mnemonic,
                                          list<Trait> traits = []> :
    PuMath_FloatTernaryOp<mnemonic,
      !listconcat([DeclareOpInterfaceMethods<PuArithRoundingModeInterface>],
                  traits)> {
  let arguments = (ins FloatLike:$a, FloatLike:$b, FloatLike:$c,
      DefaultValuedAttr<PuArith_FastMathAttr,
                        "::mlir::puarith::FastMathFlags::none">:$fastmath,
      OptionalAttr<PuArith_RoundingModeAttr>:$roundingmode);
  let builders = [
    OpBuilder<(ins "Value":$a, "Value":$b, "Value":$c,
      CArg<"::mlir::puarith::FastMathFlags",
           "::mlir::puarith::FastMathFlags::none">:$fastmath), [{
      build($_builder, $_state, a, b, c, fastmath,
            ::mlir::puarith::RoundingModeAttr{});
    }]>,
    OpBuilder<(ins "Value":$a, "Value":$b, "Value":$c,
      "::mlir::puarith::FastMathFlagsAttr":$fastmath), [{
      build($_builder, $_state, a, b, c, fastmath,
            ::mlir::puarith::RoundingModeAttr{});
    }]>,
    OpBuilder<(ins "Type":$type, "Value":$a, "Value":$b, "Value":$c,
      CArg<"::mlir::puarith::FastMathFlags",
           "::mlir::puarith::FastMathFlags::none">:$fastmath), [{
      build($_builder, $_state, type, a, b, c,
            ::mlir::puarith::FastMathFlagsAttr::get(
                $_builder.getContext(), fastmath),
            ::mlir::puarith::RoundingModeAttr{});
    }]>,
  ];
  let assemblyFormat = [{ $a `,` $b `,` $c ($roundingmode^)?
                          (`fastmath` `` $fastmath^)?
                          attr-dict `:` type($result) }];
}

def PuMath_AbsFOp : PuMath_FloatUnaryOp<"absf"> {
  let summary = "floating point absolute-value operation";
  let description = [{
    The `absf` operation computes the absolute value. It takes one operand of
    floating point type (i.e., scalar, tensor or vector) and returns one result
    of the same type.

    Example:

    ```mlir
    %a = pumath.absf %b : f64
    ```
  }];
  let hasFolder = 1;
}

def PuMath_AbsIOp : PuMath_IntegerUnaryOp<"absi"> {
  let summary = "integer absolute-value operation";
  let description = [{
    The `absi` operation computes the absolute value. It takes one operand of
    integer type (i.e., scalar, tensor or vector) and returns one result of the
    same type.

    Example:

    ```mlir
    %a = pumath.absi %b : i64
    ```
  }];
  let hasFolder = 1;
}

def PuMath_AcoshOp : PuMath_FloatUnaryOp<"acosh">{
  let summary = "Hyperbolic arcus cosine of the given value";
  let description = [{
    Syntax:

    ```
    operation ::= ssa-id `=` `math.acosh` ssa-use `:` type
    ```

    The `acosh` operation computes the arcus cosine of a given value.  It takes
    one operand of floating point type (i.e., scalar, tensor or vector) and returns
    one result of the same type. It has no standard attributes.

    Example:

    ```mlir
    %a = pumath.acosh %b : f64
    ```
  }];
  let hasFolder = 1;
}

def PuMath_AsinOp : PuMath_FloatUnaryOp<"asin">{
  let summary = "arcus sine of the given value";
  let description = [{
    Syntax:

    ```
    operation ::= ssa-id `=` `math.asin` ssa-use `:` type
    ```

    The `asin` operation computes the arcus sine of a given value.  It takes
    one operand of floating point type (i.e., scalar, tensor or vector) and returns
    one result of the same type. It has no standard attributes.

    Example:

    ```mlir
    %a = pumath.asin %b : f64
    ```
  }];
  let hasFolder = 1;
}

def PuMath_AsinhOp : PuMath_FloatUnaryOp<"asinh">{
  let summary = "hyperbolic arcus sine of the given value";
  let description = [{
    Syntax:

    ```
    operation ::= ssa-id `=` `math.asinh` ssa-use `:` type
    ```

    The `asinh` operation computes the hyperbolic arcus sine of a given value.  It takes
    one operand of floating point type (i.e., scalar, tensor or vector) and returns
    one result of the same type. It has no standard attributes.

    Example:

    ```mlir
    %a = pumath.asinh %b : f64
    ```
  }];
  let hasFolder = 1;
}

def PuMath_AtanOp : PuMath_FloatUnaryOp<"atan">{
  let summary = "arcus tangent of the given value";
  let description = [{
    The `atan` operation computes the arcus tangent of a given value.  It takes
    one operand of floating point type (i.e., scalar, tensor or vector) and returns
    one result of the same type. It has no standard attributes.

    Example:

    ```mlir
    %a = pumath.atan %b : f64
    ```
  }];
  let hasFolder = 1;
}

def PuMath_AtanhOp : PuMath_FloatUnaryOp<"atanh">{
  let summary = "hyperbolic arcus tangent of the given value";
  let description = [{
    Syntax:

    ```
    operation ::= ssa-id `=` `math.atanh` ssa-use `:` type
    ```

    The `atanh` operation computes the hyperbolic arcus tangent of a given value.  It takes
    one operand of floating point type (i.e., scalar, tensor or vector) and returns
    one result of the same type. It has no standard attributes.

    Example:

    ```mlir
    %a = pumath.atanh %b : f64
    ```
  }];
  let hasFolder = 1;
}

def PuMath_Atan2Op : PuMath_FloatBinaryOp<"atan2">{
  let summary = "2-argument arcus tangent of the given values";
  let description = [{
    The `atan2` operation takes two operands and returns one result, all of
    which must be of the same type.  The operands must be of floating point type
    (i.e., scalar, tensor or vector).

    The 2-argument arcus tangent `atan2(y, x)` returns the angle in the
    Euclidian plane between the positive x-axis and the ray through the point
    (x, y).  It is a generalization of the 1-argument arcus tangent which
    returns the angle on the basis of the ratio y/x.

    See also https:

    Example:

    ```mlir
    %a = pumath.atan2 %b, %c : f32
    ```
  }];
  let hasFolder = 1;
}

def PuMath_CbrtOp : PuMath_FloatUnaryOp<"cbrt"> {
  let summary = "cube root of the specified value";
  let description = [{
    The `cbrt` operation computes the cube root. It takes one operand of
    floating point type (i.e., scalar, tensor or vector) and returns one result
    of the same type. It has no standard attributes.

    Example:

    ```mlir
    %a = pumath.cbrt %b : f64
    ```

    Note: This op is not equivalent to powf(..., 1/3.0).
  }];

  let hasFolder = 1;
}

def PuMath_CeilOp : PuMath_FloatUnaryOp<"ceil"> {
  let summary = "ceiling of the specified value";
  let description = [{
    The `ceil` operation computes the ceiling of a given value. It takes one
    operand of floating point type (i.e., scalar, tensor or vector) and returns one
    result of the same type.  It has no standard attributes.

    Example:

    ```mlir
    %a = pumath.ceil %b : f64
    ```
  }];
  let hasFolder = 1;
}

def PuMath_ClampFOp : PuMath_FloatTernaryOp<"clampf"> {
  let summary = "floating point clamping operation";
  let description = [{
    The `clampf` operation takes three operands and returns one result, each of
    these is required to be the same type. Operands must be of floating point type
    (i.e., scalar, tensor or vector).

    The semantics of the operation are described by:
    ```
      clampf(value, min, max) = maxf(minf(value, max), min)
    ```
    If `min > max` the resulting value is poison.

    Example:

    ```mlir
    %d = pumath.clampf %value to [%min, %max] : f64
    ```
  }];
  let arguments = (ins FloatLike:$value, FloatLike:$min, FloatLike:$max,
      DefaultValuedAttr<PuArith_FastMathAttr,
                        "::mlir::puarith::FastMathFlags::none">:$fastmath);
  let assemblyFormat = [{
    $value `to` ` ` `[` $min `,` $max `]` (`fastmath` `` $fastmath^)?
    attr-dict `:` type($result)
  }];
}

def PuMath_CopySignOp : PuMath_FloatBinaryOp<"copysign"> {
  let summary = "A copysign operation";
  let description = [{
    The `copysign` returns a value with the magnitude of the first operand and
    the sign of the second operand. It takes two operands and returns one result of
    the same type. The operands must be of floating point type (i.e., scalar,
    tensor or vector). It has no standard attributes.

    Example:

    ```mlir
    %a = pumath.copysign %b, %c : f64
    ```
  }];
  let hasFolder = 1;
}

def PuMath_CosOp : PuMath_FloatUnaryOp<"cos"> {
  let summary = "cosine of the specified value";
  let description = [{
    The `cos` operation computes the cosine of a given value. It takes one
    operand of floating point type (i.e., scalar, tensor or vector) and returns one
    result of the same type.  It has no standard attributes.

    Example:

    ```mlir
    %a = pumath.cos %b : f64
    ```
  }];
  let hasFolder = 1;
}

def PuMath_AcosOp : PuMath_FloatUnaryOp<"acos"> {
  let summary = "arcus cosine of the specified value";
  let description = [{
    The `acos` operation computes the arcus cosine of a given value. It takes one
    operand of floating point type (i.e., scalar, tensor or vector) and returns one
    result of the same type.  It has no standard attributes.

    Example:

    ```mlir
    %a = pumath.acos %b : f64
    ```
  }];
  let hasFolder = 1;
}

def PuMath_CoshOp : PuMath_FloatUnaryOp<"cosh"> {
  let summary = "hyperbolic cosine of the specified value";
  let description = [{
    The `cosh` operation computes the hyperbolic cosine. It takes one operand
    of floating point type (i.e., scalar, tensor or vector) and returns one
    result of the same type. It has no standard attributes.

    Example:

    ```mlir
    %a = pumath.cosh %b : f64
    ```
  }];
  let hasFolder = 1;
}

def PuMath_SinOp : PuMath_FloatUnaryOp<"sin"> {
  let summary = "sine of the specified value";
  let description = [{
    The `sin` operation computes the sine of a given value. It takes one
    operand of floating point type (i.e., scalar, tensor or vector) and returns one
    result of the same type.  It has no standard attributes.

    Example:

    ```mlir
    %a = pumath.sin %b : f64
    ```
  }];
  let hasFolder = 1;
}

def PuMath_SinhOp : PuMath_FloatUnaryOp<"sinh"> {
  let summary = "hyperbolic sine of the specified value";
  let description = [{
    The `sinh` operation computes the hyperbolic sine. It takes one operand
    of floating point type (i.e., scalar, tensor or vector) and returns one
    result of the same type. It has no standard attributes.

    Example:

    ```mlir
    %a = pumath.sinh %b : f64
    ```
  }];
  let hasFolder = 1;
}

def PuMath_SincosOp : PuMath_Op<"sincos",
    [SameOperandsAndResultType,
     DeclareOpInterfaceMethods<PuArithFastMathInterface>]> {
  let summary = "sine and cosine of the specified value";
  let description = [{
    The `sincos` operation computes both the sine and cosine of a given value
    simultaneously. It takes one operand of floating point type (i.e., scalar,
    tensor or vector) and returns two results of the same type. This operation
    can be more efficient than computing sine and cosine separately when both
    values are needed.

    Example:

    ```mlir
    %sin, %cos = pumath.sincos %input : f64
    ```
  }];

  let arguments = (ins FloatLike:$operand,
      DefaultValuedAttr<PuArith_FastMathAttr,
                        "::mlir::puarith::FastMathFlags::none">:$fastmath);
  let results = (outs FloatLike:$sin, FloatLike:$cos);

  let assemblyFormat = [{ $operand (`fastmath` `` $fastmath^)?
                          attr-dict `:` type($operand) }];

  let extraClassDeclaration = [{
    std::optional<SmallVector<int64_t, 4>> getShapeForUnroll();
  }];

  let hasFolder = 1;
}

def PuMath_CountLeadingZerosOp : PuMath_IntegerUnaryOp<"ctlz"> {
  let summary = "counts the leading zeros an integer value";
  let description = [{
    The `ctlz` operation computes the number of leading zeros of an integer value.
    It operates on scalar, tensor or vector.

    Example:

    ```mlir
    %a = pumath.ctlz %b : i32
    ```
  }];
  let hasFolder = 1;
}

def PuMath_CountTrailingZerosOp : PuMath_IntegerUnaryOp<"cttz"> {
  let summary = "counts the trailing zeros an integer value";
  let description = [{
    The `cttz` operation computes the number of trailing zeros of an integer value.
    It operates on scalar, tensor or vector.

    Example:

    ```mlir
    %a = pumath.cttz %b : i32
    ```
  }];
  let hasFolder = 1;
}

def PuMath_CtPopOp : PuMath_IntegerUnaryOp<"ctpop"> {
  let summary = "counts the number of set bits of an integer value";
  let description = [{
    The `ctpop` operation computes the number of set bits of an integer value.
    It operates on scalar, tensor or vector.

    Example:

    ```mlir
    %a = pumath.ctpop %b : i32
    ```
  }];
  let hasFolder = 1;
}

def PuMath_ErfOp : PuMath_FloatUnaryOp<"erf"> {
  let summary = "error function of the specified value";
  let description = [{
    The `erf` operation computes the error function. It takes one operand of
    floating point type (i.e., scalar, tensor or vector) and returns one result of
    the same type. It has no standard attributes.

    Example:

    ```mlir
    %a = pumath.erf %b : f64
    ```
  }];
  let hasFolder = 1;
}

def PuMath_ErfcOp : PuMath_FloatUnaryOp<"erfc"> {
  let summary = "complementary error function of the specified value";
  let description = [{

    The `erfc` operation computes the complementary error function, defined as
    1-erf(x). This function is part of libm and is needed for accuracy, since
    simply calculating 1-erf(x) when x is close to 1 will give inaccurate results.
    It takes one operand of floating point type (i.e., scalar,
    tensor or vector) and returns one result of the same type. It has no
    standard attributes.

    Example:

    ```mlir
    %a = pumath.erfc %b : f64
    ```
  }];
  let hasFolder = 1;
}


def PuMath_ExpOp : PuMath_FloatUnaryOp<"exp"> {
  let summary = "base-e exponential of the specified value";
  let description = [{
    The `exp` operation takes one operand of floating point type (i.e., scalar,
    tensor or vector) and returns one result of the same type. It has no standard
    attributes.

    Example:

    ```mlir
    %a = pumath.exp %b : f64
    ```
  }];
  let hasFolder = 1;
}

def PuMath_Exp2Op : PuMath_FloatUnaryOp<"exp2"> {
  let summary = "base-2 exponential of the specified value";

  let description = [{
    The `exp` operation takes one operand of floating point type (i.e., scalar,
    tensor or vector) and returns one result of the same type. It has no standard
    attributes.

    Example:

    ```mlir
    %a = pumath.exp2 %b : f64
    ```
  }];
  let hasFolder = 1;
}

def PuMath_ExpM1Op : PuMath_FloatUnaryOp<"expm1"> {
  let summary = "base-e exponential of the specified value minus 1";
  let description = [{
    expm1(x) := exp(x) - 1

    The `expm1` operation takes one operand of floating point type (i.e.,
    scalar, tensor or vector) and returns one result of the same type. It has no
    standard attributes.

    Example:

    ```mlir
    %a = pumath.expm1 %b : f64
    ```
  }];
  let hasFolder = 1;
}

def PuMath_FloorOp : PuMath_FloatUnaryOp<"floor"> {
  let summary = "floor of the specified value";
  let description = [{
    The `floor` operation computes the floor of a given value. It takes one
    operand of floating point type (i.e., scalar, tensor or vector) and returns one
    result of the same type.  It has no standard attributes.

    Example:

    ```mlir
    %a = pumath.floor %b : f64
    ```
  }];

  let hasFolder = 1;
}

def PuMath_FmaOp : PuMath_FloatTernaryOpWithRoundingMode<"fma"> {
  let summary = "floating point fused multipy-add operation";
  let description = [{
    The `fma` operation takes three operands and returns one result, each of
    these is required to be the same type. Operands must be of floating point type
    (i.e., scalar, tensor or vector).

    Example:

    ```mlir
    %d = pumath.fma %a, %b, %c : f64

    %e = pumath.fma %a, %b, %c to_nearest_even : f64
    ```

    The semantics of the operation correspond to those of the `llvm.fma`
    [intrinsic](https:
    no rounding mode is set, lowering to LLVM is guaranteed to produce the
    `llvm.fma.*` intrinsic. When a rounding mode is set, the LLVM lowering
    instead produces `llvm.experimental.constrained.fma`.
  }];
}

def PuMath_IPowIOp : PuMath_IntegerBinaryOp<"ipowi"> {
  let summary = "signed integer raised to the power of operation";
  let description = [{
    The `ipowi` operation takes two operands of integer type (i.e., scalar,
    tensor or vector) and returns one result of the same type. Operands
    must have the same type.

    Example:

    ```mlir
    %a = pumath.ipowi %b, %c : i32
    ```
  }];
  let hasFolder = 1;
}

def PuMath_IsFiniteOp : PuMath_FloatClassificationOp<"isfinite"> {
  let summary = "returns true if the operand classifies as finite";
  let description = [{
    Determines if the given floating-point number has finite value i.e. it
    is normal, subnormal or zero, but not infinite or NaN.

    Example:

    ```mlir
    %f = pumath.isfinite %a : f32
    ```
  }];
  let hasFolder = 1;
}

def PuMath_IsInfOp : PuMath_FloatClassificationOp<"isinf"> {
  let summary = "returns true if the operand classifies as infinite";
  let description = [{
    Determines if the given floating-point number is positive or negative
    infinity.

    Example:

    ```mlir
    %f = pumath.isinf %a : f32
    ```
  }];
  let hasFolder = 1;
}

def PuMath_IsNaNOp : PuMath_FloatClassificationOp<"isnan"> {
  let summary = "returns true if the operand classifies as NaN";
  let description = [{
    Determines if the given floating-point number is a not-a-number (NaN)
    value.

    Example:

    ```mlir
    %f = pumath.isnan %a : f32
    ```
  }];
  let hasFolder = 1;
}


def PuMath_IsNormalOp : PuMath_FloatClassificationOp<"isnormal"> {
  let summary = "returns true if the operand classifies as normal";
  let description = [{
    Determines if the given floating-point number is normal, i.e. is neither
    zero, subnormal, infinite, nor NaN.

    Example:

    ```mlir
    %f = pumath.isnormal %a : f32
    ```
  }];
  let hasFolder = 1;
}

def PuMath_LogOp : PuMath_FloatUnaryOp<"log"> {
  let summary = "base-e logarithm of the specified value";

  let description = [{
    Computes the base-e logarithm of the given value. It takes one operand of
    floating point type (i.e., scalar, tensor or vector) and returns one result of
    the same type.

    Example:

    ```mlir
    %y = pumath.log %x : f64
    ```
  }];
  let hasFolder = 1;
}

def PuMath_Log10Op : PuMath_FloatUnaryOp<"log10"> {
  let summary = "base-10 logarithm of the specified value";

  let description = [{
    Computes the base-10 logarithm of the given value. It takes one operand of
    floating point type (i.e., scalar, tensor or vector) and returns one result of
    the same type.

    Example:

    ```mlir
    %y = pumath.log10 %x : f64
    ```
  }];
  let hasFolder = 1;
}

def PuMath_Log1pOp : PuMath_FloatUnaryOp<"log1p"> {
  let summary = "Computes the natural logarithm of one plus the given value";

  let description = [{
    Computes the base-e logarithm of one plus the given value. It takes one
    operand of floating point type (i.e., scalar, tensor or vector) and returns one
    result of the same type.

    log1p(x) := log(1 + x)

    Example:

    ```mlir
    %y = pumath.log1p %x : f64
    ```
  }];
  let hasFolder = 1;
}

def PuMath_Log2Op : PuMath_FloatUnaryOp<"log2"> {
  let summary = "base-2 logarithm of the specified value";

  let description = [{
    Computes the base-2 logarithm of the given value. It takes one operand of
    floating point type (i.e., scalar, tensor or vector) and returns one result of
    the same type.

    Example:

    ```mlir
    %y = pumath.log2 %x : f64
    ```
  }];
  let hasFolder = 1;
}

def PuMath_PowFOp : PuMath_FloatBinaryOp<"powf"> {
  let summary = "floating point raised to the power of operation";
  let description = [{
    The `powf` operation takes two operands of floating point type (i.e.,
    scalar, tensor or vector) and returns one result of the same type. Operands
    must have the same type.

    Example:

    ```mlir
    %a = pumath.powf %b, %c : f64
    ```
  }];
  let hasFolder = 1;
}

def PuMath_RsqrtOp : PuMath_FloatUnaryOp<"rsqrt"> {
  let summary = "reciprocal of sqrt (1 / sqrt of the specified value)";
  let description = [{
    The `rsqrt` operation computes the reciprocal of the square root. It takes
    one operand of floating point type (i.e., scalar, tensor or vector) and returns
    one result of the same type. It has no standard attributes.

    Example:

    ```mlir
    %a = pumath.rsqrt %b : f64
    ```
  }];
  let hasFolder = 1;
}

def PuMath_SqrtOp : PuMath_FloatUnaryOp<"sqrt"> {
  let summary = "sqrt of the specified value";
  let description = [{
    The `sqrt` operation computes the square root. It takes one operand of
    floating point type (i.e., scalar, tensor or vector) and returns one result of
    the same type. It has no standard attributes.

    Example:

    ```mlir
    %a = pumath.sqrt %b : f64
    ```
  }];
  let hasFolder = 1;
}

def PuMath_TanOp : PuMath_FloatUnaryOp<"tan"> {
  let summary = "tangent of the specified value";
  let description = [{
    The `tan` operation computes the tangent. It takes one operand
    of floating point type (i.e., scalar, tensor or vector) and returns one
    result of the same type. It has no standard attributes.

    Example:

    ```mlir
    %a = pumath.tan %b : f64
    ```
  }];
  let hasFolder = 1;
}

def PuMath_TanhOp : PuMath_FloatUnaryOp<"tanh"> {
  let summary = "hyperbolic tangent of the specified value";
  let description = [{
    The `tanh` operation computes the hyperbolic tangent. It takes one operand
    of floating point type (i.e., scalar, tensor or vector) and returns one
    result of the same type. It has no standard attributes.

    Example:

    ```mlir
    %a = pumath.tanh %b : f64
    ```
  }];
  let hasFolder = 1;
}

def PuMath_RoundEvenOp : PuMath_FloatUnaryOp<"roundeven"> {
  let summary = "round of the specified value with halfway cases to even";
  let description = [{
    The `roundeven` operation returns the operand rounded to the nearest integer
    value in floating-point format. It takes one operand of floating point type
    (i.e., scalar, tensor or vector) and produces one result of the same type.  The
    operation rounds the argument to the nearest integer value in floating-point
    format, rounding halfway cases to even, regardless of the current
    rounding direction.

    Example:

    ```mlir
    %a = pumath.roundeven %b : f64
    ```
  }];
  let hasFolder = 1;
}

def PuMath_RoundOp : PuMath_FloatUnaryOp<"round"> {
  let summary = "round of the specified value";
  let description = [{
    The `round` operation returns the operand rounded to the nearest integer
    value in floating-point format. It takes one operand of floating point type
    (i.e., scalar, tensor or vector) and produces one result of the same type.  The
    operation rounds the argument to the nearest integer value in floating-point
    format, rounding halfway cases away from zero, regardless of the current
    rounding direction.

    Example:

    ```mlir
    %a = pumath.round %b : f64
    ```
  }];
  let hasFolder = 1;
}

def PuMath_TruncOp : PuMath_FloatUnaryOp<"trunc"> {
  let summary = "trunc of the specified value";
  let description = [{
    The `trunc` operation returns the operand rounded to the nearest integer
    value in floating-point format. It takes one operand of floating point type
    (i.e., scalar, tensor or vector) and produces one result of the same type.
    The operation always rounds to the nearest integer not larger in magnitude
    than the operand, regardless of the current rounding direction.

    Example:

    ```mlir
    %a = pumath.trunc %b : f64
    ```
  }];
  let hasFolder = 1;
}

def PuMath_FPowIOp : PuMath_Op<"fpowi",
    [SameOperandsAndResultShape, AllTypesMatch<["lhs", "result"]>,
     DeclareOpInterfaceMethods<PuArithFastMathInterface>]> {
  let summary = "floating point raised to the signed integer power";
  let description = [{
    The `fpowi` operation takes a `base` operand of floating point type
    (i.e. scalar, tensor or vector) and a `power` operand of integer type
    (also scalar, tensor or vector) and returns one result of the same type
    as `base`. The result is `base` raised to the power of `power`.
    The operation is elementwise for non-scalars, e.g.:

    ```mlir
    %v = pumath.fpowi %base, %power : vector<2xf32>, vector<2xi32>
    ```

    The result is a vector of:

    ```
    [<math.fpowi %base[0], %power[0]>, <math.fpowi %base[1], %power[1]>]
    ```

    Example:

    ```mlir
    %a = pumath.fpowi %base, %power : f64, i32
    ```
  }];

  let arguments = (ins FloatLike:$lhs, SignlessIntegerOrIndexLike:$rhs,
      DefaultValuedAttr<PuArith_FastMathAttr,
                        "::mlir::puarith::FastMathFlags::none">:$fastmath);
  let results = (outs FloatLike:$result);
  let assemblyFormat = [{ $lhs `,` $rhs (`fastmath` `` $fastmath^)?
                          attr-dict `:` type($lhs) `,` type($rhs) }];

  let hasFolder = 1;
}
