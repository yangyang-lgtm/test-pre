include "mlir/IR/EnumAttr.td"
include "mlir/IR/OpBase.td"

def PuArith_Dialect : Dialect {
  let name = "puarith";
  let cppNamespace = "::mlir::puarith";
  let description = [{
    Full clone of the arith dialect. The single semantic difference is that
    `i1` is replaced by `i8` everywhere: comparison results, select
    conditions and constants are carried as `i8` so that vectorized
    predicates are `vector<Wxi8>` payloads.
  }];

  let hasConstantMaterializer = 1;
  let useDefaultAttributePrinterParser = 1;
}

def PuArith_PredI8Like : TypeConstraint<Or<[
    I8.predicate,
    VectorOfAnyRankOf<[I8]>.predicate,
    TensorOf<[I8]>.predicate]>,
    "signless-i8-or-shaped">;




def PuArith_CmpFPredicateAttr : I64EnumAttr<
    "CmpFPredicate", "",
    [
      I64EnumAttrCase<"AlwaysFalse", 0, "false">,
      I64EnumAttrCase<"OEQ", 1, "oeq">,
      I64EnumAttrCase<"OGT", 2, "ogt">,
      I64EnumAttrCase<"OGE", 3, "oge">,
      I64EnumAttrCase<"OLT", 4, "olt">,
      I64EnumAttrCase<"OLE", 5, "ole">,
      I64EnumAttrCase<"ONE", 6, "one">,
      I64EnumAttrCase<"ORD", 7, "ord">,
      I64EnumAttrCase<"UEQ", 8, "ueq">,
      I64EnumAttrCase<"UGT", 9, "ugt">,
      I64EnumAttrCase<"UGE", 10, "uge">,
      I64EnumAttrCase<"ULT", 11, "ult">,
      I64EnumAttrCase<"ULE", 12, "ule">,
      I64EnumAttrCase<"UNE", 13, "une">,
      I64EnumAttrCase<"UNO", 14, "uno">,
      I64EnumAttrCase<"AlwaysTrue", 15, "true">,
    ]> {
  let cppNamespace = "::mlir::puarith";
}

def PuArith_CmpIPredicateAttr : I64EnumAttr<
    "CmpIPredicate", "",
    [
      I64EnumAttrCase<"eq", 0>,
      I64EnumAttrCase<"ne", 1>,
      I64EnumAttrCase<"slt", 2>,
      I64EnumAttrCase<"sle", 3>,
      I64EnumAttrCase<"sgt", 4>,
      I64EnumAttrCase<"sge", 5>,
      I64EnumAttrCase<"ult", 6>,
      I64EnumAttrCase<"ule", 7>,
      I64EnumAttrCase<"ugt", 8>,
      I64EnumAttrCase<"uge", 9>,
    ]> {
  let cppNamespace = "::mlir::puarith";
}



def FASTMATH_NONE            : I32BitEnumAttrCaseNone<"none"      >;
def FASTMATH_REASSOC         : I32BitEnumAttrCaseBit<"reassoc",  0>;
def FASTMATH_NO_NANS         : I32BitEnumAttrCaseBit<"nnan",     1>;
def FASTMATH_NO_INFS         : I32BitEnumAttrCaseBit<"ninf",     2>;
def FASTMATH_NO_SIGNED_ZEROS : I32BitEnumAttrCaseBit<"nsz",      3>;
def FASTMATH_ALLOW_RECIP     : I32BitEnumAttrCaseBit<"arcp",     4>;
def FASTMATH_ALLOW_CONTRACT  : I32BitEnumAttrCaseBit<"contract", 5>;
def FASTMATH_APPROX_FUNC     : I32BitEnumAttrCaseBit<"afn",      6>;
def FASTMATH_FAST            : I32BitEnumAttrCaseGroup<
    "fast",
    [
      FASTMATH_REASSOC,         FASTMATH_NO_NANS,     FASTMATH_NO_INFS,
      FASTMATH_NO_SIGNED_ZEROS, FASTMATH_ALLOW_RECIP, FASTMATH_ALLOW_CONTRACT,
      FASTMATH_APPROX_FUNC]>;

def FastMathFlags : I32BitEnumAttr<
    "FastMathFlags",
    "Floating point fast math flags",
    [
      FASTMATH_NONE,           FASTMATH_REASSOC,         FASTMATH_NO_NANS,
      FASTMATH_NO_INFS,        FASTMATH_NO_SIGNED_ZEROS, FASTMATH_ALLOW_RECIP,
      FASTMATH_ALLOW_CONTRACT, FASTMATH_APPROX_FUNC,     FASTMATH_FAST]> {
  let separator = ",";
  let cppNamespace = "::mlir::puarith";
  let genSpecializedAttr = 0;
  let printBitEnumPrimaryGroups = 1;
}

def PuArith_FastMathAttr :
    EnumAttr<PuArith_Dialect, FastMathFlags, "fastmath"> {
  let assemblyFormat = "`<` $value `>`";
}

def PuArith_IOFnone : I32BitEnumAttrCaseNone<"none">;
def PuArith_IOFnsw  : I32BitEnumAttrCaseBit<"nsw", 0>;
def PuArith_IOFnuw  : I32BitEnumAttrCaseBit<"nuw", 1>;

def PuArith_IntegerOverflowFlags : I32BitEnumAttr<
    "IntegerOverflowFlags",
    "Integer overflow arith flags",
    [PuArith_IOFnone, PuArith_IOFnsw, PuArith_IOFnuw]> {
  let separator = ", ";
  let cppNamespace = "::mlir::puarith";
  let genSpecializedAttr = 0;
  let printBitEnumPrimaryGroups = 1;
}

def PuArith_IntegerOverflowAttr :
    EnumAttr<PuArith_Dialect, PuArith_IntegerOverflowFlags, "overflow"> {
  let assemblyFormat = "`<` $value `>`";
}

def PuArith_RToNearestTiesToEven
    : I32EnumAttrCase<"to_nearest_even", 0>;
def PuArith_RDownward
    : I32EnumAttrCase<"downward", 1>;
def PuArith_RUpward
    : I32EnumAttrCase<"upward", 2>;
def PuArith_RTowardZero
    : I32EnumAttrCase<"toward_zero", 3>;
def PuArith_RToNearestTiesAwayFromZero
    : I32EnumAttrCase<"to_nearest_away", 4>;

def PuArith_RoundingModeAttr : I32EnumAttr<
    "RoundingMode", "Floating point rounding mode",
    [PuArith_RToNearestTiesToEven, PuArith_RDownward, PuArith_RUpward,
     PuArith_RTowardZero, PuArith_RToNearestTiesAwayFromZero]> {
  let cppNamespace = "::mlir::puarith";
}
