#ifndef ARITH_PATTERNS
#define ARITH_PATTERNS

include "mlir/IR/PatternBase.td"
include "privateuse/Dialect/PuArith/IR/PuArithDialect.td"

def GetZeroAttr : NativeCodeCall<"$_builder.getZeroAttr($0.getType())">;

def IsScalarOrHasStaticShape :
    Constraint<CPred<"!::llvm::isa<::mlir::ShapedType>($0.getType()) || "
                     "::llvm::cast<::mlir::ShapedType>($0.getType())"
                     ".hasStaticShape()">,
               "type is a scalar or has a static shape">;

def AddIntAttrs : NativeCodeCall<"addIntegerAttrs($_builder, $0, $1, $2)">;

def SubIntAttrs : NativeCodeCall<"subIntegerAttrs($_builder, $0, $1, $2)">;

def MulIntAttrs : NativeCodeCall<"mulIntegerAttrs($_builder, $0, $1, $2)">;

def AndIntAttrs : NativeCodeCall<"andIntegerAttrs($_builder, $0, $1, $2)">;

def OrIntAttrs : NativeCodeCall<"orIntegerAttrs($_builder, $0, $1, $2)">;

def XorIntAttrs : NativeCodeCall<"xorIntegerAttrs($_builder, $0, $1, $2)">;

def MergeOverflow : NativeCodeCall<"mergeOverflowFlags($0, $1)">;

defvar DefOverflow = ConstantEnumCase<PuArith_IntegerOverflowAttr, "none">;

class cast<string type> : NativeCodeCall<"::mlir::cast<" # type # ">($0)">;

def TypesMatch : Constraint<CPred<"$0.getType() == $1.getType()">>;

def IntAttrTypeMatchesResult :
    Constraint<CPred<"$0.getType() == cast<IntegerAttr>($1).getType()">>;

def IsEqualityCmpIPredicate :
    Constraint<CPred<"$0.getValue() == puarith::CmpIPredicate::eq || "
                     "$0.getValue() == puarith::CmpIPredicate::ne">>;

def AddIAddConstant :
    Pat<(PuArith_AddIOp:$res
          (PuArith_AddIOp $x, (ConstantLikeMatcher APIntAttr:$c0), $ovf1),
          (ConstantLikeMatcher APIntAttr:$c1), $ovf2),
        (PuArith_AddIOp $x, (PuArith_ConstantOp (AddIntAttrs $res, $c0, $c1)),
            (MergeOverflow $ovf1, $ovf2))>;

def AddISubConstantRHS :
    Pat<(PuArith_AddIOp:$res
          (PuArith_SubIOp $x, (ConstantLikeMatcher APIntAttr:$c0), $ovf1),
          (ConstantLikeMatcher APIntAttr:$c1), $ovf2),
        (PuArith_AddIOp $x, (PuArith_ConstantOp (SubIntAttrs $res, $c1, $c0)),
            (MergeOverflow $ovf1, $ovf2))>;

def AddISubConstantLHS :
    Pat<(PuArith_AddIOp:$res
          (PuArith_SubIOp (ConstantLikeMatcher APIntAttr:$c0), $x, $ovf1),
          (ConstantLikeMatcher APIntAttr:$c1), $ovf2),
        (PuArith_SubIOp (PuArith_ConstantOp (AddIntAttrs $res, $c0, $c1)), $x,
            (MergeOverflow $ovf1, $ovf2))>;

def IsScalarOrSplatNegativeOne :
    Constraint<And<[
      CPred<"succeeded(getIntOrSplatIntValue($0))">,
      CPred<"getIntOrSplatIntValue($0)->isAllOnes()">]>>;

def AddIMulNegativeOneRhs :
    Pat<(PuArith_AddIOp
           $x,
           (PuArith_MulIOp $y, (ConstantLikeMatcher AnyAttr:$c0), $ovf1), $ovf2),
        (PuArith_SubIOp $x, $y, DefOverflow),
        [(IsScalarOrSplatNegativeOne $c0)]>;

def AddIMulNegativeOneLhs :
    Pat<(PuArith_AddIOp
           (PuArith_MulIOp $x, (ConstantLikeMatcher AnyAttr:$c0), $ovf1),
           $y, $ovf2),
        (PuArith_SubIOp $y, $x, DefOverflow),
        [(IsScalarOrSplatNegativeOne $c0)]>;

def MulIMulIConstant :
    Pat<(PuArith_MulIOp:$res
          (PuArith_MulIOp $x, (ConstantLikeMatcher APIntAttr:$c0), $ovf1),
          (ConstantLikeMatcher APIntAttr:$c1), $ovf2),
        (PuArith_MulIOp $x, (PuArith_ConstantOp (MulIntAttrs $res, $c0, $c1)),
            (MergeOverflow $ovf1, $ovf2)),
             [(IntAttrTypeMatchesResult $res, $c0),
              (IntAttrTypeMatchesResult $res, $c1)]>;

def AddUIExtendedToAddI:
    Pattern<(PuArith_AddUIExtendedOp:$res $x, $y),
             [(PuArith_AddIOp $x, $y, DefOverflow), (replaceWithValue $x)],
             [(Constraint<CPred<"$0.getUses().empty()">> $res__1)]>;

def SubUIExtendedToSubI:
    Pattern<(PuArith_SubUIExtendedOp:$res $x, $y),
             [(PuArith_SubIOp $x, $y, DefOverflow), (replaceWithValue $x)],
             [(Constraint<CPred<"$0.getUses().empty()">> $res__1)]>;

def SubIRHSAddConstant :
    Pat<(PuArith_SubIOp:$res
          (PuArith_AddIOp $x, (ConstantLikeMatcher APIntAttr:$c0), $ovf1),
          (ConstantLikeMatcher APIntAttr:$c1), $ovf2),
        (PuArith_AddIOp $x, (PuArith_ConstantOp (SubIntAttrs $res, $c0, $c1)),
            DefOverflow)>;

def SubILHSAddConstant :
    Pat<(PuArith_SubIOp:$res
          (ConstantLikeMatcher APIntAttr:$c1),
          (PuArith_AddIOp $x, (ConstantLikeMatcher APIntAttr:$c0), $ovf1), $ovf2),
        (PuArith_SubIOp (PuArith_ConstantOp (SubIntAttrs $res, $c1, $c0)), $x,
            (MergeOverflow $ovf1, $ovf2))>;

def SubIRHSSubConstantRHS :
    Pat<(PuArith_SubIOp:$res
          (PuArith_SubIOp $x, (ConstantLikeMatcher APIntAttr:$c0), $ovf1),
          (ConstantLikeMatcher APIntAttr:$c1), $ovf2),
        (PuArith_SubIOp $x, (PuArith_ConstantOp (AddIntAttrs $res, $c0, $c1)),
            (MergeOverflow $ovf1, $ovf2))>;

def SubIRHSSubConstantLHS :
    Pat<(PuArith_SubIOp:$res
          (PuArith_SubIOp (ConstantLikeMatcher APIntAttr:$c0), $x, $ovf1),
          (ConstantLikeMatcher APIntAttr:$c1), $ovf2),
        (PuArith_SubIOp (PuArith_ConstantOp (SubIntAttrs $res, $c0, $c1)), $x,
            (MergeOverflow $ovf1, $ovf2))>;

def SubILHSSubConstantRHS :
    Pat<(PuArith_SubIOp:$res
          (ConstantLikeMatcher APIntAttr:$c1),
          (PuArith_SubIOp $x, (ConstantLikeMatcher APIntAttr:$c0), $ovf1), $ovf2),
        (PuArith_SubIOp (PuArith_ConstantOp (AddIntAttrs $res, $c0, $c1)), $x,
            (MergeOverflow $ovf1, $ovf2))>;

def SubILHSSubConstantLHS :
    Pat<(PuArith_SubIOp:$res
          (ConstantLikeMatcher APIntAttr:$c1),
          (PuArith_SubIOp (ConstantLikeMatcher APIntAttr:$c0), $x, $ovf1), $ovf2),
        (PuArith_AddIOp $x, (PuArith_ConstantOp (SubIntAttrs $res, $c1, $c0)),
            (MergeOverflow $ovf1, $ovf2))>;

def SubISubILHSRHSLHS :
    Pat<(PuArith_SubIOp:$res (PuArith_SubIOp $x, $y, $ovf1), $x, $ovf2),
        (PuArith_SubIOp (PuArith_ConstantOp (GetZeroAttr $y)), $y,
            (MergeOverflow $ovf1, $ovf2)),
        [(IsScalarOrHasStaticShape $y)]>;

def MulSIExtendedToMulI :
    Pattern<(PuArith_MulSIExtendedOp:$res $x, $y),
        [(PuArith_MulIOp $x, $y, DefOverflow), (replaceWithValue $x)],
        [(Constraint<CPred<"$0.getUses().empty()">> $res__1)]>;


def IsScalarOrSplatOne :
    Constraint<And<[
      CPred<"succeeded(getIntOrSplatIntValue($0))">,
      CPred<"getIntOrSplatIntValue($0)->isStrictlyPositive()">,
      CPred<"*getIntOrSplatIntValue($0) == 1">]>>;

def MulSIExtendedRHSOne :
    Pattern<(PuArith_MulSIExtendedOp $x, (ConstantLikeMatcher AnyAttr:$c1)),
            [(replaceWithValue $x),
             (PuArith_ExtSIOp(PuArith_CmpIOp
                ConstantEnumCase<PuArith_CmpIPredicateAttr, "slt">,
                $x,
                (PuArith_ConstantOp (GetZeroAttr $x))))],
            [(IsScalarOrSplatOne $c1)]>;

def MulUIExtendedToMulI :
    Pattern<(PuArith_MulUIExtendedOp:$res $x, $y),
        [(PuArith_MulIOp $x, $y, DefOverflow), (replaceWithValue $x)],
        [(Constraint<CPred<"$0.getUses().empty()">> $res__1)]>;

def XOrIXOrIConstant :
    Pat<(PuArith_XOrIOp:$res
          (PuArith_XOrIOp $x, (ConstantLikeMatcher APIntAttr:$c0)),
          (ConstantLikeMatcher APIntAttr:$c1)),
        (PuArith_XOrIOp $x, (PuArith_ConstantOp (XorIntAttrs $res, $c0, $c1))),
             [(IntAttrTypeMatchesResult $res, $c0),
              (IntAttrTypeMatchesResult $res, $c1)]>;

def InvertPredicate : NativeCodeCall<"invertPredicate($0)">;
def XOrINotCmpI :
    Pat<(PuArith_XOrIOp
          (PuArith_CmpIOp $pred, $a, $b),
          (ConstantLikeMatcher ConstantAttr<I1Attr, "1">)),
        (PuArith_CmpIOp (InvertPredicate $pred), $a, $b)>;

def XOrIOfExtUI :
    Pat<(PuArith_XOrIOp (PuArith_ExtUIOp $x, $nneg1), (PuArith_ExtUIOp $y, $nneg2)),
        (PuArith_ExtUIOp (PuArith_XOrIOp $x, $y),
                       (NativeCodeCall<"($0 && $1) ? $0 : UnitAttr()"> $nneg1, $nneg2)),
      [(TypesMatch $x, $y)]>;

def XOrIOfExtSI :
    Pat<(PuArith_XOrIOp (PuArith_ExtSIOp $x), (PuArith_ExtSIOp $y)), (PuArith_ExtSIOp (PuArith_XOrIOp $x, $y)),
      [(TypesMatch $x, $y)]>;

def CmpIExtSI :
    Pat<(PuArith_CmpIOp $pred,
          (PuArith_ExtSIOp $a),
          (PuArith_ExtSIOp $b)),
        (PuArith_CmpIOp $pred, $a, $b),
        [(TypesMatch $a, $b), (IsEqualityCmpIPredicate $pred)]>;

def CmpIExtUI :
    Pat<(PuArith_CmpIOp $pred,
          (PuArith_ExtUIOp $a, $nneg1),
          (PuArith_ExtUIOp $b, $nneg2)),
        (PuArith_CmpIOp $pred, $a, $b),
        [(TypesMatch $a, $b), (IsEqualityCmpIPredicate $pred)]>;

def SelectNotCond :
    Pat<(SelectOp (PuArith_XOrIOp $pred, (ConstantLikeMatcher APIntAttr:$ones)), $a, $b),
        (SelectOp $pred, $b, $a),
        [(IsScalarOrSplatNegativeOne $ones)]>;

def RedundantSelectTrue :
    Pat<(SelectOp $pred, (SelectOp $pred, $a, $b), $c),
        (SelectOp $pred, $a, $c)>;

def RedundantSelectFalse :
    Pat<(SelectOp $pred, $a, (SelectOp $pred, $b, $c)),
        (SelectOp $pred, $a, $c)>;

def SelectI1ToNot :
    Pat<(SelectOp $pred,
                  (ConstantLikeMatcher ConstantAttr<I1Attr, "0">),
                  (ConstantLikeMatcher ConstantAttr<I1Attr, "1">)),
        (PuArith_XOrIOp $pred, (PuArith_ConstantOp ConstantAttr<I1Attr, "1">))>;

class SelectCmpIToMinMaxBase<string pred, Op minMaxOp> :
    Pat<(SelectOp
          (PuArith_CmpIOp
            ConstantEnumCase<PuArith_CmpIPredicateAttr, pred>,
            $a, $b),
          $a, $b),
        (minMaxOp $a, $b)>;

class SelectCmpIInverseToMinMaxBase<string pred, Op minMaxOp> :
    Pat<(SelectOp
          (PuArith_CmpIOp
            ConstantEnumCase<PuArith_CmpIPredicateAttr, pred>,
            $a, $b),
          $b, $a),
        (minMaxOp $a, $b)>;

def SelectCmpISltToMinSI : SelectCmpIToMinMaxBase<"slt", PuArith_MinSIOp>;
def SelectCmpISleToMinSI : SelectCmpIToMinMaxBase<"sle", PuArith_MinSIOp>;
def SelectCmpISgtToMaxSI : SelectCmpIToMinMaxBase<"sgt", PuArith_MaxSIOp>;
def SelectCmpISgeToMaxSI : SelectCmpIToMinMaxBase<"sge", PuArith_MaxSIOp>;
def SelectCmpIUltToMinUI : SelectCmpIToMinMaxBase<"ult", PuArith_MinUIOp>;
def SelectCmpIUleToMinUI : SelectCmpIToMinMaxBase<"ule", PuArith_MinUIOp>;
def SelectCmpIUgtToMaxUI : SelectCmpIToMinMaxBase<"ugt", PuArith_MaxUIOp>;
def SelectCmpIUgeToMaxUI : SelectCmpIToMinMaxBase<"uge", PuArith_MaxUIOp>;

def SelectCmpISltToMaxSI : SelectCmpIInverseToMinMaxBase<"slt", PuArith_MaxSIOp>;
def SelectCmpISleToMaxSI : SelectCmpIInverseToMinMaxBase<"sle", PuArith_MaxSIOp>;
def SelectCmpISgtToMinSI : SelectCmpIInverseToMinMaxBase<"sgt", PuArith_MinSIOp>;
def SelectCmpISgeToMinSI : SelectCmpIInverseToMinMaxBase<"sge", PuArith_MinSIOp>;
def SelectCmpIUltToMaxUI : SelectCmpIInverseToMinMaxBase<"ult", PuArith_MaxUIOp>;
def SelectCmpIUleToMaxUI : SelectCmpIInverseToMinMaxBase<"ule", PuArith_MaxUIOp>;
def SelectCmpIUgtToMinUI : SelectCmpIInverseToMinMaxBase<"ugt", PuArith_MinUIOp>;
def SelectCmpIUgeToMinUI : SelectCmpIInverseToMinMaxBase<"uge", PuArith_MinUIOp>;

def IndexCastOfExtSI :
    Pat<(PuArith_IndexCastOp (PuArith_ExtSIOp $x)), (PuArith_IndexCastOp $x)>;

def IndexCastUIOfExtUI :
    Pat<(PuArith_IndexCastUIOp (PuArith_ExtUIOp $x, $nneg1), $nneg2),
        (PuArith_IndexCastUIOp $x, $nneg1)>;


def BitcastOfBitcast :
    Pat<(PuArith_BitcastOp (PuArith_BitcastOp $x)), (PuArith_BitcastOp $x)>;

def ExtSIOfExtUI :
    Pat<(PuArith_ExtSIOp (PuArith_ExtUIOp $x, $nneg)),
        (PuArith_ExtUIOp $x, $nneg)>;

def AndIAndIConstant :
    Pat<(PuArith_AndIOp:$res
          (PuArith_AndIOp $x, (ConstantLikeMatcher APIntAttr:$c0)),
          (ConstantLikeMatcher APIntAttr:$c1)),
        (PuArith_AndIOp $x, (PuArith_ConstantOp (AndIntAttrs $res, $c0, $c1))),
             [(IntAttrTypeMatchesResult $res, $c0),
              (IntAttrTypeMatchesResult $res, $c1)]>;

def AndOfExtUI :
    Pat<(PuArith_AndIOp (PuArith_ExtUIOp $x, $nneg1), (PuArith_ExtUIOp $y, $nneg2)),
        (PuArith_ExtUIOp (PuArith_AndIOp $x, $y),
                       (NativeCodeCall<"$0 ? $0 : $1"> $nneg1, $nneg2)),
        [(TypesMatch $x, $y)]>;

def AndOfExtSI :
    Pat<(PuArith_AndIOp (PuArith_ExtSIOp $x), (PuArith_ExtSIOp $y)),
        (PuArith_ExtSIOp (PuArith_AndIOp $x, $y)),
        [(TypesMatch $x, $y)]>;

def OrIOrIConstant :
    Pat<(PuArith_OrIOp:$res
          (PuArith_OrIOp $x, (ConstantLikeMatcher APIntAttr:$c0)),
          (ConstantLikeMatcher APIntAttr:$c1)),
        (PuArith_OrIOp $x, (PuArith_ConstantOp (OrIntAttrs $res, $c0, $c1))),
             [(IntAttrTypeMatchesResult $res, $c0),
              (IntAttrTypeMatchesResult $res, $c1)]>;

def OrOfExtUI :
    Pat<(PuArith_OrIOp (PuArith_ExtUIOp $x, $nneg1), (PuArith_ExtUIOp $y, $nneg2)),
        (PuArith_ExtUIOp (PuArith_OrIOp $x, $y),
                       (NativeCodeCall<"($0 && $1) ? $0 : UnitAttr()"> $nneg1, $nneg2)),
        [(TypesMatch $x, $y)]>;

def OrOfExtSI :
    Pat<(PuArith_OrIOp (PuArith_ExtSIOp $x), (PuArith_ExtSIOp $y)),
        (PuArith_ExtSIOp (PuArith_OrIOp $x, $y)),
        [(TypesMatch $x, $y)]>;

def ValueWiderThan :
    Constraint<And<[
      CPred<"getScalarOrElementWidth($0) > getScalarOrElementWidth($1)">,
      CPred<"getScalarOrElementWidth($1) > 0">]>>;

def TruncationMatchesShiftAmount :
    Constraint<And<[
      CPred<"succeeded(getIntOrSplatIntValue($2))">,
      CPred<"(getScalarOrElementWidth($0) - getScalarOrElementWidth($1)) == "
              "*getIntOrSplatIntValue($2)">]>>;

def TruncIExtSIToExtSI :
    Pat<(PuArith_TruncIOp:$tr (PuArith_ExtSIOp:$ext $x), $overflow),
        (PuArith_ExtSIOp $x),
        [(ValueWiderThan $ext, $tr),
         (ValueWiderThan $tr, $x)]>;

def TruncIExtUIToExtUI :
    Pat<(PuArith_TruncIOp:$tr (PuArith_ExtUIOp:$ext $x, $nneg), $overflow),
        (PuArith_ExtUIOp $x, $nneg),
        [(ValueWiderThan $ext, $tr),
         (ValueWiderThan $tr, $x)]>;

def TruncIShrSIToTrunciShrUI :
    Pat<(PuArith_TruncIOp:$tr
          (PuArith_ShRSIOp $x, (ConstantLikeMatcher TypedAttrInterface:$c0), $exact), $overflow),
        (PuArith_TruncIOp (PuArith_ShRUIOp $x, (PuArith_ConstantOp (cast<"TypedAttr"> $c0)), $exact), $overflow),
        [(TruncationMatchesShiftAmount $x, $tr, $c0)]>;

def TruncFSIToFPToSIToFP :
    Pat<(PuArith_TruncFOp:$tr (PuArith_SIToFPOp:$fp $x), $rmf, $fmf),
        (PuArith_SIToFPOp $x),
        [(Constraint<CPred<"$0 == nullptr">, "default rounding mode"> $rmf)]>;

def TruncFUIToFPToUIToFP :
    Pat<(PuArith_TruncFOp:$tr (PuArith_UIToFPOp:$fp $x, $nneg), $rmf, $fmf),
        (PuArith_UIToFPOp $x, $nneg),
        [(Constraint<CPred<"$0 == nullptr">, "default rounding mode"> $rmf)]>;

def SIToFPOfExtSI :
    Pat<(PuArith_SIToFPOp (PuArith_ExtSIOp $x)),
        (PuArith_SIToFPOp $x)>;

def SIToFPOfExtUI :
    Pat<(PuArith_SIToFPOp (PuArith_ExtUIOp $x, $nneg)),
        (PuArith_UIToFPOp $x, $nneg)>;

def UIToFPOfExtUI :
    Pat<(PuArith_UIToFPOp (PuArith_ExtUIOp $x, $nneg1), $nneg2),
        (PuArith_UIToFPOp $x, $nneg1)>;

def AddFOfNegFLhs :
    Pat<(PuArith_AddFOp (PuArith_NegFOp $x, $_), $y, $fmf, $rm),
        (PuArith_SubFOp $y, $x, $fmf, $rm)>;

def AddFOfNegFRhs :
    Pat<(PuArith_AddFOp $x, (PuArith_NegFOp $y, $_), $fmf, $rm),
        (PuArith_SubFOp $x, $y, $fmf, $rm)>;

def IsNegZeroFloat :
    Constraint<CPred<"m_NegZeroFloat().match($0)">,
               "is negative-zero float constant">;

def SubFOfNegZero :
    Pat<(PuArith_SubFOp (ConstantLikeMatcher AnyAttr:$c), $x, $fmf, $rm),
        (PuArith_NegFOp $x, $fmf),
        [(IsNegZeroFloat $c),
         (Constraint<CPred<"$0 == nullptr">, "default rounding mode"> $rm)]>;

def MulFOfNegF :
    Pat<(PuArith_MulFOp (PuArith_NegFOp $x, $_), (PuArith_NegFOp $y, $_), $fmf, $rm),
        (PuArith_MulFOp $x, $y, $fmf, $rm),
        [(TypesMatch $x, $y),
         (Constraint<CPred<"$0 == nullptr">, "default rounding mode"> $rm)]>;

def DivFOfNegF :
    Pat<(PuArith_DivFOp (PuArith_NegFOp $x, $_), (PuArith_NegFOp $y, $_), $fmf, $rm),
        (PuArith_DivFOp $x, $y, $fmf, $rm),
        [(TypesMatch $x, $y),
         (Constraint<CPred<"$0 == nullptr">, "default rounding mode"> $rm)]>;

#endif
