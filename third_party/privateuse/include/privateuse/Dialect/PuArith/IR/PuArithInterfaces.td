include "privateuse/Dialect/PuArith/IR/PuArithBase.td"



def PuArithFastMathInterface : OpInterface<"PuArithFastMathInterface"> {
  let description = [{
    Access to operation fastmath flags.
  }];

  let cppNamespace = "::mlir::puarith";

  let methods = [
    InterfaceMethod<
               "Returns a FastMathFlagsAttr attribute for the operation",
         "FastMathFlagsAttr",
         "getFastMathFlagsAttr",
               (ins),
         [{}],
        [{
        ConcreteOp op = cast<ConcreteOp>(this->getOperation());
        return op.getFastmathAttr();
      }]
      >,
    StaticInterfaceMethod<
               [{Returns the name of the FastMathFlagsAttr attribute
                         for the operation}],
         "StringRef",
         "getFastMathAttrName",
               (ins),
         [{}],
        [{
        return "fastmath";
      }]
      >

  ];
}

def PuArithIntegerOverflowFlagsInterface : OpInterface<"PuArithIntegerOverflowFlagsInterface"> {
  let description = [{
    Access to op integer overflow flags.
  }];

  let cppNamespace = "::mlir::puarith";

  let methods = [
    InterfaceMethod<
               "Returns an IntegerOverflowFlagsAttr attribute for the operation",
         "IntegerOverflowFlagsAttr",
         "getOverflowAttr",
               (ins),
         [{}],
        [{
        auto op = cast<ConcreteOp>(this->getOperation());
        return op.getOverflowFlagsAttr();
      }]
      >,
    InterfaceMethod<
               "Returns whether the operation has the No Unsigned Wrap keyword",
         "bool",
         "hasNoUnsignedWrap",
               (ins),
         [{}],
        [{
        auto op = cast<ConcreteOp>(this->getOperation());
        IntegerOverflowFlags flags = op.getOverflowFlagsAttr().getValue();
        return bitEnumContainsAll(flags, IntegerOverflowFlags::nuw);
      }]
      >,
    InterfaceMethod<
               "Returns whether the operation has the No Signed Wrap keyword",
         "bool",
         "hasNoSignedWrap",
               (ins),
         [{}],
        [{
        auto op = cast<ConcreteOp>(this->getOperation());
        IntegerOverflowFlags flags = op.getOverflowFlagsAttr().getValue();
        return bitEnumContainsAll(flags, IntegerOverflowFlags::nsw);
      }]
      >,
    StaticInterfaceMethod<
               [{Returns the name of the IntegerOverflowFlagsAttr attribute
                         for the operation}],
         "StringRef",
         "getIntegerOverflowAttrName",
               (ins),
         [{}],
        [{
        return "overflowFlags";
      }]
      >
  ];
}

def PuArithNonNegFlagInterface : OpInterface<"PuArithNonNegFlagInterface"> {
  let description = [{
    Access to op non-negative (nneg) flag.
  }];

  let cppNamespace = "::mlir::puarith";

  let methods = [
    InterfaceMethod<
               "Returns whether the operation has the nneg flag set",
         "bool",
         "getNonNeg",
               (ins),
         [{}],
        [{
        auto op = cast<ConcreteOp>(this->getOperation());
        return op.getNonNegAttr() != nullptr;
      }]
      >,
    InterfaceMethod<
               "Set the nneg flag for the operation",
         "void",
         "setNonNeg",
               (ins "bool":$nonNeg),
         [{}],
        [{
        auto op = cast<ConcreteOp>(this->getOperation());
        if (nonNeg)
          op.setNonNegAttr(UnitAttr::get(op->getContext()));
        else
          op.removeNonNegAttr();
      }]
      >,
    StaticInterfaceMethod<
               [{Returns the name of the NonNeg flag attribute for
                         the operation}],
         "StringRef",
         "getNonNegFlagAttrName",
               (ins),
         [{}],
        [{
        return "nonNeg";
      }]
      >
  ];
}

def PuArithRoundingModeInterface : OpInterface<"PuArithRoundingModeInterface"> {
  let description = [{
    Access to op rounding mode.
  }];

  let cppNamespace = "::mlir::puarith";

  let methods = [
    InterfaceMethod<
               "Returns a RoundingModeAttr attribute for the operation",
         "RoundingModeAttr",
         "getRoundingModeAttr",
               (ins),
         [{}],
        [{
        auto op = cast<ConcreteOp>(this->getOperation());
        return op.getRoundingmodeAttr();
      }]
    >,
    StaticInterfaceMethod<
               [{Returns the name of the RoundingModeAttr attribute for
                         the operation}],
         "StringRef",
         "getRoundingModeAttrName",
               (ins),
         [{}],
        [{
        return "roundingmode";
      }]
    >
  ];
}


def PuArith_CastInterface : OpInterface<"CastInterface"> {
  let description = [{
    Marker interface for the puarith type conversion operations (`fptoui`,
    `fptosi`, `trunci`, `extui`, `extsi`, `uitofp`, `sitofp`, `bitcast`,
    ...). The register-splitting pass matches these to split wide casts into
    per-chunk conversions placed into low/high register halves.
  }];

  let cppNamespace = "::mlir::puarith";
}
