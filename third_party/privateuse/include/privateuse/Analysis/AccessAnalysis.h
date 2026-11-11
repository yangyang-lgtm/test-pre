#ifndef PRIVATEUSE_ANALYSIS_ACCESSANALYSIS_H
#define PRIVATEUSE_ANALYSIS_ACCESSANALYSIS_H

#include "privateuse/Analysis/ValueRange.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Support/LogicalResult.h"

#include <cstdint>

namespace mlir {

namespace arith {
class CmpIOp;
class ConstantOp;
class MulIOp;
} // namespace arith

namespace triton {

class BroadcastOp;
class ExpandDimsOp;
class MakeRangeOp;
class SplatOp;

enum class DeferredReason : uint8_t {
  None,
  NegativeStride,
  DataDependentOffset,
  UnsupportedDescriptor,
  Atomic,
};

struct AccessDim
{
  ExprPtr start;
  int64_t size   = 0;
  int64_t stride = 0;
};

struct AccessView
{
  Value base;
  ExprPtr offset;
  SmallVector<AccessDim> dims;

  LogicalResult normalize(Type resultType, Location loc);
};

struct LinearBound
{
  ExprPtr offset;
  int64_t stride = 0;
  bool strict    = false;
  bool lower     = false;
};

struct AccessMask
{
  bool present       = false;
  bool allTrue       = true;
  bool allFalse      = false;
  bool scalarDynamic = false;
  SmallVector<SmallVector<LinearBound>> dimBounds;

  static AccessMask uniform(bool truth);
  static AccessMask dynamicScalar();
};

struct AccessInfo
{
  bool isLoad       = false;
  int64_t elemBytes = 0;
  AccessView view;
  AccessMask mask;
  DeferredReason deferred = DeferredReason::None;

  static AccessInfo makeDeferred(DeferredReason reason, bool isLoad = false);
};

class AccessAnalyzer
{
public:
  explicit AccessAnalyzer(ModuleOp module);

  FailureOr<AccessInfo> analyze(Operation* op);

private:
  FailureOr<AccessInfo> analyzeAccess(Value ptr, Value mask, bool isLoad, Location loc);
  FailureOr<ExprPtr> buildScalar(Value value);

  FailureOr<AccessView> buildValue(Value value);
  FailureOr<AccessView> buildConstant(arith::ConstantOp op, ArrayRef<int64_t> shape);
  FailureOr<AccessView> buildSplat(SplatOp op, ArrayRef<int64_t> shape);
  FailureOr<AccessView> buildMakeRange(MakeRangeOp op, ArrayRef<int64_t> shape);
  FailureOr<AccessView> buildExpandDims(ExpandDimsOp op, ArrayRef<int64_t> shape);
  FailureOr<AccessView> buildBroadcast(BroadcastOp op, ArrayRef<int64_t> shape);
  FailureOr<AccessView> buildMultiply(arith::MulIOp op);
  FailureOr<AccessView> buildBlockArgument(BlockArgument arg);
  FailureOr<AccessView> buildScalarLeaf(Value value);
  FailureOr<AccessView> combineValues(Value lhs, Value rhs, bool subtract, Value result);
  FailureOr<AccessView> combineViews(AccessView lhs, AccessView rhs, bool subtract, Type resultType, Location loc);
  FailureOr<AccessView> resolveIterArg(BlockArgument arg);
  FailureOr<AccessView> scale(AccessView value, int64_t factor, Location loc);

  FailureOr<AccessMask> buildMask(Value value, size_t rank);
  FailureOr<AccessMask> buildMaskImpl(Value value, size_t rank);
  FailureOr<AccessMask> buildConstantMask(arith::ConstantOp op);
  FailureOr<AccessMask> buildSplatMask(SplatOp op);
  FailureOr<AccessMask> buildCompareMask(arith::CmpIOp op, size_t rank);
  FailureOr<AccessMask> buildBroadcastMask(BroadcastOp op, size_t rank);
  FailureOr<AccessMask> buildExpandDimsMask(ExpandDimsOp op, size_t rank);

  void classifyMask(AccessMask& mask, ArrayRef<int64_t> sizes);
  unsigned loopId(Operation* loop);

  SymbolBounds bounds;
  llvm::DenseMap<Operation*, unsigned> loopIds;
  unsigned nextLoopId = 0;
  llvm::DenseSet<Value> resolving;
  DeferredReason deferred = DeferredReason::None;
};

llvm::StringRef toString(DeferredReason reason);
} // namespace triton
} // namespace mlir

#endif
