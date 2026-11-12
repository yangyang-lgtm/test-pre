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

class OpBuilder;

namespace arith {
class CmpIOp;
class ConstantOp;
class MulIOp;
} // namespace arith

namespace scf {
class ForOp;
} // namespace scf

namespace triton {

class BroadcastOp;
class ExpandDimsOp;
class MakeRangeOp;
class ReshapeOp;
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
  bool isInjective() const;
};

struct LinearBound
{
  ExprPtr offset;
  int64_t stride = 0;
  bool strict    = false;
  bool lower     = false;
};

enum class MaskKind {
  None,
  ProvenTrue,
  ProvenFalse,
  ScalarDynamic,
  Box,
};

struct AccessMask
{
  MaskKind kind = MaskKind::None;
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
  FailureOr<Value> materializeIndex(const ExprPtr& expr, OpBuilder& builder, Location loc);
  FailureOr<Value> materializeBoundCount(const LinearBound& bound, OpBuilder& builder, Location loc);
  FailureOr<Value> materializeDimSize(ArrayRef<LinearBound> bounds, int64_t dimSize, OpBuilder& builder, Location loc);
  FailureOr<SmallVector<Value>> materializeMaskSizes(const AccessInfo& info, OpBuilder& builder, Location loc);

private:
  FailureOr<AccessInfo> analyzeAccess(Value ptr, Value mask, bool isLoad, Location loc);
  FailureOr<ExprPtr> buildScalar(Value value);
  Value materializeSymbol(SymKind kind, unsigned id, OpBuilder& builder, Location loc);

  FailureOr<AccessView> buildValue(Value value);
  FailureOr<AccessView> buildConstant(arith::ConstantOp op, ArrayRef<int64_t> shape);
  FailureOr<AccessView> buildSplat(SplatOp op, ArrayRef<int64_t> shape);
  FailureOr<AccessView> buildMakeRange(MakeRangeOp op, ArrayRef<int64_t> shape);
  FailureOr<AccessView> buildExpandDims(ExpandDimsOp op, ArrayRef<int64_t> shape);
  FailureOr<AccessView> buildBroadcast(BroadcastOp op, ArrayRef<int64_t> shape);
  FailureOr<AccessView> buildReshape(ReshapeOp op, ArrayRef<int64_t> shape);
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
  void recordLoopIv(scf::ForOp loop);

  SymbolBounds bounds_;
  llvm::DenseMap<Operation*, unsigned> loopIds_;
  unsigned nextLoopId_ = 0;
  llvm::DenseSet<Value> resolving_;
  DeferredReason deferred_        = DeferredReason::None;
  std::array<Value, 3> pidValues_ = {};
  llvm::DenseMap<unsigned, Value> ivValues_;
};

llvm::StringRef toString(DeferredReason reason);
} // namespace triton
} // namespace mlir

#endif
