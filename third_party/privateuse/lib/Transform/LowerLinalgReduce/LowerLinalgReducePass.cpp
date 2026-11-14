#include "privateuse/Transform/LowerLinalgReduce/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/PatternMatch.h"
#include "llvm/ADT/BitVector.h"
#include "llvm/ADT/STLExtras.h"

#include <optional>

#define GEN_PASS_DECL_LOWERLINALGREDUCE
#include "privateuse/Transform/LowerLinalgReduce/Passes.h.inc"

#define GEN_PASS_DEF_LOWERLINALGREDUCE
#include "privateuse/Transform/LowerLinalgReduce/Passes.h.inc"

using namespace mlir;
using namespace mlir::triton;

namespace {

struct SingleAxisReduce
{
  linalg::GenericOp op;
  Block* body;
  unsigned axis;
  unsigned inputRank;
  unsigned resultRank;
  int64_t redExtent;
  SmallVector<int64_t> resultShape;
  SmallVector<Value> inputs;
  SmallVector<Value> inits;
  llvm::BitVector usedInputs;
};

std::optional<unsigned> singleReductionAxis(linalg::GenericOp op)
{
  std::optional<unsigned> axis;
  for (auto [index, type] : llvm::enumerate(op.getIteratorTypesArray())) {
    if (type != utils::IteratorType::reduction) continue;
    if (axis) return std::nullopt;
    axis = index;
  }
  return axis;
}

FailureOr<SingleAxisReduce> matchReduce(linalg::GenericOp op, IRRewriter& rewriter)
{
  if (op.getNumDpsInits() == 0 || op.getNumDpsInputs() != op.getNumDpsInits())
    return rewriter.notifyMatchFailure(op, "expected one input per init");

  std::optional<unsigned> axis = singleReductionAxis(op);
  if (!axis) return rewriter.notifyMatchFailure(op, "expected exactly one reduction iterator");

  auto inputType = dyn_cast<MemRefType>(op.getDpsInputOperand(0)->get().getType());
  auto initType  = dyn_cast<MemRefType>(op.getDpsInitOperand(0)->get().getType());
  if (!inputType || !initType || !inputType.hasStaticShape() || !initType.hasStaticShape())
    return rewriter.notifyMatchFailure(op, "expected statically shaped memrefs (run after bufferization)");

  unsigned inputRank  = inputType.getRank();
  unsigned resultRank = initType.getRank();
  if (inputRank != resultRank + 1 || inputRank != op.getNumLoops())
    return rewriter.notifyMatchFailure(op, "expected the reduction to drop exactly one dimension");

  ArrayRef<int64_t> inputShape  = inputType.getShape();
  ArrayRef<int64_t> resultShape = initType.getShape();
  for (unsigned dim = 0; dim < resultRank; ++dim)
    if (resultShape[dim] != inputShape[dim < *axis ? dim : dim + 1])
      return rewriter.notifyMatchFailure(op, "init shape does not match the input without the reduction axis");

  SmallVector<Value> inputs(op.getDpsInputs());
  SmallVector<Value> inits(op.getDpsInits());
  for (Value input : inputs) {
    auto type = dyn_cast<MemRefType>(input.getType());
    if (!type || !type.hasStaticShape() || type.getRank() != inputRank)
      return rewriter.notifyMatchFailure(op, "expected every input to share the reduced memref shape");
  }

  Block* body = op.getBlock();
  if (body->getNumArguments() != inputs.size() + inits.size())
    return rewriter.notifyMatchFailure(op, "unexpected linalg block signature");

  llvm::BitVector usedInputs(inputs.size());
  for (unsigned i = 0; i < inputs.size(); ++i) usedInputs[i] = !body->getArgument(i).use_empty();
  if (usedInputs.none()) return rewriter.notifyMatchFailure(op, "reduce body does not use any input");

  MLIRContext* context = op.getContext();
  AffineMap identity   = AffineMap::getMultiDimIdentityMap(inputRank, context);
  for (OpOperand* operand : op.getDpsInputOperands())
    if (op.getMatchingIndexingMap(operand) != identity)
      return rewriter.notifyMatchFailure(op, "expected identity indexing maps on the inputs");

  SmallVector<AffineExpr> keptDims;
  for (unsigned dim = 0; dim < inputRank; ++dim)
    if (dim != *axis) keptDims.push_back(getAffineDimExpr(dim, context));
  AffineMap initMap = AffineMap::get(inputRank, 0, keptDims, context);
  for (unsigned i = 0; i < op.getNumDpsInits(); ++i)
    if (op.getMatchingIndexingMap(op.getDpsInitOperand(i)) != initMap)
      return rewriter.notifyMatchFailure(op, "expected the inits to index the non-reduced dimensions");

  SingleAxisReduce reduce;
  reduce.op         = op;
  reduce.body       = body;
  reduce.axis       = *axis;
  reduce.inputRank  = inputRank;
  reduce.resultRank = resultRank;
  reduce.redExtent  = inputShape[*axis];
  reduce.resultShape.assign(resultShape.begin(), resultShape.end());
  reduce.inputs     = std::move(inputs);
  reduce.inits      = std::move(inits);
  reduce.usedInputs = std::move(usedInputs);
  return reduce;
}

Value indexConstant(OpBuilder& b, Location loc, int64_t value)
{
  return arith::ConstantIndexOp::create(b, loc, value);
}

Value loadElement(const SingleAxisReduce& reduce, OpBuilder& b, Value input, Value redIdx, ValueRange ivs)
{
  SmallVector<Value> indices;
  for (unsigned dim = 0; dim < reduce.inputRank; ++dim) {
    if (dim == reduce.axis) {
      indices.push_back(redIdx);
      continue;
    }
    indices.push_back(ivs[dim < reduce.axis ? dim : dim - 1]);
  }
  return memref::LoadOp::create(b, reduce.op->getLoc(), input, indices);
}

SmallVector<Value> loadInputs(const SingleAxisReduce& reduce, OpBuilder& b, Value redIdx, ValueRange ivs)
{
  SmallVector<Value> elems(reduce.inputs.size());
  Value placeholder;
  for (auto [i, input] : llvm::enumerate(reduce.inputs))
    if (reduce.usedInputs[i]) {
      elems[i] = loadElement(reduce, b, input, redIdx, ivs);
      if (!placeholder) placeholder = elems[i];
    }

  for (unsigned i = 0; i < elems.size(); ++i)
    if (!reduce.usedInputs[i]) elems[i] = placeholder;
  return elems;
}

SmallVector<Value> combine(const SingleAxisReduce& reduce, OpBuilder& b, ValueRange elems, ValueRange accs)
{
  IRMapping mapping;
  for (unsigned i = 0; i < elems.size(); ++i) mapping.map(reduce.body->getArgument(i), elems[i]);
  for (unsigned j = 0; j < accs.size(); ++j) mapping.map(reduce.body->getArgument(elems.size() + j), accs[j]);

  for (Operation& inner : reduce.body->without_terminator()) b.clone(inner, mapping);
  return llvm::map_to_vector(reduce.body->getTerminator()->getOperands(), [&](Value value) {
    return mapping.lookup(value);
  });
}

void emitReduction(const SingleAxisReduce& reduce, OpBuilder& b, ValueRange ivs)
{
  Location loc = reduce.op->getLoc();

  SmallVector<Value> seeds = loadInputs(reduce, b, indexConstant(b, loc, 0), ivs);
  auto loop                = scf::ForOp::create(
    b, loc, indexConstant(b, loc, 1), indexConstant(b, loc, reduce.redExtent), indexConstant(b, loc, 1), seeds,
    [&](OpBuilder& ib, Location, Value redIdx, ValueRange accs) {
      SmallVector<Value> elems = loadInputs(reduce, ib, redIdx, ivs);
      scf::YieldOp::create(ib, loc, combine(reduce, ib, elems, accs));
    }
  );
  for (auto [result, init] : llvm::zip_equal(loop.getResults(), reduce.inits))
    memref::StoreOp::create(b, loc, result, init, ivs);
}

void emitLoops(const SingleAxisReduce& reduce, OpBuilder& b, ValueRange ivs)
{
  if (ivs.size() == reduce.resultRank) {
    emitReduction(reduce, b, ivs);
    return;
  }

  Location loc = reduce.op->getLoc();
  unsigned dim = ivs.size();
  scf::ForOp::create(
    b, loc, indexConstant(b, loc, 0), indexConstant(b, loc, reduce.resultShape[dim]), indexConstant(b, loc, 1),
    ValueRange{},
    [&](OpBuilder& ib, Location, Value iv, ValueRange) {
      SmallVector<Value> nested(ivs.begin(), ivs.end());
      nested.push_back(iv);
      emitLoops(reduce, ib, nested);
      scf::YieldOp::create(ib, loc, ValueRange{});
    }
  );
}

void lowerReduce(linalg::GenericOp op, IRRewriter& rewriter)
{
  FailureOr<SingleAxisReduce> reduce = matchReduce(op, rewriter);
  if (failed(reduce)) return;

  OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPoint(op);
  emitLoops(*reduce, rewriter, {});
  rewriter.eraseOp(op);
  return;
}

struct LowerLinalgReducePass : public ::impl::LowerLinalgReduceBase<LowerLinalgReducePass>
{
  using LowerLinalgReduceBase::LowerLinalgReduceBase;

  void getDependentDialects(DialectRegistry& registry) const override
  {
    // clang-format off
    registry.insert<
        arith::ArithDialect,
        func::FuncDialect,
        linalg::LinalgDialect,
        memref::MemRefDialect,
        scf::SCFDialect
    >();
    // clang-format on
  }

  void runOnOperation() override
  {
    SmallVector<linalg::GenericOp> candidates;
    getOperation().walk([&](linalg::GenericOp generic) {
      if (generic.hasPureBufferSemantics() && singleReductionAxis(generic)) candidates.push_back(generic);
    });

    IRRewriter rewriter(&getContext());
    for (linalg::GenericOp generic : candidates) lowerReduce(generic, rewriter);
  }
};

} // namespace

std::unique_ptr<OperationPass<ModuleOp>> triton::createLowerLinalgReducePass()
{
  return std::make_unique<LowerLinalgReducePass>();
}
