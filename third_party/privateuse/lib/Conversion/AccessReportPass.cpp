#include "privateuse/Analysis/AccessAnalysis.h"
#include "privateuse/Conversion/Passes.h"

#include "llvm/Support/raw_ostream.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "triton/Dialect/Triton/IR/Dialect.h"

#include <array>

#define GEN_PASS_DEF_ANALYZEACCESS
#include "privateuse/Conversion/Passes.h.inc"

using namespace mlir;
using namespace mlir::triton;

namespace {

std::string describeDims(const SmallVector<AccessDim>& dims)
{
  std::string text;
  llvm::raw_string_ostream os(text);
  os << '[';
  for (size_t d = 0; d < dims.size(); ++d) {
    if (d) os << ", ";
    os << "dim" << d << "={start:";
    printExpr(os, dims[d].start);
    os << ",stride:" << dims[d].stride << ",size:" << dims[d].size << '}';
  }
  os << ']';
  os.flush();
  return text;
}

std::string describeMask(const AccessMask& mask)
{
  switch (mask.kind) {
    case MaskKind::None: return "none";
    case MaskKind::ScalarDynamic: return "scalar";
    case MaskKind::ProvenTrue: return "proven_true";
    case MaskKind::ProvenFalse: return "proven_false";
    case MaskKind::Box: break;
  }

  std::string text = "box";
  llvm::raw_string_ostream os(text);
  for (size_t d = 0; d < mask.dimBounds.size(); ++d) {
    if (mask.dimBounds[d].empty()) continue;
    os << " dim" << d << "=[";
    for (size_t b = 0; b < mask.dimBounds[d].size(); ++b) {
      if (b) os << ", ";
      const LinearBound& bound = mask.dimBounds[d][b];
      os << '(';
      printExpr(os, bound.offset);
      os << " + " << bound.stride << "*i" << (bound.strict ? " < 0" : " <= 0") << ')';
    }
    os << ']';
  }
  os.flush();
  return text;
}

struct MemoryOpInfo
{
  bool matches      = false;
  bool isAtomic     = false;
  bool isDescriptor = false;
};

MemoryOpInfo classifyOp(Operation* op)
{
  MemoryOpInfo info;
  if (isa<LoadOp, StoreOp>(op)) {
    info.matches = true;
  } else if (isa<AtomicRMWOp>(op)) {
    info.matches  = true;
    info.isAtomic = true;
  } else if (isa<DescriptorLoadOp, DescriptorStoreOp, DescriptorReduceOp, DescriptorGatherOp, DescriptorScatterOp>(op
             )) {
    info.matches      = true;
    info.isDescriptor = true;
  } else if (isa<GatherOp>(op)) {
    info.matches = true;
  }
  return info;
}

struct AnalyzeAccessPass : public ::impl::AnalyzeAccessBase<AnalyzeAccessPass>
{
  using AnalyzeAccessBase::AnalyzeAccessBase;

  void runOnOperation() override
  {
    ModuleOp module = getOperation();
    AccessAnalyzer analyzer(module);

    unsigned index       = 0;
    unsigned loads       = 0;
    unsigned stores      = 0;
    unsigned atomics     = 0;
    unsigned descriptors = 0;
    unsigned deferred    = 0;
    unsigned errors      = 0;
    std::array<unsigned, 5> reasons{};

    module.walk([&](Operation* op) {
      MemoryOpInfo kind = classifyOp(op);
      if (!kind.matches) return;

      ++index;
      if (isa<LoadOp>(op))
        ++loads;
      else if (isa<StoreOp>(op))
        ++stores;
      else if (kind.isAtomic)
        ++atomics;
      else if (kind.isDescriptor)
        ++descriptors;

      FailureOr<AccessInfo> info = analyzer.analyze(op);
      if (failed(info)) {
        ++errors;
        return;
      }
      if (info->deferred != DeferredReason::None) {
        ++deferred;
        ++reasons[static_cast<unsigned>(info->deferred)];
      }

      std::string text;
      llvm::raw_string_ostream os(text);
      os << "[privateuse-access] op " << index << ": " << op->getName().getStringRef();

      if (info->view.base) {
        os << " rank=" << info->view.dims.size() << " elem_bytes=" << info->elemBytes << " offset=";
        printExpr(os, info->view.offset);
        os << " dims=" << describeDims(info->view.dims) << " mask=" << describeMask(info->mask);
      } else {
        os << " rank=? elem_bytes=? offset=? dims=[] mask=?";
      }

      os << " deferred=" << toString(info->deferred);
      os.flush();
      llvm::errs() << text << '\n';
    });

    std::string text;
    llvm::raw_string_ostream os(text);
    os << "[privateuse-access] summary ops=" << index << " loads=" << loads << " stores=" << stores
       << " atomics=" << atomics << " descriptors=" << descriptors << " deferred=" << deferred << " errors=" << errors
       << " reasons={";
    bool first = true;
    for (unsigned reason = 1; reason < reasons.size(); ++reason) {
      if (!reasons[reason]) continue;
      if (!first) os << ',';
      first = false;
      os << toString(static_cast<DeferredReason>(reason)) << '=' << reasons[reason];
    }
    os << '}';
    os.flush();
    llvm::errs() << text << '\n';

    if (errors) signalPassFailure();
  }
};

} // namespace

std::unique_ptr<OperationPass<ModuleOp>> triton::createAnalyzeAccessPass()
{
  return std::make_unique<AnalyzeAccessPass>();
}
