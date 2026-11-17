#ifndef PRIVATEUSE_ANALYSIS_DATAFLOWANALYSIS_H
#define PRIVATEUSE_ANALYSIS_DATAFLOWANALYSIS_H

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Value.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Support/LogicalResult.h"

#include <cstdint>
#include <memory>

namespace mlir {
namespace triton {

enum class Space : uint8_t { Dram = 0, Sram = 128, Unknown = 255 };
enum class PlaceKind : uint8_t { KernelArg, SramAlloc, LoopCarried };
enum class PathKind : uint8_t { Loop, IfThen, IfElse };
enum class Dir : uint8_t { DramToSram, SramToDram, SramToSram };
enum class OpKind : uint8_t { Copy, Gather, Scatter };
enum class Role : uint8_t { Read, Write, ScalarRead, ScalarWrite, Atomic };
enum class Branch : uint8_t { None, Init, Yield };
enum class Guard : uint8_t { None, FirstIter, BodyIter };

enum class DataflowDeferredReason : uint8_t {
  None = 0,
  UnsupportedControlFlow,
  UnmodeledRegion,
  MultiplePidLoops,
  UnsupportedMemorySpace,
  SramAllocOutsidePid,
  PidCarriesMemref,
  NestedCarrier,
  SymbolicTrip,
  UnresolvedMemref,
  TransferBetweenCarriers,
  DramToDramTransfer,
  MemrefDealloc,
  MemrefEscape,
  ScalarWriteToSram,
  LoadDepthMismatch,
  StoreDepthMismatch,
  MultipleLoadProducers,
  MultipleStoreConsumers,
  StationWithoutProducer,
  Count
};

struct PathInfo
{
  Operation* op            = nullptr;
  PathKind kind            = PathKind::Loop;
  unsigned id              = 0;
  unsigned parentPathId    = 0;
  int64_t blockPosInParent = 0;
  unsigned loopId          = 0;
  unsigned ifId            = 0;
};

struct LoopInfo
{
  scf::ForOp op;
  int64_t parentId         = -1;
  unsigned pathId          = 0;
  unsigned parentPathId    = 0;
  unsigned id              = 0;
  int64_t blockPosInParent = 0;
  int64_t trip             = -1;
  bool isPid               = false;
};

struct ProgramPoint
{
  unsigned pathId = 0;
  SmallVector<int64_t> blockPos;
  bool isLoopExit      = false;
  unsigned exitLoopId  = 0;
  int64_t delta        = 0;
  unsigned shiftLoopId = 0;
};

struct Transfer
{
  Operation* op = nullptr;
  unsigned src = 0, dst = 0;
  Dir dir            = Dir::DramToSram;
  OpKind opKind      = OpKind::Copy;
  unsigned pathId    = 0;
  int64_t blockOrder = 0;
  bool masked = false, dynamic = false, indirect = false;
  int64_t delta     = 0;
  int64_t carrierId = -1;
  bool carrierAtDst = false;
  Branch branch     = Branch::None;
  Guard guard       = Guard::None;
  int64_t groupId   = -1;
};

struct Access
{
  Operation* op         = nullptr;
  unsigned place        = 0;
  Role role             = Role::Read;
  unsigned pathId       = 0;
  int64_t blockOrder    = 0;
  int64_t delta         = 0;
  int64_t carrierId     = -1;
  Branch branch         = Branch::None;
  Guard guard           = Guard::None;
  int64_t groupId       = -1;
  unsigned operandIndex = 0;
};

struct Carrier
{
  BlockArgument arg;
  unsigned loopId     = 0;
  unsigned initPlace  = 0;
  unsigned yieldPlace = 0;
  bool rotation       = false;
  bool inPlace        = false;
  bool wroteBody      = false;
  unsigned id         = 0;
  unsigned placeId    = 0;
};

struct Place
{
  struct EventRef
  {
    enum Kind : uint8_t { TransferK, AccessK } kind = AccessK;
    unsigned id                                     = 0;
  };

  unsigned id    = 0;
  PlaceKind kind = PlaceKind::KernelArg;
  Space space    = Space::Unknown;
  Value anchor;
  unsigned defPathId  = 0;
  unsigned allocDepth = 0;
  int64_t allocOrder  = 0;
  SmallVector<EventRef> producers, consumers;
  SmallVector<Access> accesses;
  ProgramPoint start, retire;
  SmallVector<unsigned> carries;
  bool dead = false, through = false, indirect = false, carried = false;
};

struct FunctionInfo
{
  FunctionOpInterface op;
  SmallVector<LoopInfo> loops;
  SmallVector<PathInfo> paths;
  SmallVector<Place, 2> places;
  SmallVector<Transfer> transfers;
  SmallVector<Carrier> carriers;
  SmallVector<Access> accesses;
  unsigned unresolved = 0;
  unsigned tailWrites = 0;
};

class DataflowAnalyzer
{
public:
  struct AnalysisResult
  {
    FunctionInfo info;
    DataflowDeferredReason deferred = DataflowDeferredReason::None;

    bool isDeferred() const { return deferred != DataflowDeferredReason::None; }
  };

  explicit DataflowAnalyzer(FunctionOpInterface func);
  DataflowAnalyzer(DataflowAnalyzer&&) noexcept;
  DataflowAnalyzer& operator=(DataflowAnalyzer&&) noexcept;
  ~DataflowAnalyzer();

  FailureOr<AnalysisResult> analyze(DenseI64ArrayAttr grid);

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

llvm::StringRef toString(Space space);
llvm::StringRef toString(PlaceKind kind);
llvm::StringRef toString(Dir dir);
llvm::StringRef toString(OpKind kind);
llvm::StringRef toString(Role role);
llvm::StringRef toString(Branch branch);
llvm::StringRef toString(Guard guard);
llvm::StringRef toString(DataflowDeferredReason reason);

} // namespace triton
} // namespace mlir

#endif
