// LJX — the trace recorder: bytecode → IR, one instruction at a time.
// Layer 5.
//
// Structure kept from LuaJIT (the contracts are the architecture):
//  * the recorder runs ONE instruction BEHIND execution, peeking at runtime
//    values (copies taken before side effects) and patching decisions that
//    depend on not-yet-computed results via EPostProc fixups;
//  * a slot map mirrors the interpreter stack exactly — one TRef_t per slot,
//    IR type in the top byte, frame/continuation flags on the right slots;
//    CheckSlotMap() is the executable spec (debug builds);
//  * recording runs inside a protected call; aborts throw a trace error and
//    unwind — a user error thrown by a metamethod mid-recording aborts the
//    trace for free;
//  * demand-driven narrowing (int32 index expressions) via a backpropagation
//    stack machine with a small round-robin cache; predictive narrowing for
//    numeric-for induction variables backed by a scalar-evolution cache;
//  * per-opcode record handlers dispatched from a table (the 2000-line switch,
//    factored; same semantics).
#pragma once

#include <cstdint>

#include "ljx/jit/Fold.hpp"
#include "ljx/jit/Ir.hpp"
#include "ljx/jit/Snapshot.hpp"
#include "ljx/vm/Bytecode.hpp"
#include "ljx/vm/Interpreter.hpp"

namespace ljx::jit {

enum class ETraceError : std::uint8_t {
    None,
    RecordDepth, TraceOverflow, SnapOverflow, StackOverflow,
    LeftLoop, InnerLoop, DownRecursion, NyiBytecode, NyiBuiltin,
    GuardAlwaysFails, TypeInstability, RetryAssembly, MachineCodeLimit,
    Blacklisted,
};

// Fixup mode for decisions made one instruction behind execution.
enum class EPostProc : std::uint8_t {
    None, FixComparison, FixGuard, FixBool, FixConst, RetryBuiltin,
};

// How a finished trace continues (stored on the trace, drives linking).
enum class ETraceLink : std::uint8_t {
    None, Root, Loop, TailRecursion, UpRecursion, DownRecursion,
    Interpreter, Return, Stitch,
};

// Scalar-evolution cache for numeric-for loops (start/stop/step facts).
struct ScalarEvolution_t {
    const vm::BcIns_t* pForPc = nullptr;
    IrRef_t rStart = 0, rStop = 0, rStep = 0;
    EIrType eType{};
    std::uint8_t uDirection = 0;
};

class C_TraceRecorder {
public:
    static constexpr std::uint32_t kMaxSlots = 255;
    static constexpr std::uint32_t kMaxRecordedIns = 4000;  // param: maxrecord

    C_TraceRecorder(vm::C_Universe& uni, C_IrBuffer& irBuf) noexcept;

    // --- lifecycle (driven by the JIT engine state machine) -----------------
    void StartRoot(vm::C_GcProto* pProto, const vm::BcIns_t* pStartPc);
    void StartSide(std::uint32_t uParentTrace, std::uint32_t uExitNumber);

    // Record one bytecode (called from the recording dispatch stub, one
    // instruction behind the interpreter). Throws ETraceError on abort.
    void RecordInstruction(const vm::BcIns_t* pPc, const vm::TValue_t* pBase);

    // Trace-end conditions → hand off to optimizer/assembler.
    [[nodiscard]] ETraceLink PendingLink() const noexcept { return m_eLink; }

    // --- slot map -----------------------------------------------------------
    [[nodiscard]] TRef_t Slot(std::uint32_t uSlot) const noexcept {
        return m_pSlots[uSlot];
    }
    void SetSlot(std::uint32_t uSlot, TRef_t trValue) noexcept;
    void CheckSlotMap(const vm::TValue_t* pBase) const noexcept;  // debug spec

    // --- IR emission (everything flows through fold) ------------------------
    [[nodiscard]] TRef_t Emit(EIrOp eOp, std::uint8_t uTypeBits, TRef_t trOp1, TRef_t trOp2);
    [[nodiscard]] FoldState_t& FoldState() noexcept { return m_FoldState; }
    [[nodiscard]] C_IrBuffer& Ir() noexcept { return *m_pIr; }
    [[nodiscard]] C_SnapshotWriter& Snapshots() noexcept { return m_SnapWriter; }

    // Force a snapshot after operations whose effects an exit could observe.
    void RequireSnapshot() noexcept { m_bNeedSnapshot = true; }

    // --- narrowing ----------------------------------------------------------
    [[nodiscard]] TRef_t NarrowIndex(TRef_t trValue);       // FP → int32 index
    [[nodiscard]] TRef_t NarrowForLoop(const vm::BcIns_t* pForPc);

    [[noreturn]] void Abort(ETraceError eError);

private:
    // Per-opcode record handlers (table-dispatched; grouped by family).
    using RecordHandler_f = void (C_TraceRecorder::*)(const vm::BcIns_t*);
    static const RecordHandler_f s_vHandlers[];

    vm::C_Universe* m_pUniverse;
    C_IrBuffer* m_pIr;
    C_FoldEngine m_Fold;
    C_SnapshotWriter m_SnapWriter;
    FoldState_t m_FoldState{};

    TRef_t* m_pSlots = nullptr;       // slot map: mirror of the Lua stack
    std::uint32_t m_uBaseSlot = 0;
    std::uint32_t m_uMaxSlot = 0;
    std::uint32_t m_uFrameDepth = 0;

    ScalarEvolution_t m_Scev{};
    EPostProc m_ePostProc = EPostProc::None;
    ETraceLink m_eLink = ETraceLink::None;
    std::uint32_t m_uParentTrace = 0;
    std::uint32_t m_uParentExit = 0;
    bool m_bNeedSnapshot = false;

    // Narrowing backprop cache (round-robin, small by design).
    struct BackpropEntry_t {
        IrRef_t rKey = 0;
        IrRef_t rValue = 0;
        std::uint16_t uMode = 0;
    };
    static constexpr std::uint32_t kBackpropSlots = 16;
    BackpropEntry_t m_vBackprop[kBackpropSlots]{};
};

// ---------------------------------------------------------------------------
// C_LoopOptimizer — unrolling by copy-substitution (NOT classic LICM; see the
// design essay carried over from lj_opt_loop.c): the recorded body is
// re-emitted through the FULL fold/CSE pipeline with substituted operands, so
// invariant guards hoist via CSE and store→load forwarding crosses the back
// edge; loop-carried refs become PHIs placed BELOW the body. Type-unstable
// PHIs get conversion fixes or roll everything back and let the RECORDER keep
// unrolling instead (budgeted retry — contract with C_TraceRecorder).
// ---------------------------------------------------------------------------

class C_LoopOptimizer {
public:
    static constexpr std::uint32_t kMaxUnrollRetries = 4;
    static constexpr std::uint32_t kMaxPhis = 64;

    [[nodiscard]] ETraceError Run(C_TraceRecorder& rec);

private:
    void SubstituteAndReemit(C_TraceRecorder& rec);
    void PlacePhis(C_TraceRecorder& rec);
    void Undo(C_TraceRecorder& rec, IrRef_t rSavedTop) noexcept;
};

// C_SinkOptimizer — allocation sinking: unaliased short-lived allocations
// (with constant-keyed stores) move into snapshots and materialize only on
// exits. The sink marks, the assembler's skip logic, and the deopt-side
// rematerialization form one contract and live in one module.
class C_SinkOptimizer {
public:
    [[nodiscard]] bool Run(C_IrBuffer& irBuf, C_SnapshotWriter& snapWriter);
};

}  // namespace ljx::jit
