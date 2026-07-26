// LJX — trace lifecycle: hotness, recording orchestration, the trace cache,
// side traces, stitching, penalties and blacklisting.
// Layer 5.
//
// The tuning constants are LuaJIT's, preserved NUMERICALLY — they embody
// years of production tuning (exposed as constexpr config, not "cleaned up").
#pragma once

#include <cstdint>

#include "ljx/jit/Backend.hpp"
#include "ljx/jit/Recorder.hpp"
#include "ljx/vm/Object.hpp"

namespace ljx::jit {

// ---------------------------------------------------------------------------
// C_GcTrace — a finished trace, first-class GC object: compacted IR +
// snapshots + machine-code metadata + the trace-graph links. Allocated as ONE
// block (header, IR, snapshots, entry map) copied out of the shared build
// buffers on completion.
// ---------------------------------------------------------------------------

class C_GcTrace {
public:
    vm::GcHeader_t m_Header;
    std::uint16_t m_uSnapshotCount = 0;
    IrRef_t m_rInsCount = 0;          // biased
    IrRef_t m_rConstFloor = 0;        // biased
    core::GcRef_t m_rGcList;
    IrIns_t* m_pIr = nullptr;         // stored pre-biased
    Snapshot_t* m_pSnapshots = nullptr;
    SnapEntry_t* m_pEntryMap = nullptr;
    std::uint32_t m_uEntryMapCount = 0;

    core::GcRef_t m_rStartProto;
    core::MRef_t m_rStartPc;
    vm::BcIns_t m_insOriginal;        // pre-patch bytecode (unpatch on flush)

    std::uint8_t* m_pMachineCode = nullptr;
    std::uint32_t m_uMachineCodeSize = 0;
    std::uint32_t m_uLoopOffset = 0;

    // Trace graph (16-bit ids; 64K traces max — trace anchor width contract).
    std::uint16_t m_uTraceNumber = 0;
    std::uint16_t m_uLink = 0;        // linked trace / self for loops / 0
    std::uint16_t m_uRoot = 0;        // 0 for root traces
    std::uint16_t m_uNextRoot = 0;    // per-proto root chain
    std::uint16_t m_uNextSide = 0;    // per-root side chain
    std::uint8_t m_uChildCount = 0;
    ETraceLink m_eLinkType = ETraceLink::None;
    std::uint16_t m_uSpAdjust = 0;
    std::uint8_t m_uTopSlot = 0;      // stack already checked up to here
};
static_assert(core::IsFrozenLayout<C_GcTrace>,
              "trace headers are GC objects addressed by baked offsets");

// ---------------------------------------------------------------------------
// Penalties & blacklisting (numerically preserved).
// ---------------------------------------------------------------------------

inline constexpr std::uint32_t kPenaltyMin = 36 * 2;
inline constexpr std::uint32_t kPenaltyMax = 60000;
inline constexpr std::uint32_t kPenaltyRandomBits = 4;   // de-phase-locking
inline constexpr std::uint32_t kPenaltySlots = 64;       // round-robin cache
inline constexpr std::uint32_t kHotSideExitThreshold = 10;  // param: hotexit
inline constexpr std::uint32_t kMaxSideTracesPerRoot = 100; // param: maxside

struct HotPenalty_t {
    const vm::BcIns_t* pPc = nullptr;
    std::uint16_t uValue = 0;
    ETraceError eReason = ETraceError::None;
};

// ---------------------------------------------------------------------------
// C_TraceCache — trace array + per-proto anchors + flush protocol.
// ---------------------------------------------------------------------------

class C_TraceCache {
public:
    [[nodiscard]] C_GcTrace* Lookup(std::uint16_t uTraceNumber) noexcept;
    [[nodiscard]] std::uint16_t Insert(C_GcTrace* pTrace);

    // Root-trace patching: Loop/ForL/FuncF → their J-variants with the trace
    // number in D (zero-overhead entry); original kept for unpatching.
    void PatchTraceEntry(vm::C_GcProto* pProto, vm::BcIns_t* pPc, std::uint16_t uTraceNumber);

    // Full flush: unpatch all entry points, free ALL machine code, clear the
    // penalty cache and exit-stub-group cache (stub addresses die with the
    // areas), zero stale stitch links.
    void FlushAll(vm::C_Universe& uni);

private:
    C_GcTrace** m_pTraces = nullptr;
    std::uint16_t m_uCount = 0;
    std::uint16_t m_uCapacity = 0;
};

// ---------------------------------------------------------------------------
// C_JitEngine — pipeline orchestration (the trace-compiler state machine).
//
//   Idle → Record → (End) → Optimize → Assemble → Publish
// with abort/retry edges: penalty & blacklist on abort; retry-with-new-mcode-
// area; restart-as-down-recursion. Runs inside a protected call; a Lua error
// thrown during recording unwinds through here and aborts the trace cleanly.
// ---------------------------------------------------------------------------

enum class EJitState : std::uint8_t {
    Idle, Start, Record, End, Assemble, Error,
};

class C_JitEngine {
public:
    explicit C_JitEngine(vm::C_Universe& uni) noexcept;

    // Entry from the interpreter's hot-count underflow (loop or call site).
    void OnHotLoop(vm::C_LuaThread* pThread, const vm::BcIns_t* pPc);
    void OnHotCall(vm::C_LuaThread* pThread, const vm::BcIns_t* pPc);

    // Entry from a trace exit (may start a side trace when the exit is hot,
    // or blacklist the exit after repeated failures).
    void OnTraceExit(vm::C_LuaThread* pThread, const ExitState_t& exitState);

    // Trace stitching: O(1) trace→trace hand-off across unrecordable calls.
    void StitchContinuation(vm::C_LuaThread* pThread, std::uint16_t uInvokingTrace);

    [[nodiscard]] C_TraceCache& Cache() noexcept { return m_Cache; }
    [[nodiscard]] EJitState State() const noexcept { return m_eState; }

    void SetEnabled(bool bEnabled) noexcept;
    void Flush() noexcept;

private:
    void RunStateMachine(vm::C_LuaThread* pThread);
    void PenalizeStart(const vm::BcIns_t* pPc, ETraceError eReason);  // ×2 + rnd
    void BlacklistStart(vm::C_GcProto* pProto, vm::BcIns_t* pPc);     // I-variant

    vm::C_Universe* m_pUniverse;
    C_TraceCache m_Cache;
    C_IrBuffer m_IrBuffer;             // shared build buffer, reused per trace
    C_TraceRecorder m_Recorder;
    C_LoopOptimizer m_LoopOpt;
    C_SinkOptimizer m_SinkOpt;
    C_MachineCodeArena m_McArena;
    HotPenalty_t m_vPenalties[kPenaltySlots]{};
    std::uint8_t m_uPenaltyCursor = 0;
    EJitState m_eState = EJitState::Idle;
};

}  // namespace ljx::jit
