// LJX — the continuation-passing tail-call interpreter and the VM universe.
// Layer 3.
//
// One C++ function per opcode; every dispatch is a guaranteed tail call
// through the dispatch table, so each handler keeps the pinned state in
// argument registers and ends in its OWN indirect jump (per-opcode BTB entry —
// the replicated-dispatch property of the assembly VM). preserve_none frees
// ~all registers for VM state. See ARCHITECTURE.md §6 for the five hard rules
// (musttail, replication, lazy cracking, slow-path discipline, dual tables).
#pragma once

#include <atomic>
#include <cstdint>
#include <type_traits>

#include "ljx/core/Memory.hpp"
#include "ljx/vm/Bytecode.hpp"
#include "ljx/vm/Frame.hpp"
#include "ljx/vm/Object.hpp"

namespace ljx::vm {

class C_Universe;

// ---------------------------------------------------------------------------
// The handler ABI — this signature IS the interpreter's register file and,
// symmetrically, the JIT's deoptimization target contract. Do not widen it.
//
//   pBase   — current frame base            (asm VM: BASE)
//   pPc     — NEXT instruction              (asm VM: PC)
//   uRa     — decoded A operand             (asm VM: RA)
//   uRd     — decoded D (B/C cracked lazily in the handler)   (asm VM: RD)
//   pUni    — the universe: dispatch tables, globals, JIT, hot counters, all
//             addressable at constant offsets (asm VM: DISPATCH register)
//   pKBase  — current proto's constant base (asm VM: KBASE); must be
//             rematerialized after every control transfer that changes the
//             function (call/return/deopt) — same invariant as LuaJIT.
// ---------------------------------------------------------------------------

using BcHandler_f = void (*)(TValue_t* pBase, const BcIns_t* pPc, std::uint64_t uRa,
                             std::uint64_t uRd, C_Universe* pUni,
                             const TValue_t* pKBase);

enum class EDispatchMode : std::uint8_t {
    Normal      = 0,
    Recording   = 1 << 0,  // JIT trace recorder observes every instruction
    HookLine    = 1 << 1,
    HookCount   = 1 << 2,  // rides the same per-instruction variant as HookLine
    HookCall    = 1 << 3,
    HookReturn  = 1 << 4,
    Profiling   = 1 << 5,
    JitOff      = 1 << 6,
};

// Mode values compose as flag sets.
[[nodiscard]] constexpr EDispatchMode operator|(EDispatchMode eA, EDispatchMode eB) noexcept {
    return static_cast<EDispatchMode>(static_cast<std::uint8_t>(eA) |
                                      static_cast<std::uint8_t>(eB));
}
[[nodiscard]] constexpr EDispatchMode operator&(EDispatchMode eA, EDispatchMode eB) noexcept {
    return static_cast<EDispatchMode>(static_cast<std::uint8_t>(eA) &
                                      static_cast<std::uint8_t>(eB));
}
[[nodiscard]] constexpr bool HasMode(EDispatchMode eSet, EDispatchMode eFlag) noexcept {
    return (eSet & eFlag) != EDispatchMode::Normal;
}

// VM execution states (negative complemented values in the vmstate field).
enum class EVmState : std::uint8_t {
    Interp, CCode, Gc, Exit, Record, Opt, Asm,
};

// ---------------------------------------------------------------------------
// C_DispatchTable — dynamic + static pair.
//
// Instrumentation (hooks, recording, profiling) NEVER adds per-instruction
// checks: it swaps dynamic entries; stubs re-dispatch through the static copy.
// Entries are atomics because lua_sethook is async-signal-safe: relaxed loads
// on the hot path (free on x86/ARM64), release stores on mode changes.
// ---------------------------------------------------------------------------

class C_DispatchTable {
public:
    static constexpr std::uint32_t kBuiltinOps = 64;   // synthetic fast-function ops
    static constexpr std::uint32_t kEntries =
        static_cast<std::uint32_t>(EBcOp::Count_) + kBuiltinOps;

    [[nodiscard]] LJX_FORCEINLINE BcHandler_f Dynamic(std::uint32_t uOp) const noexcept {
        return m_vDynamic[uOp].load(std::memory_order_relaxed);
    }
    [[nodiscard]] LJX_FORCEINLINE BcHandler_f Static(std::uint32_t uOp) const noexcept {
        return m_vStatic[uOp];
    }

    // Recomputes the dynamic table for the given mode set (hot-count variants,
    // record/hook/profile stubs); release-ordered publication.
    void SetMode(EDispatchMode eMode) noexcept;

private:
    std::atomic<BcHandler_f> m_vDynamic[kEntries]{};
    BcHandler_f m_vStatic[kEntries]{};
};

// ---------------------------------------------------------------------------
// C_HotCountTable — hashed, shared, deliberately tiny.
// 3-instruction decrement in loop/call headers; trigger on borrow. Collisions
// provide free aging and cost ~nothing (LuaJIT-measured); constants preserved.
// ---------------------------------------------------------------------------

class C_HotCountTable {
public:
    static constexpr std::uint32_t kSlots = 64;
    static constexpr std::uint16_t kLoopCost = 2;
    static constexpr std::uint16_t kCallCost = 1;
    static constexpr std::uint16_t kHotLoopThreshold = 56;  // ≙ 56 loops / 112 calls
    static constexpr std::uint16_t kArmedValue = kHotLoopThreshold * kLoopCost;

    // Counters must start ARMED: a zero-initialized table would report every
    // site hot on its first decrement.
    constexpr C_HotCountTable() noexcept {
        for (std::uint16_t& uCount : m_vCounts) uCount = kArmedValue;
    }

    [[nodiscard]] LJX_FORCEINLINE static std::uint32_t SlotFor(const BcIns_t* pPc) noexcept {
        // Cheap PC hash — collisions between sites are accepted by design.
        return static_cast<std::uint32_t>(
                   (reinterpret_cast<std::uintptr_t>(pPc) >> 2)) & (kSlots - 1);
    }

    // True when the counter borrows (site became hot). Borrow is detected on
    // the PRE-decrement value so it stays correct for counters re-armed with
    // large penalty values (up to kPenaltyMax) after trace aborts — compiles
    // to sub + jb, the same 3-instruction budget as the assembly VM.
    [[nodiscard]] LJX_FORCEINLINE bool DecrementLoop(const BcIns_t* pPc) noexcept {
        std::uint16_t& uCount = m_vCounts[SlotFor(pPc)];
        const std::uint16_t uOld = uCount;
        uCount = static_cast<std::uint16_t>(uOld - kLoopCost);
        return uOld < kLoopCost;
    }
    [[nodiscard]] LJX_FORCEINLINE bool DecrementCall(const BcIns_t* pPc) noexcept {
        std::uint16_t& uCount = m_vCounts[SlotFor(pPc)];
        const std::uint16_t uOld = uCount;
        uCount = static_cast<std::uint16_t>(uOld - kCallCost);
        return uOld < kCallCost;
    }

    // Re-arm with an explicit value: kArmedValue after a compiled trace, or a
    // penalty value (doubling + random bits) after an abort.
    void Reset(const BcIns_t* pPc, std::uint16_t uValue) noexcept {
        m_vCounts[SlotFor(pPc)] = uValue;
    }

private:
    std::uint16_t m_vCounts[kSlots]{};
};

// ---------------------------------------------------------------------------
// C_Universe — the one-allocation VM (LuaJIT's GG_State, formalized).
//
// Main thread + global state + JIT state + hot counters + dispatch tables in
// one huge-page block, so ONE pinned register (pUni) addresses everything hot
// with [reg+disp32]. Every offset the interpreter/JIT bakes in is exported as
// a constexpr and static_assert-pinned in vm/UniverseLayout.hpp (generated).
// ---------------------------------------------------------------------------

class C_IVmEventSink;

class C_Universe {
public:
    // --- creation ----------------------------------------------------------
    [[nodiscard]] static C_Universe* Create() noexcept;   // reserves arena, seeds PRNG
    void Destroy() noexcept;

    // --- pinned-register accessors (constant offsets from `this`) ----------
    [[nodiscard]] C_LuaThread* MainThread() noexcept;
    [[nodiscard]] C_DispatchTable& Dispatch() noexcept { return m_Dispatch; }
    [[nodiscard]] C_HotCountTable& HotCounts() noexcept { return m_HotCounts; }
    [[nodiscard]] std::uintptr_t ArenaBase() const noexcept { return m_uArenaBase; }

    // Shared sentinels (branch removers — see rt/ and gc/):
    [[nodiscard]] const TValue_t* NilSentinel() const noexcept { return &m_tvNil; }
    [[nodiscard]] const TableNode_t* NilNode() const noexcept { return &m_NilNode; }

    // --- vmstate contract ---------------------------------------------------
    // NEGATIVE values are ~EVmState; NON-NEGATIVE values are the executing
    // trace number — the two ranges are disjoint by construction, which is
    // what lets the profiler signal handler classify with one sign test.
    // Atomic (not volatile): read from a signal handler.
    void SetVmState(EVmState eState) noexcept {
        m_nVmState.store(~static_cast<std::int32_t>(eState), std::memory_order_relaxed);
    }
    void SetVmTrace(std::uint16_t uTraceNumber) noexcept {
        m_nVmState.store(static_cast<std::int32_t>(uTraceNumber), std::memory_order_relaxed);
    }

    // --- observation (jit.dump/-jv equivalents, trace diagnostics) ----------
    void SetEventSink(C_IVmEventSink* pSink) noexcept;   // see vm/VmEvent.hpp

    // NOTE: all data members share one access level — C_Universe must remain
    // standard-layout (asserted below): the interpreter and JIT address these
    // fields as [contextReg + disp32] with static_assert-pinned offsets.
    std::atomic<std::int32_t> m_nVmState{~static_cast<std::int32_t>(EVmState::Interp)};
    core::MRef_t m_rJitBase;   // non-null iff a trace is executing (GC interlock)
    C_DispatchTable m_Dispatch;
    C_HotCountTable m_HotCounts;
    TValue_t m_tvNil;
    TableNode_t m_NilNode;
    std::uintptr_t m_uArenaBase = 0;
    C_IVmEventSink* m_pEventSink = nullptr;
    // … global state, string interner, GC, JIT state follow at fixed offsets.
};
static_assert(std::is_standard_layout_v<C_Universe>,
              "offset-addressed from the pinned context register");

// ---------------------------------------------------------------------------
// C_Interpreter — the handler family and entry points.
//
// Handlers are template instantiations over the opcode (one source of truth
// for operand-family variants: `if constexpr` replaces DynASM's parameterized
// macro families). All slow paths are separate noinline functions that
// RE-ENTER via tail call — never a call that returns into a handler.
// ---------------------------------------------------------------------------

class C_Interpreter {
public:
    // Call into bytecode from C++ (establishes a C frame; resume/yield-aware).
    static std::int32_t Call(C_LuaThread* pThread, TValue_t* pFunc, std::int32_t nArgs,
                             std::int32_t nResults) noexcept;
    static std::int32_t ProtectedCall(C_LuaThread* pThread, TValue_t* pFunc,
                                      std::int32_t nArgs, std::int32_t nResults,
                                      std::uint64_t uErrFunc) noexcept;
    static std::int32_t Resume(C_LuaThread* pThread, std::int32_t nArgs) noexcept;

    // The per-opcode handler template; explicit instantiations populate the
    // static dispatch table (definition in vm/Interpreter.cpp).
    template <EBcOp TOp>
    LJX_PRESERVE_NONE static void OpHandler(TValue_t* pBase, const BcIns_t* pPc,
                                            std::uint64_t uRa, std::uint64_t uRd,
                                            C_Universe* pUni, const TValue_t* pKBase);

    // Deopt re-entry: restores interpreter state from a trace exit and tail
    // calls the handler for the restored PC (contract with jit/Snapshot.hpp).
    LJX_PRESERVE_NONE static void ReenterFromExit(TValue_t* pBase, const BcIns_t* pPc,
                                                  C_Universe* pUni);
};

}  // namespace ljx::vm
