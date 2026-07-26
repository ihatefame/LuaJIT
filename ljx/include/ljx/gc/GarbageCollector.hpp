// LJX — incremental (and optionally generational) arena garbage collector.
// Layer 2.
//
// What is kept from LuaJIT verbatim (ARCHITECTURE.md §8):
//  * tri-color invariant with the exact barrier taxonomy — BACK barrier for
//    tables (gray-again list ≡ remembered set), FORWARD barrier for the rest,
//    make-white degradation outside propagate/atomic, threads never black,
//    and every documented elision rule (stack slots, roots, fresh objects,
//    self-stores). Table-store fast path stays ONE load + test + branch.
//  * trace interlocks: atomic/finalize never run while a trace executes;
//    on-trace GC steps force a trace exit instead.
//  * finalizer discipline: mark-then-resurrect-once, scheduled (never called
//    from sweep), errors to a VM event, hooks/threshold saved around the call.
//  * incremental pacing: stepmul/pause percentage model, exact byte accounting.
//
// What is new:
//  * arena heap with per-arena MARK BITMAPS — sweep is popcount word-scans and
//    whole-arena frees; the intrusive ALL-OBJECTS sweep list disappears.
//    (The per-object gclist threading for gray/gray-again sets is KEPT — see
//    below — and GcHeader_t::rNextGc remains as the per-type chain field:
//    string intern chains, the finalizer registry.)
//  * generational mode: sticky mark bits + minor collections that re-traverse
//    only the remembered (gray-again) set. Same barriers, same cost model.
//  * bounded atomic phase: per-thread dirty flags, incremental weak clearing,
//    a dedicated finalizer registry.
//
// Gray-set representation decision (normative): gray and gray-again sets are
// INTRUSIVE chains threaded through the objects' m_rGcList field (the
// offset-aliased field pinned in vm/Object.hpp), exactly as in LuaJIT — an
// external-vector worklist would make the table back-barrier allocating,
// breaking the one-load+test+branch barrier budget. Mark BITS live in arena
// bitmaps; list MEMBERSHIP lives in the objects.
#pragma once

#include <cstddef>
#include <cstdint>

#include "ljx/core/Memory.hpp"
#include "ljx/vm/Object.hpp"

namespace ljx::gc {

enum class EGcState : std::uint8_t {
    Pause,
    Propagate,
    Atomic,
    SweepString,
    Sweep,
    Finalize,
};

enum class EGcMode : std::uint8_t {
    Incremental,
    Generational,
};

struct GcStats_t {
    core::GcSize_t uTotalBytes = 0;      // exact (sized alloc/free contract)
    core::GcSize_t uThreshold = 0;
    core::GcSize_t uEstimate = 0;
    std::uint64_t uMinorCollections = 0;
    std::uint64_t uMajorCollections = 0;
    std::uint64_t uMaxAtomicNanos = 0;   // bounded-atomic regression metric
};

// Pacing defaults (percent, LuaJIT-compatible semantics).
inline constexpr std::uint32_t kDefaultStepMul = 200;
inline constexpr std::uint32_t kDefaultPause = 200;

// ---------------------------------------------------------------------------
// C_GcArena — a 2 MB heap block: object granules + mark/type bitmaps in the
// block header. Small objects bump-allocate per size class; mark state never
// touches object memory during sweep.
// ---------------------------------------------------------------------------

class C_GcArena {
public:
    static constexpr std::size_t kGranuleShift = 4;   // 16-byte allocation granule

    [[nodiscard]] void* TryBumpAlloc(std::size_t uBytes) noexcept;
    [[nodiscard]] bool MarkGranule(const void* pObject) noexcept;  // returns previously-unmarked
    [[nodiscard]] bool IsMarked(const void* pObject) const noexcept;
    void SweepByBitmap() noexcept;  // popcount scan; may release the whole arena

private:
    std::uint64_t* m_pMarkBits = nullptr;
    std::uint64_t* m_pBlockBits = nullptr;  // allocation-start bits
    std::uint8_t* m_pBumpCursor = nullptr;
    std::uint8_t* m_pBumpLimit = nullptr;
};

// ---------------------------------------------------------------------------
// C_GarbageCollector
// ---------------------------------------------------------------------------

class C_GarbageCollector {
public:
    explicit C_GarbageCollector(vm::C_Universe& uni, core::C_SegregatedAllocator& alloc) noexcept;

    // --- allocation entry (inlined trigger: one compare) --------------------
    [[nodiscard]] LJX_FORCEINLINE bool NeedsStep() const noexcept {
        return m_Stats.uTotalBytes >= m_Stats.uThreshold;
    }
    void Step() noexcept;                       // one incremental quantum
    void StepFromTrace(std::uint32_t uSteps) noexcept;  // may force a trace exit
    void FullCollection() noexcept;
    void SetMode(EGcMode eMode) noexcept;

    // --- write barriers (hot; must stay branch-minimal) ---------------------
    // Tables: back barrier — blacken→gray-again, re-traversed once in atomic.
    LJX_FORCEINLINE void BarrierBackTable(vm::C_GcTable* pTable) noexcept;
    // Everything else: forward barrier (marks target, or whitens source
    // outside propagate/atomic).
    void BarrierForward(vm::GcHeader_t* pSource, const vm::GcHeader_t* pTarget) noexcept;
    // Closed-upvalue store barrier (value-pointer form for handler use).
    void BarrierUpvalue(vm::C_GcUpvalue* pUpval, const vm::TValue_t& tvStored) noexcept;

    // --- object lifecycle ---------------------------------------------------
    [[nodiscard]] void* AllocObject(vm::EGcObjectType eType, std::size_t uBytes);
    void FixObject(vm::GcHeader_t* pHeader) noexcept;   // pin (never collected)

    // --- finalization -------------------------------------------------------
    void RegisterFinalizer(vm::GcHeader_t* pObject) noexcept;
    void RunPendingFinalizers() noexcept;               // scheduled, budgeted

    [[nodiscard]] const GcStats_t& Stats() const noexcept { return m_Stats; }
    [[nodiscard]] EGcState State() const noexcept { return m_eState; }

private:
    // Gray chains: threaded through the objects' m_rGcList field (intrusive,
    // allocation-free pushes — the barrier fast-path budget depends on this).
    core::GcRef_t m_rGrayHead;
    core::GcRef_t m_rGrayAgainHead;   // tables gone back-gray (≡ remembered set)
    GcStats_t m_Stats{};
    EGcState m_eState = EGcState::Pause;
    EGcMode m_eMode = EGcMode::Incremental;
    std::uint8_t m_uCurrentWhite = 0;
};

}  // namespace ljx::gc
