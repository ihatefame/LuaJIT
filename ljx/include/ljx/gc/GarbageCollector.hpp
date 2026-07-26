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
    C_GarbageCollector() noexcept = default;
    void Init(vm::C_Universe& uni, core::C_SegregatedAllocator& alloc) noexcept;

    // --- allocation entry (inlined trigger: one compare) --------------------
    [[nodiscard]] LJX_FORCEINLINE bool NeedsStep() const noexcept {
        return m_pAllocatorPublic->TotalAllocated() >= m_Stats.uThreshold;
    }
    // v1 collector is a precise stop-the-world mark-sweep invoked ONLY from
    // interpreter safe points (function entry / allocation ops with a synced
    // stack top). The incremental/generational engine replaces Step()'s body
    // without changing any call site.
    void Step() noexcept { CollectNow(); }
    void CollectNow() noexcept;
    void FullCollection() noexcept { CollectNow(); }
    void SetMode(EGcMode eMode) noexcept { m_eMode = eMode; }

    // --- write barriers -----------------------------------------------------
    // v1 stop-the-world: no mutation can interleave marking, so all barriers
    // are no-ops. The declarations (and every call site the interpreter
    // plants) are the real deliverable — the incremental collector fills the
    // bodies in without touching handlers.
    LJX_FORCEINLINE void BarrierBackTable(vm::C_GcTable*) noexcept {}
    LJX_FORCEINLINE void BarrierForward(vm::GcHeader_t*, const vm::GcHeader_t*) noexcept {}
    LJX_FORCEINLINE void BarrierUpvalue(vm::C_GcUpvalue*, const vm::TValue_t&) noexcept {}

    // --- object lifecycle ---------------------------------------------------
    // Allocates + links the object into the all-objects list (non-strings).
    [[nodiscard]] void* AllocObject(vm::EGcObjectType eType, std::size_t uBytes);
    void FixObject(vm::GcHeader_t* pHeader) noexcept;   // pin (never collected)

    [[nodiscard]] const GcStats_t& Stats() const noexcept { return m_Stats; }
    [[nodiscard]] EGcState State() const noexcept { return m_eState; }

private:
    void MarkValue(const vm::TValue_t& tvValue) noexcept;
    void MarkObject(vm::GcHeader_t* pHeader) noexcept;
    void TraverseGrays() noexcept;
    void SweepObjects() noexcept;
    [[nodiscard]] std::size_t ObjectSize(const vm::GcHeader_t* pHeader) const noexcept;

    vm::C_Universe* m_pUniverse = nullptr;
    core::C_SegregatedAllocator* m_pAllocatorPublic = nullptr;
    core::GcRef_t m_rAllObjects;      // every non-string GC object (v1 sweep list)
    vm::GcHeader_t** m_pGrayStack = nullptr;   // collector-internal scratch
    std::size_t m_uGrayCount = 0, m_uGrayCapacity = 0;
    GcStats_t m_Stats{};
    EGcState m_eState = EGcState::Pause;
    EGcMode m_eMode = EGcMode::Incremental;
};

}  // namespace ljx::gc
