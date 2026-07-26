// LJX — reserved virtual-address arena, compressed references, allocators.
// Layer 0.
//
// Design (ARCHITECTURE.md §4.2): one contiguous VA range is reserved below
// 2^47 at startup and all GC memory is carved from it. This upgrades LuaJIT's
// two fragile schemes (MAP_32BIT/probing heap placement in non-GC64 mode; the
// "hope mmap stays under 47 bits" assumption of GC64) into a guarantee, and
// lets object-internal references be 32-bit arena offsets — LuaJIT's proven
// cache-density win, generalized (V8-style pointer compression).
#pragma once

#include <cstddef>
#include <cstdint>
#include <span>

#include "ljx/core/Types.hpp"

namespace ljx::core {

// ---------------------------------------------------------------------------
// C_VirtualArena — reserve once, commit on demand, decommit on release.
// 2 MB-granule aligned so the heap is huge-page friendly.
// ---------------------------------------------------------------------------

class C_VirtualArena {
public:
    // Reserves uReserveBytes of VA (no commit) with every address < 2^47.
    // uReserveBytes must not exceed kMaxArenaReserve (declared below): the
    // 32-bit granule-scaled ref width is a hard contract, checked here so the
    // two constants cannot drift apart. Fails hard (returns false) rather
    // than degrade the boxing or ref-width contracts.
    [[nodiscard]] bool Reserve(std::size_t uReserveBytes) noexcept;
    void ReleaseAll() noexcept;

    [[nodiscard]] void* CommitGranules(std::size_t uGranuleCount) noexcept;
    void Decommit(void* pBlock, std::size_t uBytes) noexcept;

    [[nodiscard]] std::uintptr_t BaseAddress() const noexcept { return m_uBase; }
    [[nodiscard]] bool Contains(const void* pAddr) const noexcept;

private:
    std::uintptr_t m_uBase = 0;
    std::size_t m_uReserved = 0;
    std::size_t m_uCommitted = 0;
};

// ---------------------------------------------------------------------------
// Compressed references.
//
// GcRef_t / MRef_t are 32-bit GRANULE indices from the owning universe's
// arena base: every referenced address is 16-byte aligned (the GC allocation
// granule; vector blocks are likewise 16-aligned), so a 32-bit index scaled
// by kCompressedRefShift addresses 2^36 bytes = 64 GB of arena — which is why
// kMaxArenaReserve below caps C_VirtualArena reservations. The scaled add is
// still a single addressing-mode operand on both ISAs ([base + idx*16] on
// x86-64; ADD extended-register with LSL #4 on AArch64).
//
// Refs appear inside GC objects (headers, table nodes, proto constants…)
// where density matters; TValue_t payloads still carry full 47-bit pointers
// so value fast paths never pay a base-add for tag-stripped access.
//
// Deliberately not implicit-converting: decompression requires the arena
// base, passed explicitly (kept in the pinned context register at runtime).
// ---------------------------------------------------------------------------

inline constexpr unsigned kCompressedRefShift = 4;   // 16-byte referent alignment
inline constexpr std::size_t kMaxArenaReserve =
    (std::size_t{1} << (32 + kCompressedRefShift));  // 64 GB: the ref-width contract

struct GcRef_t {
    std::uint32_t uIndex = 0;   // granule index, not a byte offset

    [[nodiscard]] constexpr bool IsNull() const noexcept { return uIndex == 0; }
    [[nodiscard]] constexpr bool operator==(const GcRef_t&) const noexcept = default;
};

struct MRef_t {
    std::uint32_t uIndex = 0;   // granule index, not a byte offset

    [[nodiscard]] constexpr bool IsNull() const noexcept { return uIndex == 0; }
    [[nodiscard]] constexpr bool operator==(const MRef_t&) const noexcept = default;
};

static_assert(IsFrozenLayout<GcRef_t> && sizeof(GcRef_t) == 4);
static_assert(IsFrozenLayout<MRef_t> && sizeof(MRef_t) == 4);

[[nodiscard]] LJX_FORCEINLINE void* RefToPtr(std::uintptr_t uArenaBase, MRef_t rRef) noexcept {
    return reinterpret_cast<void*>(
        uArenaBase + (std::uintptr_t{rRef.uIndex} << kCompressedRefShift));
}

// Precondition: pPtr is 16-byte aligned and within kMaxArenaReserve of the
// base (debug builds assert both; release builds rely on the allocator).
[[nodiscard]] LJX_FORCEINLINE MRef_t PtrToRef(std::uintptr_t uArenaBase, const void* pPtr) noexcept {
    return MRef_t{static_cast<std::uint32_t>(
        (reinterpret_cast<std::uintptr_t>(pPtr) - uArenaBase) >> kCompressedRefShift)};
}

// ---------------------------------------------------------------------------
// C_SegregatedAllocator — GC-object allocation front end.
//
// Small GC objects bump-allocate from size-class regions inside arena blocks
// (removing the ~10-instruction smallbin path from the hottest sites: strings,
// tables, nodes); large/vector allocations fall back to a dlmalloc-class
// backend. The sized-free contract of lua_Alloc is exploited, not ignored:
// every Free carries the old size, keeping gc.total accounting exact.
// ---------------------------------------------------------------------------

class C_SegregatedAllocator {
public:
    explicit C_SegregatedAllocator(C_VirtualArena& arena) noexcept : m_pArena(&arena) {}

    [[nodiscard]] void* AllocGcObject(std::size_t uBytes) noexcept;   // bump path
    [[nodiscard]] void* AllocVector(std::size_t uBytes) noexcept;     // backend path
    [[nodiscard]] void* ReallocVector(void* pOld, std::size_t uOldBytes,
                                      std::size_t uNewBytes) noexcept;
    void Free(void* pBlock, std::size_t uBytes) noexcept;

    [[nodiscard]] GcSize_t TotalAllocated() const noexcept { return m_uTotal; }

private:
    C_VirtualArena* m_pArena;
    GcSize_t m_uTotal = 0;
};

}  // namespace ljx::core
