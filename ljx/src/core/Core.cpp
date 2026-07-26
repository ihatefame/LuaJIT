// LJX — core implementations: virtual arena, allocator, PRNG, FP→int kernels.
#include <cstdlib>
#include <cstring>
#include <sys/mman.h>
#include <sys/random.h>

#if defined(__x86_64__)
#include <immintrin.h>
#endif

#include "ljx/core/Memory.hpp"
#include "ljx/core/Types.hpp"

namespace ljx::core {

// ---------------------------------------------------------------------------
// FP → int conversion kernels (bit-identical across interpreter/runtime/JIT).
// ---------------------------------------------------------------------------

std::int32_t NumToInt32Trunc(double flValue) noexcept {
#if defined(__x86_64__)
    return _mm_cvttsd_si32(_mm_set_sd(flValue));  // INDEFINITE 0x80000000 on NaN/ovf
#else
    if (!(flValue >= -2147483648.0 && flValue <= 2147483647.0)) return INT32_MIN;
    return static_cast<std::int32_t>(flValue);
#endif
}

bool NumToInt32Check(double flValue, std::int32_t& nOut) noexcept {
    const std::int32_t nTrunc = NumToInt32Trunc(flValue);
    if (static_cast<double>(nTrunc) != flValue) return false;  // rejects NaN/ovf/frac
    if (nTrunc == 0 && std::bit_cast<std::uint64_t>(flValue) != 0) return false;  // -0
    nOut = nTrunc;
    return true;
}

std::int64_t NumToInt64Trunc(double flValue) noexcept {
#if defined(__x86_64__)
    return _mm_cvttsd_si64(_mm_set_sd(flValue));
#else
    return static_cast<std::int64_t>(flValue);
#endif
}

std::uint64_t NumToUint64Trunc(double flValue) noexcept {
    if (flValue >= 9223372036854775808.0)
        return static_cast<std::uint64_t>(NumToInt64Trunc(flValue - 9223372036854775808.0)) +
               0x8000000000000000ull;
    return static_cast<std::uint64_t>(NumToInt64Trunc(flValue));
}

std::int32_t NumToBit(double flValue) noexcept {
    // BitOp semantics: wrap-around via the 2^52+2^51 magic add.
    const double flMagic = 6755399441055744.0;  // 2^52 + 2^51
    const double flBiased = flValue + flMagic;
    return static_cast<std::int32_t>(
        static_cast<std::uint32_t>(std::bit_cast<std::uint64_t>(flBiased)));
}

// ---------------------------------------------------------------------------
// PRNG — xoshiro256** seeded via splitmix64 from OS entropy.
// ---------------------------------------------------------------------------

static std::uint64_t SplitMix64(std::uint64_t& uState) noexcept {
    std::uint64_t uZ = (uState += 0x9e3779b97f4a7c15ull);
    uZ = (uZ ^ (uZ >> 30)) * 0xbf58476d1ce4e5b9ull;
    uZ = (uZ ^ (uZ >> 27)) * 0x94d049bb133111ebull;
    return uZ ^ (uZ >> 31);
}

bool C_Prng::SeedSecure() noexcept {
    std::uint64_t uSeed = 0;
    if (getrandom(&uSeed, sizeof uSeed, 0) != sizeof uSeed) return false;
    for (std::uint64_t& uWord : m_State.vState) uWord = SplitMix64(uSeed);
    return true;
}

std::uint64_t C_Prng::Next() noexcept {
    std::uint64_t* pS = m_State.vState;
    const std::uint64_t uResult = std::rotl(pS[1] * 5, 7) * 9;
    const std::uint64_t uT = pS[1] << 17;
    pS[2] ^= pS[0];
    pS[3] ^= pS[1];
    pS[1] ^= pS[2];
    pS[0] ^= pS[3];
    pS[2] ^= uT;
    pS[3] = std::rotl(pS[3], 45);
    return uResult;
}

double C_Prng::NextDouble() noexcept {
    return static_cast<double>(Next() >> 11) * (1.0 / 9007199254740992.0);
}

// ---------------------------------------------------------------------------
// C_VirtualArena
// ---------------------------------------------------------------------------

bool C_VirtualArena::Reserve(std::size_t uReserveBytes) noexcept {
    if (uReserveBytes > kMaxArenaReserve || uReserveBytes < kArenaGranule) return false;
    // Reserve PROT_NONE VA; retry with explicit low hints until the whole
    // range sits below 2^47 (the boxing contract).
    for (int nTry = 0; nTry < 16; ++nTry) {
        void* pHint = nTry == 0 ? nullptr
                                : reinterpret_cast<void*>(0x100000000000ull +
                                                          0x20000000000ull * (nTry - 1));
        void* pBase = mmap(pHint, uReserveBytes, PROT_NONE,
                           MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0);
        if (pBase == MAP_FAILED) continue;
        const std::uintptr_t uBase = reinterpret_cast<std::uintptr_t>(pBase);
        if (uBase + uReserveBytes <= (std::uintptr_t{1} << kBoxedPointerBits)) {
            m_uBase = uBase;
            m_uReserved = uReserveBytes;
            m_uCommitted = 0;
            return true;
        }
        munmap(pBase, uReserveBytes);
    }
    return false;
}

void C_VirtualArena::ReleaseAll() noexcept {
    if (m_uBase) munmap(reinterpret_cast<void*>(m_uBase), m_uReserved);
    m_uBase = m_uReserved = m_uCommitted = 0;
}

void* C_VirtualArena::CommitGranules(std::size_t uGranuleCount) noexcept {
    const std::size_t uBytes = uGranuleCount * kArenaGranule;
    if (m_uCommitted + uBytes > m_uReserved) return nullptr;
    void* pBlock = reinterpret_cast<void*>(m_uBase + m_uCommitted);
    if (mprotect(pBlock, uBytes, PROT_READ | PROT_WRITE) != 0) return nullptr;
    madvise(pBlock, uBytes, MADV_HUGEPAGE);
    m_uCommitted += uBytes;
    return pBlock;
}

void C_VirtualArena::Decommit(void* pBlock, std::size_t uBytes) noexcept {
    madvise(pBlock, uBytes, MADV_DONTNEED);
}

bool C_VirtualArena::Contains(const void* pAddr) const noexcept {
    const std::uintptr_t uAddr = reinterpret_cast<std::uintptr_t>(pAddr);
    return uAddr >= m_uBase && uAddr < m_uBase + m_uCommitted;
}

// ---------------------------------------------------------------------------
// C_SegregatedAllocator — bump allocation + segregated free lists, all
// in-arena (compressed refs must be able to point at anything we hand out).
// Free-list node reuses the block's first word.
// ---------------------------------------------------------------------------

void* C_SegregatedAllocator::AllocRaw(std::size_t uBytes) noexcept {
    uBytes = (uBytes + 15) & ~std::size_t{15};
    // Size-class free list first (classes: 16..4096 in 16-byte steps).
    if (uBytes <= kMaxSmallClass) {
        void** ppHead = &m_vFreeLists[uBytes >> 4];
        if (*ppHead) {
            void* pBlock = *ppHead;
            *ppHead = *static_cast<void**>(pBlock);
            m_uTotal += uBytes;
            return pBlock;
        }
    }
    // Bump path.
    if (static_cast<std::size_t>(m_pBumpLimit - m_pBumpCursor) < uBytes) {
        const std::size_t uGranules =
            (uBytes + kArenaGranule - 1) / kArenaGranule + (uBytes <= kArenaGranule ? 3 : 0);
        std::uint8_t* pBlock = static_cast<std::uint8_t*>(m_pArena->CommitGranules(uGranules));
        if (!pBlock) return nullptr;
        // A fresh region is contiguous with the previous bump window whenever
        // commits are sequential; start a new window regardless (simple v1).
        m_pBumpCursor = pBlock;
        m_pBumpLimit = pBlock + uGranules * kArenaGranule;
    }
    void* pResult = m_pBumpCursor;
    m_pBumpCursor += uBytes;
    m_uTotal += uBytes;
    return pResult;
}

void* C_SegregatedAllocator::AllocGcObject(std::size_t uBytes) noexcept { return AllocRaw(uBytes); }
void* C_SegregatedAllocator::AllocVector(std::size_t uBytes) noexcept { return AllocRaw(uBytes); }

void* C_SegregatedAllocator::ReallocVector(void* pOld, std::size_t uOldBytes,
                                           std::size_t uNewBytes) noexcept {
    void* pNew = AllocRaw(uNewBytes);
    if (!pNew) return nullptr;
    if (pOld && uOldBytes) {
        std::memcpy(pNew, pOld, uOldBytes < uNewBytes ? uOldBytes : uNewBytes);
        Free(pOld, uOldBytes);
    }
    return pNew;
}

void C_SegregatedAllocator::Free(void* pBlock, std::size_t uBytes) noexcept {
    if (!pBlock) return;
    uBytes = (uBytes + 15) & ~std::size_t{15};
    m_uTotal -= uBytes;
    // LJX_POISON=1: make any use-after-free deterministic instead of
    // layout-dependent. 0xDB bytes form a value with an out-of-range tag, so
    // a stale TValue read trips a type guard or an assert immediately.
    static const bool bPoison = std::getenv("LJX_POISON") != nullptr;
    if (bPoison) std::memset(pBlock, 0xDB, uBytes);
    if (uBytes <= kMaxSmallClass) {
        void** ppHead = &m_vFreeLists[uBytes >> 4];
        *static_cast<void**>(pBlock) = *ppHead;
        *ppHead = pBlock;
    }
    // Large blocks are leaked to the bump region in v1 (reclaimed at arena
    // release); the GC's arena-bitmap sweep replaces this wholesale.
}

}  // namespace ljx::core
