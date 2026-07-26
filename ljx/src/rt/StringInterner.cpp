// LJX — string interning.
#include <cstring>

#include "ljx/rt/StringInterner.hpp"
#include "ljx/vm/Interpreter.hpp"

namespace ljx::rt {

using vm::C_GcString;

void C_StringInterner::Init(vm::C_Universe& uni) {
    m_pUniverse = &uni;
    m_uSeed = uni.Prng().Next();
    m_uNextSid = static_cast<core::StrId_t>(uni.Prng().Next());
    m_uMask = kMinTableSize - 1;
    m_pChains = static_cast<core::GcRef_t*>(
        uni.Allocator().AllocVector(kMinTableSize * sizeof(core::GcRef_t)));
    std::memset(m_pChains, 0, kMinTableSize * sizeof(core::GcRef_t));
    m_pEmpty = Intern(std::string_view{});
    m_pEmpty->m_Header.uMarked |= static_cast<std::uint8_t>(vm::EGcMark::Fixed);
}

// Keyed constant-time-ish hash: full mix for short strings, sparse sampling
// (head/middle/tail words) for long ones — LuaJIT's O(1) interning property.
core::StrHash_t C_StringInterner::HashSparse(std::string_view svBytes) const noexcept {
    std::uint64_t uH = m_uSeed ^ (0x9e3779b97f4a7c15ull * (svBytes.size() + 1));
    auto MixWord = [&uH](std::uint64_t uWord) {
        uH ^= uWord;
        uH *= 0xff51afd7ed558ccdull;
        uH ^= uH >> 33;
    };
    auto LoadWord = [&svBytes](std::size_t uOfs) {
        std::uint64_t uWord = 0;
        const std::size_t uN = svBytes.size() - uOfs < 8 ? svBytes.size() - uOfs : 8;
        std::memcpy(&uWord, svBytes.data() + uOfs, uN);
        return uWord;
    };
    const std::size_t uLen = svBytes.size();
    if (uLen <= 32) {
        for (std::size_t uOfs = 0; uOfs < uLen; uOfs += 8) MixWord(LoadWord(uOfs));
    } else {
        MixWord(LoadWord(0));
        MixWord(LoadWord(8));
        MixWord(LoadWord((uLen / 2) & ~std::size_t{7}));
        MixWord(LoadWord(uLen - 8));
        MixWord(LoadWord(uLen - 16));
    }
    return static_cast<core::StrHash_t>(uH ^ (uH >> 32));
}

C_GcString* C_StringInterner::Intern(std::string_view svBytes) {
    const core::StrHash_t uHash = HashSparse(svBytes);
    const std::uint32_t uChain = uHash & m_uMask;
    const std::uintptr_t uBase = m_pUniverse->ArenaBase();

    for (core::GcRef_t rWalk = m_pChains[uChain]; !rWalk.IsNull();) {
        auto* pStr = reinterpret_cast<C_GcString*>(
            uBase + (std::uintptr_t{rWalk.uIndex} << core::kCompressedRefShift));
        if (pStr->m_uHash == uHash && pStr->m_uLength == svBytes.size() &&
            (svBytes.empty() ||
             std::memcmp(pStr->Data(), svBytes.data(), svBytes.size()) == 0)) {
            pStr->m_Header.uMarked |= static_cast<std::uint8_t>(vm::EGcMark::Black);
            return pStr;  // hit (resurrected if it was dying this cycle)
        }
        rWalk = pStr->m_Header.rNextGc;
    }

    // Miss: allocate, zero the padded tail (SIMD-compare contract), copy.
    const std::size_t uAlloc = C_GcString::AllocSize(static_cast<std::uint32_t>(svBytes.size()));
    auto* pStr = static_cast<C_GcString*>(m_pUniverse->Allocator().AllocGcObject(uAlloc));
    char* pData = reinterpret_cast<char*>(pStr + 1);
    std::memset(pData + (svBytes.size() & ~std::size_t{15}), 0, 16);
    if (!svBytes.empty()) std::memcpy(pData, svBytes.data(), svBytes.size());
    pStr->m_Header = vm::GcHeader_t{m_pChains[uChain], 0, vm::EGcObjectType::String, 0, 0};
    pStr->m_uSid = m_uNextSid++;
    pStr->m_uHash = uHash;
    pStr->m_uLength = static_cast<std::uint32_t>(svBytes.size());
    m_pChains[uChain] = m_pUniverse->MakeRef(pStr);
    ++m_uCount;
    if (m_uCount > m_uMask) MaybeResize();
    return pStr;
}

void C_StringInterner::MaybeResize() {
    if (m_uCount <= m_uMask || m_uMask + 1 >= (1u << 26)) return;
    const std::uint32_t uNewSize = (m_uMask + 1) * 2;
    auto* pNew = static_cast<core::GcRef_t*>(
        m_pUniverse->Allocator().AllocVector(uNewSize * sizeof(core::GcRef_t)));
    std::memset(pNew, 0, uNewSize * sizeof(core::GcRef_t));
    const std::uintptr_t uBase = m_pUniverse->ArenaBase();
    for (std::uint32_t uI = 0; uI <= m_uMask; ++uI) {
        core::GcRef_t rWalk = m_pChains[uI];
        while (!rWalk.IsNull()) {
            auto* pStr = reinterpret_cast<C_GcString*>(
                uBase + (std::uintptr_t{rWalk.uIndex} << core::kCompressedRefShift));
            const core::GcRef_t rNext = pStr->m_Header.rNextGc;
            const std::uint32_t uNewChain = pStr->m_uHash & (uNewSize - 1);
            pStr->m_Header.rNextGc = pNew[uNewChain];
            pNew[uNewChain] = rWalk;
            rWalk = rNext;
        }
    }
    m_pUniverse->Allocator().Free(m_pChains, (m_uMask + 1) * sizeof(core::GcRef_t));
    m_pChains = pNew;
    m_uMask = uNewSize - 1;
}

void C_StringInterner::SweepAll() noexcept {
    const std::uintptr_t uBase = m_pUniverse->ArenaBase();
    constexpr std::uint8_t uKeep = static_cast<std::uint8_t>(vm::EGcMark::Black) |
                                   static_cast<std::uint8_t>(vm::EGcMark::Fixed);
    for (std::uint32_t uI = 0; uI <= m_uMask; ++uI) {
        core::GcRef_t* pAnchor = &m_pChains[uI];
        while (!pAnchor->IsNull()) {
            auto* pStr = reinterpret_cast<C_GcString*>(
                uBase + (std::uintptr_t{pAnchor->uIndex} << core::kCompressedRefShift));
            if (pStr->m_Header.uMarked & uKeep) {
                pStr->m_Header.uMarked &= ~static_cast<std::uint8_t>(vm::EGcMark::Black);
                pAnchor = &pStr->m_Header.rNextGc;
            } else {
                *pAnchor = pStr->m_Header.rNextGc;
                m_pUniverse->Allocator().Free(pStr, C_GcString::AllocSize(pStr->m_uLength));
                --m_uCount;
            }
        }
    }
}

}  // namespace ljx::rt
