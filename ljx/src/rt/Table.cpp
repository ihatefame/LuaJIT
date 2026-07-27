// LJX — hybrid array+hash tables: main-position chaining, Brent's eviction,
// dead-key slot stability, sid-hashed string keys.
#include <cstring>

#include "ljx/gc/GarbageCollector.hpp"
#include "ljx/rt/StringInterner.hpp"
#include "ljx/vm/Interpreter.hpp"
#include "ljx/vm/Object.hpp"

namespace ljx::vm {

namespace {

LJX_FORCEINLINE std::uint32_t HashRot(std::uint32_t uLo, std::uint32_t uHi) noexcept {
    uLo ^= uHi;
    uHi = core::RotL32(uHi, 14);
    uLo -= uHi;
    uHi = core::RotL32(uHi, 5);
    uHi ^= uLo;
    uHi -= core::RotL32(uLo, 13);
    return uHi;
}

LJX_FORCEINLINE TableNode_t* NodeArray(C_Universe& uni, const C_GcTable* pTab) noexcept {
    return static_cast<TableNode_t*>(core::RefToPtr(uni.ArenaBase(), pTab->m_rNodes));
}

LJX_FORCEINLINE TValue_t* ArrayPart(C_Universe& uni, const C_GcTable* pTab) noexcept {
    return static_cast<TValue_t*>(core::RefToPtr(uni.ArenaBase(), pTab->m_rArray));
}

// Canonicalized key (−0 → +0); returns the node hash chain head for the key.
LJX_FORCEINLINE TableNode_t* MainPosition(C_Universe& uni, const C_GcTable* pTab,
                                          const TValue_t& tvKey) noexcept {
    std::uint32_t uHash;
    if (tvKey.Is(EValueTag::String)) {
        uHash = static_cast<const C_GcString*>(tvKey.AsGcPointer())->m_uSid;
    } else if (tvKey.IsDouble()) {
        const std::uint64_t uBits = tvKey.uRaw;
        uHash = HashRot(static_cast<std::uint32_t>(uBits),
                        static_cast<std::uint32_t>(uBits >> 32));
    } else {
        // GC objects / booleans: full-word mix (identity hashing).
        uHash = HashRot(static_cast<std::uint32_t>(tvKey.uRaw >> 4),
                        static_cast<std::uint32_t>(tvKey.uRaw >> 36));
    }
    return NodeArray(uni, pTab) + (uHash & pTab->m_uHashMask);
}

// Integer-valued array index for a numeric key, or ~0u.
LJX_FORCEINLINE std::uint32_t ArrayIndex(const TValue_t& tvKey) noexcept {
    if (tvKey.IsDouble()) {
        const double flKey = tvKey.AsDouble();
        std::int32_t nKey;
        if (core::NumToInt32Check(flKey, nKey) && nKey >= 0)
            return static_cast<std::uint32_t>(nKey);
    }
    return ~0u;
}

LJX_FORCEINLINE TValue_t CanonicalKey(const TValue_t& tvKey) noexcept {
    // −0 keys are stored as +0 so bitwise key compare is exact.
    if (tvKey.uRaw == std::uint64_t{1} << 63) return TValue_t::Number(0.0);
    return tvKey;
}

}  // namespace

C_GcTable* C_GcTable::New(C_Universe& uni, std::uint32_t uArraySizeHint,
                          std::uint32_t uHashBits) {
    auto* pTab = static_cast<C_GcTable*>(
        uni.Gc().AllocObject(EGcObjectType::Table, sizeof(C_GcTable)));
    pTab->m_Header.uExtra1 = 0xff;  // fresh table: every metamethod definitely absent
    pTab->m_rMetatable = core::GcRef_t{};
    pTab->m_uArraySize = uArraySizeHint;
    if (uArraySizeHint) {
        auto* pArray =
            static_cast<TValue_t*>(uni.Allocator().AllocVector(uArraySizeHint * sizeof(TValue_t)));
        for (std::uint32_t uI = 0; uI < uArraySizeHint; ++uI) pArray[uI] = TValue_t::Nil();
        pTab->m_rArray = core::PtrToRef(uni.ArenaBase(), pArray);
    } else {
        pTab->m_rArray = core::MRef_t{};
    }
    if (uHashBits) {
        const std::uint32_t uCount = 1u << uHashBits;
        auto* pNodes =
            static_cast<TableNode_t*>(uni.Allocator().AllocVector(uCount * sizeof(TableNode_t)));
        for (std::uint32_t uI = 0; uI < uCount; ++uI)
            pNodes[uI] = TableNode_t{TValue_t::Nil(), TValue_t::Nil(), core::MRef_t{}, 0};
        pTab->m_rNodes = core::PtrToRef(uni.ArenaBase(), pNodes);
        pTab->m_uHashMask = uCount - 1;
        pTab->m_rFreeTop = core::PtrToRef(uni.ArenaBase(), pNodes + uCount);
    } else {
        pTab->m_rNodes = core::PtrToRef(uni.ArenaBase(),
                                        const_cast<TableNode_t*>(uni.NilNode()));
        pTab->m_uHashMask = 0;
        pTab->m_rFreeTop = pTab->m_rNodes;
    }
    return pTab;
}

const TValue_t* C_GcTable::GetInt(C_Universe& uni, std::uint32_t uKey) const noexcept {
    if (uKey < m_uArraySize) return &ArrayPart(uni, this)[uKey];
    const TValue_t tvKey = TValue_t::Number(static_cast<double>(uKey));
    const TableNode_t* pNode = MainPosition(uni, this, tvKey);
    do {
        if (pNode->tvKey == tvKey) return &pNode->tvValue;
        pNode = static_cast<const TableNode_t*>(core::RefToPtr(uni.ArenaBase(), pNode->rNext));
    } while (pNode != reinterpret_cast<const TableNode_t*>(uni.ArenaBase()));
    return nullptr;
}

const TValue_t* C_GcTable::GetStr(C_Universe& uni, const C_GcString* pKey) const noexcept {
    const TableNode_t* pNode = NodeArray(uni, this) + (pKey->m_uSid & m_uHashMask);
    const TValue_t tvKey = TValue_t::GcObject(EValueTag::String, pKey);
    do {
        if (pNode->tvKey == tvKey) return &pNode->tvValue;  // one fused compare
        pNode = static_cast<const TableNode_t*>(core::RefToPtr(uni.ArenaBase(), pNode->rNext));
    } while (pNode != reinterpret_cast<const TableNode_t*>(uni.ArenaBase()));
    return nullptr;
}

const TValue_t* C_GcTable::Get(C_Universe& uni, const TValue_t& tvRawKey) const noexcept {
    const std::uint32_t uIndex = ArrayIndex(tvRawKey);
    if (uIndex < m_uArraySize) return &ArrayPart(uni, this)[uIndex];
    if (tvRawKey.Is(EValueTag::String))
        return GetStr(uni, static_cast<const C_GcString*>(tvRawKey.AsGcPointer()));
    if (tvRawKey.IsNil()) return nullptr;
    const TValue_t tvKey = CanonicalKey(tvRawKey);
    const TableNode_t* pNode = MainPosition(uni, this, tvKey);
    do {
        if (pNode->tvKey == tvKey) return &pNode->tvValue;
        pNode = static_cast<const TableNode_t*>(core::RefToPtr(uni.ArenaBase(), pNode->rNext));
    } while (pNode != reinterpret_cast<const TableNode_t*>(uni.ArenaBase()));
    return nullptr;
}

// --- hash-part growth -------------------------------------------------------

void C_GcTable::Resize(C_Universe& uni, std::uint32_t uNewArraySize, std::uint32_t uHashBits) {
    BumpVersion();
    // Array growth: migrate hash entries with in-range integer keys.
    if (uNewArraySize > m_uArraySize) {
        auto* pNew =
            static_cast<TValue_t*>(uni.Allocator().AllocVector(uNewArraySize * sizeof(TValue_t)));
        TValue_t* pOld = ArrayPart(uni, this);
        for (std::uint32_t uI = 0; uI < m_uArraySize; ++uI) pNew[uI] = pOld[uI];
        for (std::uint32_t uI = m_uArraySize; uI < uNewArraySize; ++uI) pNew[uI] = TValue_t::Nil();
        if (m_uArraySize)
            uni.Allocator().Free(pOld, m_uArraySize * sizeof(TValue_t));
        const std::uint32_t uOldArraySize = m_uArraySize;
        m_rArray = core::PtrToRef(uni.ArenaBase(), pNew);
        m_uArraySize = uNewArraySize;
        // Pull in-range integer keys out of the hash part.
        if (m_uHashMask) {
            TableNode_t* pNodes = NodeArray(uni, this);
            for (std::uint32_t uI = 0; uI <= m_uHashMask; ++uI) {
                TableNode_t& node = pNodes[uI];
                if (node.tvValue.IsNil()) continue;
                const std::uint32_t uIndex = ArrayIndex(node.tvKey);
                if (uIndex >= uOldArraySize && uIndex < uNewArraySize) {
                    pNew[uIndex] = node.tvValue;
                    node.tvValue = TValue_t::Nil();  // dead key stays (stability)
                }
            }
        }
    }

    // Hash growth: fresh node array, reinsert everything live.
    const std::uint32_t uOldMask = m_uHashMask;
    TableNode_t* pOldNodes = m_uHashMask ? NodeArray(uni, this) : nullptr;
    if (uHashBits) {
        const std::uint32_t uCount = 1u << uHashBits;
        auto* pNodes =
            static_cast<TableNode_t*>(uni.Allocator().AllocVector(uCount * sizeof(TableNode_t)));
        for (std::uint32_t uI = 0; uI < uCount; ++uI)
            pNodes[uI] = TableNode_t{TValue_t::Nil(), TValue_t::Nil(), core::MRef_t{}, 0};
        m_rNodes = core::PtrToRef(uni.ArenaBase(), pNodes);
        m_uHashMask = uCount - 1;
        m_rFreeTop = core::PtrToRef(uni.ArenaBase(), pNodes + uCount);
        if (pOldNodes) {
            for (std::uint32_t uI = 0; uI <= uOldMask; ++uI) {
                if (!pOldNodes[uI].tvValue.IsNil())
                    *Set(uni, pOldNodes[uI].tvKey) = pOldNodes[uI].tvValue;
            }
            uni.Allocator().Free(pOldNodes, (uOldMask + 1) * sizeof(TableNode_t));
        }
    }
}

TValue_t* C_GcTable::Set(C_Universe& uni, const TValue_t& tvRawKey) {
    m_Header.uExtra1 = 0;  // any store invalidates the negative metamethod cache

    const std::uint32_t uIndex = ArrayIndex(tvRawKey);
    if (uIndex < m_uArraySize) return &ArrayPart(uni, this)[uIndex];
    // Append pattern: key == asize grows the array part (amortized ×2).
    if (uIndex != ~0u && uIndex == m_uArraySize && uIndex < (1u << 27)) {
        const std::uint32_t uNewSize = uIndex < 4 ? 4 : uIndex * 2;
        Resize(uni, uNewSize, 0);
        return &ArrayPart(uni, this)[uIndex];
    }

    if (tvRawKey.IsNil()) RaiseError(uni, "table index is nil");
    if (tvRawKey.IsDouble() && tvRawKey.AsDouble() != tvRawKey.AsDouble())
        RaiseError(uni, "table index is NaN");
    const TValue_t tvKey = CanonicalKey(tvRawKey);

    // Existing key (live or dead — dead keys keep their node).
    if (m_uHashMask) {
        TableNode_t* pNode = MainPosition(uni, this, tvKey);
        do {
            if (pNode->tvKey == tvKey) return &pNode->tvValue;
            pNode = static_cast<TableNode_t*>(core::RefToPtr(uni.ArenaBase(), pNode->rNext));
        } while (pNode != reinterpret_cast<TableNode_t*>(uni.ArenaBase()));
    }

    // New key insertion.
    if (!m_uHashMask) {
        Resize(uni, m_uArraySize, 1);
        return Set(uni, tvKey);
    }
    TableNode_t* pNodes = NodeArray(uni, this);
    TableNode_t* pMain = MainPosition(uni, this, tvKey);
    if (!pMain->tvKey.IsNil()) {
        // Find a free node scanning down from the free-top cursor.
        auto* pFree = static_cast<TableNode_t*>(core::RefToPtr(uni.ArenaBase(), m_rFreeTop));
        for (;;) {
            if (pFree == pNodes) {  // full: grow ×2 and retry
                Resize(uni, m_uArraySize,
                       static_cast<std::uint32_t>(core::FindFirstSet(m_uHashMask + 1)) + 1);
                return Set(uni, tvKey);
            }
            --pFree;
            if (pFree->tvKey.IsNil()) break;
        }
        m_rFreeTop = core::PtrToRef(uni.ArenaBase(), pFree);
        TableNode_t* pCollider = pMain;
        TableNode_t* pColliderMain = MainPosition(uni, this, pCollider->tvKey);
        BumpVersion();   // key creation: node addresses change meaning
        if (pColliderMain != pMain) {
            // Brent's eviction: the resident node doesn't belong here — move
            // it to the free node so the main position owns our key.
            TableNode_t* pPrev = pColliderMain;
            while (static_cast<TableNode_t*>(core::RefToPtr(uni.ArenaBase(), pPrev->rNext)) !=
                   pMain)
                pPrev = static_cast<TableNode_t*>(core::RefToPtr(uni.ArenaBase(), pPrev->rNext));
            *pFree = *pMain;
            pPrev->rNext = core::PtrToRef(uni.ArenaBase(), pFree);
            pMain->tvKey = tvKey;
            pMain->tvValue = TValue_t::Nil();
            pMain->rNext = core::MRef_t{};
            return &pMain->tvValue;
        }
        // Same main position: chain the new key in behind it.
        pFree->tvKey = tvKey;
        pFree->tvValue = TValue_t::Nil();
        pFree->rNext = pMain->rNext;
        pMain->rNext = core::PtrToRef(uni.ArenaBase(), pFree);
        return &pFree->tvValue;
    }
    BumpVersion();   // key creation
    pMain->tvKey = tvKey;
    pMain->tvValue = TValue_t::Nil();
    pMain->rNext = core::MRef_t{};
    return &pMain->tvValue;
}

std::uint32_t C_GcTable::Length(C_Universe& uni) const noexcept {
    // Border search: array part first (binary search for t[n]≠nil, t[n+1]==nil).
    const TValue_t* pArray = ArrayPart(uni, this);
    if (m_uArraySize > 1 && !pArray[1].IsNil()) {
        std::uint32_t uLo = 1, uHi = m_uArraySize - 1;
        // Find the highest non-nil in the array part.
        if (!pArray[uHi].IsNil()) {
            // Full array part: continue into the hash for t[asize], t[asize+1]…
            std::uint32_t uN = uHi;
            for (;;) {
                const TValue_t* pVal = GetInt(uni, uN + 1);
                if (!pVal || pVal->IsNil()) return uN;
                ++uN;
            }
        }
        while (uHi - uLo > 1) {
            const std::uint32_t uMid = (uLo + uHi) / 2;
            if (pArray[uMid].IsNil()) uHi = uMid;
            else uLo = uMid;
        }
        return uLo;
    }
    if (m_uArraySize <= 1 || pArray[1].IsNil()) {
        // Nothing in the array part: probe the hash for 1, 2, …
        std::uint32_t uN = 0;
        for (;;) {
            const TValue_t* pVal = GetInt(uni, uN + 1);
            if (!pVal || pVal->IsNil()) return uN;
            ++uN;
        }
    }
    return 0;
}

// Iteration support for `next`: returns the successor of tvKey (nil = start),
// false when the traversal is exhausted.
bool TableNext(C_Universe& uni, C_GcTable* pTab, const TValue_t& tvKey, TValue_t& tvOutKey,
               TValue_t& tvOutVal) {
    TValue_t* pArray = ArrayPart(uni, pTab);
    std::uint32_t uStart = 0;
    if (!tvKey.IsNil()) {
        const std::uint32_t uIndex = ArrayIndex(tvKey);
        if (uIndex < pTab->m_uArraySize) {
            uStart = uIndex + 1;
        } else {
            // Resume from a hash node. The universe's one-entry hint caches
            // where this key was found on the previous step, collapsing the
            // hash lookup to one compare for in-order iteration. The hint is
            // validated by re-reading the node's key, so a stale entry
            // (resize, free, address reuse) misses instead of misdirecting.
            TableNode_t* pNodes = NodeArray(uni, pTab);
            std::uint32_t uNode;
            if (uni.m_pIterHintTab == pTab && uni.m_uIterHintKey == tvKey.uRaw &&
                uni.m_uIterHintNode <= pTab->m_uHashMask &&
                pNodes[uni.m_uIterHintNode].tvKey == tvKey) {
                uNode = uni.m_uIterHintNode;
            } else {
                const TValue_t* pSlot = pTab->Get(uni, tvKey);
                if (!pSlot) RaiseError(uni, "invalid key to 'next'");
                uNode = static_cast<std::uint32_t>(
                    reinterpret_cast<const TableNode_t*>(pSlot) - pNodes);
            }
            for (std::uint32_t uI = uNode + 1; uI <= pTab->m_uHashMask; ++uI) {
                if (!pNodes[uI].tvValue.IsNil()) {
                    tvOutKey = pNodes[uI].tvKey;
                    tvOutVal = pNodes[uI].tvValue;
                    uni.m_pIterHintTab = pTab;
                    uni.m_uIterHintKey = tvOutKey.uRaw;
                    uni.m_uIterHintNode = uI;
                    return true;
                }
            }
            return false;
        }
    }
    for (std::uint32_t uI = uStart; uI < pTab->m_uArraySize; ++uI) {
        if (!pArray[uI].IsNil()) {
            tvOutKey = TValue_t::Number(static_cast<double>(uI));
            tvOutVal = pArray[uI];
            return true;
        }
    }
    if (pTab->m_uHashMask) {
        TableNode_t* pNodes = NodeArray(uni, pTab);
        for (std::uint32_t uI = 0; uI <= pTab->m_uHashMask; ++uI) {
            if (!pNodes[uI].tvValue.IsNil()) {
                tvOutKey = pNodes[uI].tvKey;
                tvOutVal = pNodes[uI].tvValue;
                uni.m_pIterHintTab = pTab;
                uni.m_uIterHintKey = tvOutKey.uRaw;
                uni.m_uIterHintNode = uI;
                return true;
            }
        }
    }
    return false;
}

}  // namespace ljx::vm
