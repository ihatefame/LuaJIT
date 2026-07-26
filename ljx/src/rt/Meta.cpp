// LJX — metamethod resolution with the per-table negative cache.
#include "ljx/rt/Meta.hpp"

#include "ljx/gc/GarbageCollector.hpp"
#include "ljx/rt/StringInterner.hpp"
#include "ljx/vm/Interpreter.hpp"

namespace ljx::rt {

namespace {
constexpr const char* kNames[] = {
    "__index", "__newindex", "__gc",  "__mode", "__eq",  "__len",
    "__lt",    "__le",       "__concat", "__call", "__add", "__sub",
    "__mul",   "__div",      "__mod", "__pow",  "__unm", "__tostring",
    "__metatable", "__new",
};
static_assert(sizeof(kNames) / sizeof(kNames[0]) ==
              static_cast<std::size_t>(EMetaMethod::Count_));
}  // namespace

void C_MetaResolver::Init(vm::C_Universe& uni) {
    m_pUniverse = &uni;
    for (std::size_t uI = 0; uI < static_cast<std::size_t>(EMetaMethod::Count_); ++uI) {
        vm::C_GcString* pName = uni.Interner().Intern(kNames[uI]);
        uni.Gc().FixObject(&pName->m_Header);  // pinned: lookup is pointer identity
        m_vNames[uI] = pName;
    }
}

vm::C_GcString* C_MetaResolver::Name(EMetaMethod eMethod) const noexcept {
    return m_vNames[static_cast<std::size_t>(eMethod)];
}

void C_MetaResolver::InvalidateNegativeCache(vm::C_GcTable* pTable) noexcept {
    pTable->m_Header.uExtra1 = 0;
}

const vm::TValue_t* C_MetaResolver::Lookup(vm::C_GcTable* pMetatable,
                                           EMetaMethod eMethod) noexcept {
    if (!pMetatable) return nullptr;
    const std::uint8_t uBit =
        static_cast<std::uint8_t>(1u << static_cast<unsigned>(eMethod));
    const bool bCacheable =
        static_cast<std::uint8_t>(eMethod) < kNegativeCacheableCount;
    if (bCacheable && (pMetatable->m_Header.uExtra1 & uBit)) return nullptr;  // definite miss
    const vm::TValue_t* pSlot =
        pMetatable->GetStr(*m_pUniverse, Name(eMethod));
    if (!pSlot || pSlot->IsNil()) {
        if (bCacheable) pMetatable->m_Header.uExtra1 |= uBit;
        return nullptr;
    }
    return pSlot;
}

const vm::TValue_t* C_MetaResolver::LookupForValue(const vm::TValue_t& tvValue,
                                                   EMetaMethod eMethod) noexcept {
    vm::C_GcTable* pMetatable = nullptr;
    if (tvValue.Is(vm::EValueTag::Table)) {
        auto* pTab = static_cast<vm::C_GcTable*>(tvValue.AsGcPointer());
        pMetatable = m_pUniverse->Deref<vm::C_GcTable>(pTab->m_rMetatable);
    } else if (tvValue.Is(vm::EValueTag::UserData)) {
        auto* pUd = static_cast<vm::C_GcUserData*>(tvValue.AsGcPointer());
        pMetatable = m_pUniverse->Deref<vm::C_GcTable>(pUd->m_rMetatable);
    } else {
        // Base-type metatables indexed by tag rank (strings get string-library
        // methods once the stdlib installs them).
        const std::uint32_t uRank = ~tvValue.TagBits();
        if (uRank < 16)
            pMetatable = m_pUniverse->Deref<vm::C_GcTable>(m_vBaseMetatables[uRank]);
    }
    return Lookup(pMetatable, eMethod);
}

void C_MetaResolver::SetBaseMetatable(vm::EValueTag eTag, vm::C_GcTable* pMetatable) noexcept {
    m_vBaseMetatables[~static_cast<std::uint32_t>(eTag)] = m_pUniverse->MakeRef(pMetatable);
}

vm::C_GcTable* C_MetaResolver::BaseMetatable(vm::EValueTag eTag) noexcept {
    return m_pUniverse->Deref<vm::C_GcTable>(m_vBaseMetatables[~static_cast<std::uint32_t>(eTag)]);
}

}  // namespace ljx::rt
