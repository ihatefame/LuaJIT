// LJX — GC v1: precise stop-the-world mark-sweep from interpreter safe points.
// The incremental/generational engine replaces the internals behind the same
// interface (barrier call sites are already planted; they're no-ops here).
#include <cstdlib>

#include "ljx/gc/GarbageCollector.hpp"
#include "ljx/rt/Meta.hpp"
#include "ljx/rt/StringInterner.hpp"
#include "ljx/vm/Interpreter.hpp"

namespace ljx::gc {

using vm::EGcMark;
using vm::EGcObjectType;
using vm::GcHeader_t;
using vm::TValue_t;

namespace {
constexpr std::uint8_t kBlack = static_cast<std::uint8_t>(EGcMark::Black);
constexpr std::uint8_t kFixed = static_cast<std::uint8_t>(EGcMark::Fixed);
}  // namespace

void C_GarbageCollector::Init(vm::C_Universe& uni, core::C_SegregatedAllocator& alloc) noexcept {
    m_pUniverse = &uni;
    m_pAllocatorPublic = &alloc;
    m_Stats.uThreshold = 1u << 22;  // first collection after ~4 MB
}

void* C_GarbageCollector::AllocObject(EGcObjectType eType, std::size_t uBytes) {
    auto* pObj = static_cast<GcHeader_t*>(m_pAllocatorPublic->AllocGcObject(uBytes));
    *pObj = GcHeader_t{m_rAllObjects, 0, eType, 0, 0};
    m_rAllObjects = m_pUniverse->MakeRef(pObj);
    return pObj;
}

void C_GarbageCollector::FixObject(GcHeader_t* pHeader) noexcept {
    pHeader->uMarked |= kFixed;
}

void C_GarbageCollector::MarkObject(GcHeader_t* pHeader) noexcept {
    if (pHeader->uMarked & kBlack) return;
    pHeader->uMarked |= kBlack;
    if (pHeader->eType == EGcObjectType::String) return;  // no children
    if (m_uGrayCount == m_uGrayCapacity) {
        m_uGrayCapacity = m_uGrayCapacity ? m_uGrayCapacity * 2 : 1024;
        m_pGrayStack = static_cast<GcHeader_t**>(
            std::realloc(m_pGrayStack, m_uGrayCapacity * sizeof(GcHeader_t*)));
    }
    m_pGrayStack[m_uGrayCount++] = pHeader;
}

void C_GarbageCollector::MarkValue(const TValue_t& tvValue) noexcept {
    if (!tvValue.IsGcObject()) return;
    MarkObject(static_cast<GcHeader_t*>(tvValue.AsGcPointer()));
}

void C_GarbageCollector::TraverseGrays() noexcept {
    vm::C_Universe& uni = *m_pUniverse;
    while (m_uGrayCount) {
        GcHeader_t* pHeader = m_pGrayStack[--m_uGrayCount];
        switch (pHeader->eType) {
            case EGcObjectType::Table: {
                auto* pTab = reinterpret_cast<vm::C_GcTable*>(pHeader);
                if (auto* pMt = uni.Deref<vm::C_GcTable>(pTab->m_rMetatable))
                    MarkObject(&pMt->m_Header);
                auto* pArray = static_cast<TValue_t*>(
                    core::RefToPtr(uni.ArenaBase(), pTab->m_rArray));
                for (std::uint32_t uI = 0; uI < pTab->m_uArraySize; ++uI)
                    MarkValue(pArray[uI]);
                if (pTab->m_uHashMask) {
                    auto* pNodes = static_cast<vm::TableNode_t*>(
                        core::RefToPtr(uni.ArenaBase(), pTab->m_rNodes));
                    for (std::uint32_t uI = 0; uI <= pTab->m_uHashMask; ++uI) {
                        if (!pNodes[uI].tvValue.IsNil()) {
                            MarkValue(pNodes[uI].tvKey);
                            MarkValue(pNodes[uI].tvValue);
                        }
                    }
                }
                break;
            }
            case EGcObjectType::Function: {
                auto* pFn = reinterpret_cast<vm::C_GcFunction*>(pHeader);
                if (auto* pEnv = uni.Deref<vm::C_GcTable>(pFn->m_rEnv))
                    MarkObject(&pEnv->m_Header);
                if (pFn->IsLua()) {
                    auto* pProto = vm::C_GcProto::FromBytecode(pFn->m_pPc);
                    MarkObject(&const_cast<vm::C_GcProto*>(pProto)->m_Header);
                    for (std::uint8_t uI = 0; uI < pFn->UpvalCount(); ++uI) {
                        if (auto* pUv = uni.Deref<vm::C_GcUpvalue>(pFn->UpvalRefs()[uI]))
                            MarkObject(&pUv->m_Header);
                    }
                } else {
                    for (std::uint8_t uI = 0; uI < pFn->UpvalCount(); ++uI)
                        MarkValue(pFn->CUpvalues()[uI]);
                }
                break;
            }
            case EGcObjectType::Proto: {
                auto* pProto = reinterpret_cast<vm::C_GcProto*>(pHeader);
                if (auto* pName = uni.Deref<vm::C_GcString>(pProto->m_rChunkName))
                    MarkObject(&pName->m_Header);
                // GC constants: GcRefs at negative indices from the split k.
                auto* pK = static_cast<core::GcRef_t*>(
                    core::RefToPtr(uni.ArenaBase(), pProto->m_rConstants));
                for (std::uint32_t uI = 1; uI <= pProto->m_uGcConstCount; ++uI) {
                    if (auto* pObj = uni.Deref<GcHeader_t>(pK[-static_cast<std::int32_t>(uI)]))
                        MarkObject(pObj);
                }
                break;
            }
            case EGcObjectType::UpValue: {
                auto* pUv = reinterpret_cast<vm::C_GcUpvalue*>(pHeader);
                if (pUv->m_Header.uExtra1)  // closed: owns its value
                    MarkValue(pUv->m_tvClosed);
                break;
            }
            case EGcObjectType::Thread: {
                auto* pThread = reinterpret_cast<vm::C_LuaThread*>(pHeader);
                for (TValue_t* pSlot = pThread->m_pStack; pSlot < pThread->m_pTop; ++pSlot)
                    MarkValue(*pSlot);  // frame links classify as doubles → skipped
                // Everything between the live top and the deepest frame ever
                // reached is dead: clear it so a future frame can never expose
                // a slot still holding a pointer this cycle just freed. This is
                // what lets call frames skip clearing their temp slots.
                for (TValue_t* pSlot = pThread->m_pTop; pSlot < pThread->m_pHighWater; ++pSlot)
                    *pSlot = TValue_t::Nil();
                pThread->m_pHighWater = pThread->m_pTop;
                break;
            }
            case EGcObjectType::UserData: {
                auto* pUd = reinterpret_cast<vm::C_GcUserData*>(pHeader);
                if (auto* pMt = uni.Deref<vm::C_GcTable>(pUd->m_rMetatable))
                    MarkObject(&pMt->m_Header);
                if (auto* pEnv = uni.Deref<vm::C_GcTable>(pUd->m_rEnv))
                    MarkObject(&pEnv->m_Header);
                break;
            }
            default: break;
        }
    }
}

std::size_t C_GarbageCollector::ObjectSize(const GcHeader_t* pHeader) const noexcept {
    switch (pHeader->eType) {
        case EGcObjectType::Table: return sizeof(vm::C_GcTable);
        case EGcObjectType::Proto:
            return reinterpret_cast<const vm::C_GcProto*>(pHeader)->m_uTotalSize;
        case EGcObjectType::Function: {
            auto* pFn = reinterpret_cast<const vm::C_GcFunction*>(pHeader);
            return pFn->m_Header.uExtra1 == 0
                       ? vm::C_GcFunction::LuaAllocSize(pFn->m_Header.uExtra2)
                       : vm::C_GcFunction::CAllocSize(pFn->m_Header.uExtra2);
        }
        case EGcObjectType::UpValue: return sizeof(vm::C_GcUpvalue);
        case EGcObjectType::Thread: return sizeof(vm::C_LuaThread);
        case EGcObjectType::UserData:
            return sizeof(vm::C_GcUserData) +
                   reinterpret_cast<const vm::C_GcUserData*>(pHeader)->m_uLength;
        default: return 0;
    }
}

void C_GarbageCollector::SweepObjects() noexcept {
    vm::C_Universe& uni = *m_pUniverse;
    core::GcRef_t* pAnchor = &m_rAllObjects;
    while (!pAnchor->IsNull()) {
        auto* pObj = uni.Deref<GcHeader_t>(*pAnchor);
        if (pObj->uMarked & (kBlack | kFixed)) {
            pObj->uMarked &= ~kBlack;
            pAnchor = &pObj->rNextGc;
            continue;
        }
        *pAnchor = pObj->rNextGc;
        // Subsidiary vectors first.
        switch (pObj->eType) {
            case EGcObjectType::Table: {
                auto* pTab = reinterpret_cast<vm::C_GcTable*>(pObj);
                if (pTab->m_uArraySize)
                    m_pAllocatorPublic->Free(
                        core::RefToPtr(uni.ArenaBase(), pTab->m_rArray),
                        pTab->m_uArraySize * sizeof(TValue_t));
                if (pTab->m_uHashMask)
                    m_pAllocatorPublic->Free(
                        core::RefToPtr(uni.ArenaBase(), pTab->m_rNodes),
                        (pTab->m_uHashMask + 1) * sizeof(vm::TableNode_t));
                break;
            }
            case EGcObjectType::Proto: {
                auto* pProto = reinterpret_cast<vm::C_GcProto*>(pObj);
                if (!pProto->m_rLineInfo.IsNull())
                    m_pAllocatorPublic->Free(
                        core::RefToPtr(uni.ArenaBase(), pProto->m_rLineInfo),
                        pProto->m_uBcCount * sizeof(core::BcLine_t));
                break;
            }
            case EGcObjectType::Thread: {
                auto* pThread = reinterpret_cast<vm::C_LuaThread*>(pObj);
                if (pThread->m_pStack)
                    m_pAllocatorPublic->Free(pThread->m_pStack,
                                             pThread->m_uStackSize * sizeof(TValue_t));
                break;
            }
            default: break;
        }
        m_pAllocatorPublic->Free(pObj, ObjectSize(pObj));
    }
}

void C_GarbageCollector::CollectNow() noexcept {
    vm::C_Universe& uni = *m_pUniverse;
    m_eState = EGcState::Propagate;

    // Roots: globals, registry, main thread (object + live stack), metamethod
    // names (Fixed anyway), base metatables.
    MarkObject(&uni.Globals()->m_Header);
    MarkObject(&uni.Registry()->m_Header);
    MarkObject(&uni.TracePins()->m_Header);
    MarkObject(&uni.MainThread()->m_Header);
    for (std::uint32_t uRank = 0; uRank < 16; ++uRank) {
        const auto eTag = static_cast<vm::EValueTag>(~uRank);
        if (auto* pMt = uni.Meta().BaseMetatable(eTag)) MarkObject(&pMt->m_Header);
    }
    TraverseGrays();

    m_eState = EGcState::Sweep;
    uni.Interner().SweepAll();
    SweepObjects();
    // Concat memo holds raw string pointers; anything may have died.
    for (auto& entry : uni.m_vConcatCache) entry.pResult = nullptr;

    m_eState = EGcState::Pause;
    const core::GcSize_t uLive = m_pAllocatorPublic->TotalAllocated();
    m_Stats.uTotalBytes = uLive;
    m_Stats.uEstimate = uLive;
    m_Stats.uThreshold = uLive * (kDefaultPause / 100);  // 2× live
    ++m_Stats.uMajorCollections;
}

}  // namespace ljx::gc
