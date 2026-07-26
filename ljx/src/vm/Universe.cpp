// LJX — universe bootstrap and shared error raising.
#include <csetjmp>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <new>

#include "ljx/gc/GarbageCollector.hpp"
#include "ljx/rt/Meta.hpp"
#include "ljx/rt/StringInterner.hpp"
#include "ljx/vm/Interpreter.hpp"
#include "ljx/jit/FuncJit.hpp"
#include "ljx/jit/LoopJit.hpp"
#include "ljx/jit/TraceJit.hpp"
#include "ljx/vm/Object.hpp"

namespace ljx::vm {

C_Universe* C_Universe::Create(std::size_t uArenaReserveBytes) noexcept {
    core::C_VirtualArena arena;
    if (!arena.Reserve(uArenaReserveBytes)) return nullptr;

    core::C_SegregatedAllocator bootAlloc(arena);
    // Burn the first granule slot so compressed-ref index 0 stays "null".
    if (!bootAlloc.AllocGcObject(16)) {
        arena.ReleaseAll();
        return nullptr;
    }

    void* pUniMem = bootAlloc.AllocGcObject(sizeof(C_Universe));
    if (!pUniMem) {
        arena.ReleaseAll();
        return nullptr;
    }
    C_Universe* pUni = new (pUniMem) C_Universe();
    pUni->m_Arena = arena;
    pUni->m_uArenaBase = arena.BaseAddress();

    if (!pUni->m_Prng.SeedSecure()) {  // hard requirement: no entropy, no VM
        pUni->m_Arena.ReleaseAll();
        return nullptr;
    }

    // The real allocator lives in-arena; copy the bootstrap state across and
    // rebind it to the arena's final home inside the universe.
    void* pAllocMem = bootAlloc.AllocGcObject(sizeof(core::C_SegregatedAllocator));
    auto* pAlloc = new (pAllocMem) core::C_SegregatedAllocator(bootAlloc);
    pAlloc->RebindArena(pUni->m_Arena);
    pUni->m_pAllocator = pAlloc;

    pUni->m_tvNil = TValue_t::Nil();
    pUni->m_NilNode = TableNode_t{TValue_t::Nil(), TValue_t::Nil(), core::MRef_t{}, 0};

    auto* pGc = new (pAlloc->AllocGcObject(sizeof(gc::C_GarbageCollector)))
        gc::C_GarbageCollector();
    pGc->Init(*pUni, *pAlloc);
    pUni->m_pGc = pGc;

    auto* pInterner = new (pAlloc->AllocGcObject(sizeof(rt::C_StringInterner)))
        rt::C_StringInterner();
    pInterner->Init(*pUni);
    pUni->m_pInterner = pInterner;

    auto* pMeta = new (pAlloc->AllocGcObject(sizeof(rt::C_MetaResolver))) rt::C_MetaResolver();
    pMeta->Init(*pUni);
    pUni->m_pMeta = pMeta;

    // Main thread: full stack committed up front (no growth, no fixups).
    auto* pThread = static_cast<C_LuaThread*>(
        pGc->AllocObject(EGcObjectType::Thread, sizeof(C_LuaThread)));
    const std::uint32_t uSlots = kMaxStackSlots + kStackExtraSlots;
    auto* pStack = static_cast<TValue_t*>(pAlloc->AllocVector(uSlots * sizeof(TValue_t)));
    for (std::uint32_t uI = 0; uI < uSlots; ++uI) pStack[uI] = TValue_t::Nil();
    pThread->m_pStack = pStack;
    pThread->m_uStackSize = uSlots;
    pThread->m_pMaxStack = pStack + kMaxStackSlots;
    pThread->m_pBase = pStack + 2;   // room for a dummy frame [func][link]
    pThread->m_pTop = pThread->m_pBase;
    pThread->m_pHighWater = pThread->m_pBase;
    pThread->m_rGlobal = core::PtrToRef(pUni->m_uArenaBase, pUni);
    pThread->m_pUniverse = pUni;
    pUni->m_pMainThread = pThread;
    pGc->FixObject(&pThread->m_Header);

    pUni->m_pGlobals = C_GcTable::New(*pUni, 0, 6);
    pUni->m_pRegistry = C_GcTable::New(*pUni, 0, 2);

    pUni->m_insCFuncHeader = BcIns_t::MakeAD(EBcOp::FuncC, 0, 0);
    auto* pLoopJit = new (pAlloc->AllocGcObject(sizeof(jit::C_LoopJit)))
        jit::C_LoopJit(*pUni);
    pUni->SetLoopJit(pLoopJit);
    auto* pFuncJit = new (pAlloc->AllocGcObject(sizeof(jit::C_FuncJit)))
        jit::C_FuncJit(*pUni);
    pUni->SetFuncJit(pFuncJit);
    auto* pTraceJit = new (pAlloc->AllocGcObject(sizeof(jit::C_TraceJit)))
        jit::C_TraceJit(*pUni);
    pUni->SetTraceJit(pTraceJit);
    C_Interpreter::InitDispatchTables(pUni->m_Dispatch);
    return pUni;
}

void C_Universe::Destroy() noexcept {
    core::C_VirtualArena arena = m_Arena;  // copy out before the memory dies
    arena.ReleaseAll();
}

void C_Universe::SetEventSink(C_IVmEventSink* pSink) noexcept { m_pEventSink = pSink; }

[[noreturn]] void RaiseErrorValue(C_Universe& uni, const TValue_t& tvError) {
    uni.m_tvErrorValue = tvError;
    if (!uni.m_pErrorTop) {  // unprotected: last resort
        std::fprintf(stderr, "ljx: unprotected error\n");
        std::abort();
    }
    std::longjmp(uni.m_pErrorTop->jb, 1);
}

[[noreturn]] void RaiseError(C_Universe& uni, const char* sFormat, ...) {
    char vBuffer[512];
    va_list args;
    va_start(args, sFormat);
    std::vsnprintf(vBuffer, sizeof vBuffer, sFormat, args);
    va_end(args);
    C_GcString* pMessage = uni.Interner().Intern(vBuffer);
    RaiseErrorValue(uni, TValue_t::GcObject(EValueTag::String, pMessage));
}

}  // namespace ljx::vm
