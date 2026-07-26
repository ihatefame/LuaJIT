// LJX — the continuation-passing tail-call interpreter.
//
// One function per opcode; every dispatch is a guaranteed tail call (clang
// musttail), each handler ends in its own indirect jump (replicated
// dispatch), slow paths are separate noinline functions. Frames live on the
// value stack (FR2 two-slot layout); returns re-derive the caller's state
// from the frame link + the call instruction, exactly like the assembly VM.
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

#include "ljx/gc/GarbageCollector.hpp"
#include "ljx/rt/Meta.hpp"
#include "ljx/rt/StringBuffer.hpp"
#include "ljx/rt/StringInterner.hpp"
#include "ljx/jit/FuncJit.hpp"
#include "ljx/jit/LoopJit.hpp"
#include "ljx/jit/TraceJit.hpp"
#include "ljx/vm/Interpreter.hpp"

namespace ljx::vm {

// Defined in rt/Table.cpp.
bool TableNext(C_Universe& uni, C_GcTable* pTab, const TValue_t& tvKey, TValue_t& tvOutKey,
               TValue_t& tvOutVal);

namespace {

#define LJX_HANDLER_ARGS                                                        \
    [[maybe_unused]] TValue_t* pBase, [[maybe_unused]] const BcIns_t* pPc,      \
        [[maybe_unused]] std::uint64_t uRa, [[maybe_unused]] std::uint64_t uRd, \
        [[maybe_unused]] C_Universe* pUni, [[maybe_unused]] const TValue_t* pKBase

#define LJX_PASS_ARGS pBase, pPc, uRa, uRd, pUni, pKBase

// Replicated dispatch tail — expanded at the end of every handler.
#define LJX_NEXT()                                                              \
    do {                                                                        \
        const BcIns_t insNext{pPc->uRaw};                                       \
        BcHandler_f fnNext = pUni->Dispatch().Dynamic(insNext.uRaw & 0xff);     \
        LJX_MUSTTAIL return fnNext(pBase, pPc + 1, insNext.A(), insNext.D(),    \
                                   pUni, pKBase);                               \
    } while (0)

// ---------------------------------------------------------------------------
// Shared helpers (cold paths are noinline; hot helpers force-inline).
// ---------------------------------------------------------------------------

LJX_FORCEINLINE C_GcFunction* FrameFunc(TValue_t* pBase) noexcept {
    return static_cast<C_GcFunction*>(pBase[-2].AsGcPointer());
}

LJX_FORCEINLINE const C_GcProto* FrameProto(C_Universe* pUni, TValue_t* pBase) noexcept {
    return C_GcProto::FromBytecode(static_cast<const std::uint32_t*>(
        FrameFunc(pBase)->m_pPc));
}

LJX_FORCEINLINE const TValue_t* KBaseOf(C_Universe* pUni, const C_GcProto* pProto) noexcept {
    return static_cast<const TValue_t*>(core::RefToPtr(pUni->ArenaBase(), pProto->m_rConstants));
}

LJX_FORCEINLINE C_GcString* KgcString(C_Universe* pUni, const TValue_t* pKBase,
                                      std::uint32_t uIdx) noexcept {
    const auto* pKgc = reinterpret_cast<const core::GcRef_t*>(pKBase);
    return pUni->Deref<C_GcString>(pKgc[-static_cast<std::int32_t>(uIdx + 1)]);
}

LJX_NOINLINE const char* TypeName(const TValue_t& tvValue) {
    if (tvValue.IsNil()) return "nil";
    if (tvValue.IsNumber()) return "number";
    switch (static_cast<EValueTag>(tvValue.TagBits())) {
        case EValueTag::False: case EValueTag::True: return "boolean";
        case EValueTag::String: return "string";
        case EValueTag::Table: return "table";
        case EValueTag::Function: return "function";
        case EValueTag::UserData: return "userdata";
        case EValueTag::Thread: return "thread";
        default: return "value";
    }
}

// GC safe point: stack extent is well-defined here.
LJX_NOINLINE void GcCheck(C_Universe* pUni, TValue_t* pBase) {
    // A collection while the recorder is running could free an object the
    // in-progress trace has already specialized on.
    if (pUni->m_uRecording) return;
    C_LuaThread* pThread = pUni->MainThread();
    pThread->m_pBase = pBase;
    pThread->m_pTop = pBase + FrameProto(pUni, pBase)->FrameSize();
    if (pUni->Gc().NeedsStep()) pUni->Gc().CollectNow();
}

[[noreturn]] LJX_NOINLINE void ErrorAtPc(C_Universe* pUni, TValue_t* pBase,
                                         const BcIns_t* pPc, const char* sFormat,
                                         const char* sDetail) {
    const C_GcProto* pProto = FrameProto(pUni, pBase);
    const C_GcString* pChunk = pUni->Deref<C_GcString>(pProto->m_rChunkName);
    core::BcLine_t uLine = 0;
    if (!pProto->m_rLineInfo.IsNull()) {
        const auto* pLines = static_cast<const core::BcLine_t*>(
            core::RefToPtr(pUni->ArenaBase(), pProto->m_rLineInfo));
        const std::size_t uIdx =
            static_cast<std::size_t>(pPc - 1 -
                                     reinterpret_cast<const BcIns_t*>(pProto->Bytecode()));
        if (uIdx < pProto->m_uBcCount) uLine = pLines[uIdx];
    }
    char vBuffer[256];
    std::snprintf(vBuffer, sizeof vBuffer, sFormat, sDetail);
    RaiseError(*pUni, "%s:%u: %s", pChunk ? pChunk->Data() : "?", uLine, vBuffer);
}

// Number coercion for arithmetic (strings are NOT coerced in v1).
LJX_FORCEINLINE bool BothNumbers(const TValue_t& tvA, const TValue_t& tvB) noexcept {
    return static_cast<int>(tvA.IsDouble()) & static_cast<int>(tvB.IsDouble());
}

// Load a slot's payload straight into the FP domain. Going through
// TValue_t::AsDouble() would bit_cast the integer the tag check already
// loaded, and the compiler turns that into a GPR->XMM `movq`; reloading from
// the same address instead keeps the value in the FP domain (the second load
// is free — it hits the same cache line the tag check just touched).
LJX_FORCEINLINE double LoadNum(const TValue_t& tvValue) noexcept {
    double flValue;
    std::memcpy(&flValue, &tvValue, sizeof flValue);
    return flValue;
}

// Reentrant metamethod invocation (v1: nested dispatch loop via a fresh
// C-entry frame above the current one; correct, costs one C++ frame).
TValue_t MetaCall2(C_Universe* pUni, TValue_t* pBase, const TValue_t& tvFunc,
                   const TValue_t& tvArg1, const TValue_t& tvArg2) {
    TValue_t* pTop = pBase + FrameProto(pUni, pBase)->FrameSize();
    pTop[0] = tvFunc;
    pTop[1] = TValue_t::Nil();  // link slot
    pTop[2] = tvArg1;
    pTop[3] = tvArg2;
    C_Interpreter::Call(pUni->MainThread(), pTop, 2, 1);
    return pTop[0];
}

// __index / __newindex resolution chains (tables and functions).
LJX_NOINLINE TValue_t IndexSlow(C_Universe* pUni, TValue_t* pBase, const BcIns_t* pPc,
                                TValue_t tvTable, TValue_t tvKey) {
    for (int nDepth = 0; nDepth < 100; ++nDepth) {
        if (tvTable.Is(EValueTag::Table)) {
            auto* pTab = static_cast<C_GcTable*>(tvTable.AsGcPointer());
            const TValue_t* pSlot = pTab->Get(*pUni, tvKey);
            if (pSlot && !pSlot->IsNil()) return *pSlot;
            auto* pMt = pUni->Deref<C_GcTable>(pTab->m_rMetatable);
            const TValue_t* pIndex = pUni->Meta().Lookup(pMt, rt::EMetaMethod::Index);
            if (!pIndex) return TValue_t::Nil();
            if (pIndex->Is(EValueTag::Function))
                return MetaCall2(pUni, pBase, *pIndex, tvTable, tvKey);
            tvTable = *pIndex;  // __index table: repeat lookup in it
            continue;
        }
        const TValue_t* pIndex = pUni->Meta().LookupForValue(tvTable, rt::EMetaMethod::Index);
        if (!pIndex)
            ErrorAtPc(pUni, pBase, pPc, "attempt to index a %s value", TypeName(tvTable));
        if (pIndex->Is(EValueTag::Function))
            return MetaCall2(pUni, pBase, *pIndex, tvTable, tvKey);
        tvTable = *pIndex;
    }
    ErrorAtPc(pUni, pBase, pPc, "'__index' chain too long%s", "");
}

LJX_NOINLINE void NewIndexSlow(C_Universe* pUni, TValue_t* pBase, const BcIns_t* pPc,
                               TValue_t tvTable, TValue_t tvKey, TValue_t tvValue) {
    for (int nDepth = 0; nDepth < 100; ++nDepth) {
        if (tvTable.Is(EValueTag::Table)) {
            auto* pTab = static_cast<C_GcTable*>(tvTable.AsGcPointer());
            const TValue_t* pSlot = pTab->Get(*pUni, tvKey);
            if (pSlot && !pSlot->IsNil()) {  // existing key: raw store
                *const_cast<TValue_t*>(pSlot) = tvValue;
                pTab->BumpVersion();   // an inline cache may have resolved here
                return;
            }
            auto* pMt = pUni->Deref<C_GcTable>(pTab->m_rMetatable);
            const TValue_t* pNewIndex = pUni->Meta().Lookup(pMt, rt::EMetaMethod::NewIndex);
            if (!pNewIndex) {
                *pTab->Set(*pUni, tvKey) = tvValue;
                pUni->Gc().BarrierBackTable(pTab);
                return;
            }
            if (pNewIndex->Is(EValueTag::Function)) {
                TValue_t* pTop = pBase + FrameProto(pUni, pBase)->FrameSize();
                pTop[0] = *pNewIndex;
                pTop[1] = TValue_t::Nil();
                pTop[2] = tvTable;
                pTop[3] = tvKey;
                pTop[4] = tvValue;
                C_Interpreter::Call(pUni->MainThread(), pTop, 3, 0);
                return;
            }
            tvTable = *pNewIndex;
            continue;
        }
        ErrorAtPc(pUni, pBase, pPc, "attempt to index a %s value", TypeName(tvTable));
    }
    ErrorAtPc(pUni, pBase, pPc, "'__newindex' chain too long%s", "");
}

// --- inline cache for metatable-resolved field lookups ----------------------
//
// `obj:method()` and `obj.field` on a class-style table both miss on the
// receiver and resolve through its metatable's __index. That is two extra
// table searches plus a metamethod lookup, on the hottest operation in
// object-oriented Lua. The cache records what the chain produced, keyed on the
// identity AND version of both tables involved, so a hit is four compares and
// a stale entry can never be used.
LJX_FORCEINLINE InlineCache_t* CacheLine(C_Universe* pUni, TValue_t* pBase,
                                         const BcIns_t* pPc) noexcept {
    const C_GcProto* pProto = FrameProto(pUni, pBase);
    if (pProto->m_rInlineCache.IsNull()) return nullptr;
    auto* pCaches = static_cast<InlineCache_t*>(
        core::RefToPtr(pUni->ArenaBase(), pProto->m_rInlineCache));
    const std::size_t uIdx = static_cast<std::size_t>(
        pPc - 1 - reinterpret_cast<const BcIns_t*>(pProto->Bytecode()));
    return uIdx < pProto->m_uBcCount ? &pCaches[uIdx] : nullptr;
}

// Returns true and fills tvOut when the cache is valid for this receiver.
LJX_FORCEINLINE bool CacheProbe(C_Universe* pUni, const InlineCache_t* pCache,
                                const C_GcTable* pTab, TValue_t& tvOut) noexcept {
    if (!pCache || pCache->rMeta.IsNull() || pTab->m_rMetatable != pCache->rMeta)
        return false;
    const auto* pMeta = pUni->Deref<C_GcTable>(pCache->rMeta);
    if (pMeta->m_uVersion != pCache->uMetaVersion) return false;
    const auto* pIndexTab = pUni->Deref<C_GcTable>(pCache->rIndexTable);
    if (!pIndexTab || pIndexTab->m_uVersion != pCache->uIndexVersion) return false;
    tvOut = pCache->tvValue;
    return true;
}

// Resolves obj[key] through the metatable and, when the chain is a plain
// __index table, records it for next time.
LJX_NOINLINE TValue_t IndexMissCached(C_Universe* pUni, TValue_t* pBase, const BcIns_t* pPc,
                                      C_GcTable* pTab, const TValue_t& tvKey,
                                      InlineCache_t* pCache) {
    auto* pMeta = pUni->Deref<C_GcTable>(pTab->m_rMetatable);
    const TValue_t* pIndex = pUni->Meta().Lookup(pMeta, rt::EMetaMethod::Index);
    if (!pIndex) return TValue_t::Nil();
    if (pIndex->Is(EValueTag::Table)) {
        auto* pIndexTab = static_cast<C_GcTable*>(pIndex->AsGcPointer());
        const TValue_t* pSlot = pIndexTab->Get(*pUni, tvKey);
        const TValue_t tvResult = pSlot ? *pSlot : TValue_t::Nil();
        // Only a single-level __index table is cacheable: a deeper chain or a
        // function handler is re-resolved every time.
        // Cache ONLY what this single level actually produced. `!pSlot` with a
        // further metatable falls through to IndexSlow below, which walks the
        // rest of the chain — caching nil here would make every later lookup
        // of an inherited member return nil.
        if (pCache && ((pSlot && !pSlot->IsNil()) || pIndexTab->m_rMetatable.IsNull())) {
            pCache->rMeta = pTab->m_rMetatable;
            pCache->uMetaVersion = pMeta->m_uVersion;
            pCache->rIndexTable = pUni->MakeRef(pIndexTab);
            pCache->uIndexVersion = pIndexTab->m_uVersion;
            pCache->tvValue = tvResult;
        }
        if (pSlot && !pSlot->IsNil()) return tvResult;
        if (pIndexTab->m_rMetatable.IsNull()) return TValue_t::Nil();
    }
    return IndexSlow(pUni, pBase, pPc, TValue_t::GcObject(EValueTag::Table, pTab), tvKey);
}

// Arithmetic metamethod fallback.
LJX_NOINLINE TValue_t ArithSlow(C_Universe* pUni, TValue_t* pBase, const BcIns_t* pPc,
                                rt::EMetaMethod eMethod, TValue_t tvA, TValue_t tvB) {
    const TValue_t* pHandler = pUni->Meta().LookupForValue(tvA, eMethod);
    if (!pHandler) pHandler = pUni->Meta().LookupForValue(tvB, eMethod);
    if (!pHandler)
        ErrorAtPc(pUni, pBase, pPc, "attempt to perform arithmetic on a %s value",
                  TypeName(tvA.IsNumber() ? tvB : tvA));
    return MetaCall2(pUni, pBase, *pHandler, tvA, tvB);
}

// ---------------------------------------------------------------------------
// Return path: results already copied to pBase-2; uRd = result count.
// Shares the handler signature so Ret* handlers musttail into it.
// ---------------------------------------------------------------------------

// Return continuation. Contract: uRa = first result slot (relative to pBase),
// uRd = result count. The frame link is read BEFORE any result is copied —
// results land at pBase-2.. which overlaps the link at pBase-1, so the order
// is load-bearing (a 2+-result return would otherwise clobber its own link).
LJX_PRESERVE_NONE void ReturnDispatch(LJX_HANDLER_ARGS) {
    const FrameLink_t link{pBase[-1].uRaw};
    const std::uint32_t uResults = static_cast<std::uint32_t>(uRd);
    const std::uint32_t uSrc = static_cast<std::uint32_t>(uRa);
    TValue_t* pResults = pBase - 2;
    for (std::uint32_t uI = 0; uI < uResults; ++uI) pResults[uI] = pBase[uSrc + uI];

    if (!link.IsLua()) {
        // C-entry frame: results already at pBase-2; unwind to C++.
        pUni->m_uEntryResults = uResults;
        C_LuaThread* pThread = pUni->MainThread();
        pThread->m_pTop = pBase - 2 + uResults;
        return;
    }
    const BcIns_t* pRetPc = link.ReturnPc();
    const BcIns_t insCall{pRetPc[-1].uRaw};
    TValue_t* pPrevBase = pBase - 2 - insCall.A();
    const std::uint32_t uExpected = insCall.B();  // nresults+1; 0 = all
    if (uExpected == 0)
        pUni->m_uMultRes = uResults;
    else
        for (std::uint32_t uI = uResults; uI < uExpected - 1; ++uI)
            pResults[uI] = TValue_t::Nil();
    const C_GcProto* pProto = FrameProto(pUni, pPrevBase);
    const TValue_t* pPrevKBase = KBaseOf(pUni, pProto);
    const BcIns_t insNext{pRetPc->uRaw};
    BcHandler_f fnNext = pUni->Dispatch().Dynamic(insNext.uRaw & 0xff);
    LJX_MUSTTAIL return fnNext(pPrevBase, pRetPc + 1, insNext.A(), insNext.D(), pUni,
                               pPrevKBase);
}

// ---------------------------------------------------------------------------
// Call setup: func at pBase[uRa], link written, dispatch on callee header.
// uRd carries the argument count into the header handlers.
// ---------------------------------------------------------------------------

LJX_PRESERVE_NONE void CallDispatch(LJX_HANDLER_ARGS) {
    // uRa = func slot, uRd = nargs. pPc already points past the call.
    TValue_t& tvFunc = pBase[uRa];
    if (!tvFunc.Is(EValueTag::Function)) [[unlikely]] {
        // v1: no __call metamethod.
        ErrorAtPc(pUni, pBase, pPc, "attempt to call a %s value", TypeName(tvFunc));
    }
    auto* pFn = static_cast<C_GcFunction*>(tvFunc.AsGcPointer());
    pBase[uRa + 1] = TValue_t{FrameLink_t::FromReturnPc(pPc).uRaw};
    TValue_t* pNewBase = pBase + uRa + 2;
    const auto* pCalleePc = reinterpret_cast<const BcIns_t*>(pFn->m_pPc);
    const BcIns_t insHeader{pCalleePc->uRaw};
    BcHandler_f fnHeader = pUni->Dispatch().Dynamic(insHeader.uRaw & 0xff);
    LJX_MUSTTAIL return fnHeader(pNewBase, pCalleePc + 1, insHeader.A(), uRd, pUni, pKBase);
}

// ---------------------------------------------------------------------------
// Opcode handlers.
// ---------------------------------------------------------------------------

#define LJX_H(name) LJX_PRESERVE_NONE void Op##name(LJX_HANDLER_ARGS)

LJX_H(FuncF) {
    // uRa = framesize, uRd = nargs (from CallDispatch / entry).
    const std::uint32_t uFrame = static_cast<std::uint32_t>(uRa);
    C_LuaThread* pThread = pUni->MainThread();
    if (pBase + uFrame + kStackExtraSlots > pThread->m_pMaxStack) [[unlikely]]
        RaiseError(*pUni, "stack overflow");
    auto* pProto = const_cast<C_GcProto*>(C_GcProto::FromBytecode(
        reinterpret_cast<const std::uint32_t*>(pPc - 1)));

    // Whole-function JIT: a numeric-closed function with number arguments can
    // run entirely as native code — recursion included — with no Lua frame and
    // no possibility of GC. The type check here IS the only guard the compiled
    // body needs, which is what makes call-heavy numeric code compilable.
    if (pProto->m_pNative != reinterpret_cast<void*>(1) && !pUni->m_uRecording &&
        static_cast<std::uint32_t>(uRd) == pProto->ParamCount()) [[unlikely]] {
        bool bAllNumbers = true;
        for (std::uint32_t uI = 0; uI < uRd; ++uI)
            bAllNumbers &= pBase[uI].IsDouble();
        if (bAllNumbers) {
            const jit::CompiledFunc_t* pCompiled =
                pUni->FuncJit()->LookupOrTick(pProto, FrameFunc(pBase));
            if (pCompiled) {
                double flResult;
                switch (pCompiled->uArity) {
                    case 1:
                        flResult = reinterpret_cast<jit::NativeFn1_f>(pCompiled->pCode)(
                            pBase[0].AsDouble());
                        break;
                    case 2:
                        flResult = reinterpret_cast<jit::NativeFn2_f>(pCompiled->pCode)(
                            pBase[0].AsDouble(), pBase[1].AsDouble());
                        break;
                    case 3:
                        flResult = reinterpret_cast<jit::NativeFn3_f>(pCompiled->pCode)(
                            pBase[0].AsDouble(), pBase[1].AsDouble(), pBase[2].AsDouble());
                        break;
                    default:
                        flResult = reinterpret_cast<jit::NativeFn4_f>(pCompiled->pCode)(
                            pBase[0].AsDouble(), pBase[1].AsDouble(), pBase[2].AsDouble(),
                            pBase[3].AsDouble());
                        break;
                }
                pBase[0] = TValue_t::Number(flResult);
                uRa = 0;
                uRd = 1;
                LJX_MUSTTAIL return ReturnDispatch(LJX_PASS_ARGS);
            }
        }
    }

    // Only MISSING PARAMETERS are cleared. Temp slots are left alone — the
    // collector nils everything above the live top (see m_pHighWater), so the
    // marker never sees a stale slot, and calls stop paying for the frame.
    for (std::uint32_t uI = static_cast<std::uint32_t>(uRd), uP = pProto->ParamCount();
         uI < uP; ++uI)
        pBase[uI] = TValue_t::Nil();
    if (pBase + uFrame > pThread->m_pHighWater) pThread->m_pHighWater = pBase + uFrame;
    if (pUni->Gc().NeedsStep() && !pUni->m_uRecording) [[unlikely]] {
        pThread->m_pBase = pBase;
        pThread->m_pTop = pBase + uFrame;
        pUni->Gc().CollectNow();
    }
    pKBase = KBaseOf(pUni, pProto);
    LJX_NEXT();
}

LJX_H(FuncC) {
    // uRd = nargs. C functions see args at base[0..n); results written at
    // base[0..] and their count returned.
    C_GcFunction* pFn = FrameFunc(pBase);
    C_LuaThread* pThread = pUni->MainThread();
    pThread->m_pBase = pBase;
    pThread->m_pTop = pBase + uRd;
    const std::int32_t nResults = pFn->CFunc()(reinterpret_cast<lua_State*>(pThread));
    // C results sit at pBase[0..n); ReturnDispatch copies them down after
    // reading the link (uRa = 0 source slot, uRd = count).
    uRa = 0;
    uRd = static_cast<std::uint64_t>(nResults);
    LJX_MUSTTAIL return ReturnDispatch(LJX_PASS_ARGS);
}

LJX_H(FuncCW) { LJX_MUSTTAIL return OpFuncC(LJX_PASS_ARGS); }

// --- constants --------------------------------------------------------------

LJX_H(KShort) {
    pBase[uRa] = TValue_t::Number(static_cast<double>(static_cast<std::int16_t>(uRd)));
    LJX_NEXT();
}
LJX_H(KNum) {
    pBase[uRa] = pKBase[uRd];
    LJX_NEXT();
}
LJX_H(KStr) {
    pBase[uRa] = TValue_t::GcObject(EValueTag::String,
                                    KgcString(pUni, pKBase, static_cast<std::uint32_t>(uRd)));
    LJX_NEXT();
}
LJX_H(KPri) {
    pBase[uRa] = uRd == 0 ? TValue_t::Nil() : TValue_t::Boolean(uRd == 2);
    LJX_NEXT();
}
LJX_H(KNil) {
    for (std::uint64_t uI = uRa; uI <= uRd; ++uI) pBase[uI] = TValue_t::Nil();
    LJX_NEXT();
}
LJX_H(KCData) { ErrorAtPc(pUni, pBase, pPc, "cdata constants not supported%s", ""); }

// --- moves & unary ----------------------------------------------------------

LJX_H(Mov) {
    pBase[uRa] = pBase[uRd];
    LJX_NEXT();
}
LJX_H(Not) {
    pBase[uRa] = TValue_t::Boolean(!pBase[uRd].IsTruthy());
    LJX_NEXT();
}
LJX_H(Unm) {
    const TValue_t tvValue = pBase[uRd];
    if (tvValue.IsDouble()) [[likely]] {
        pBase[uRa] = TValue_t::Number(-tvValue.AsDouble());
    } else {
        pBase[uRa] = ArithSlow(pUni, pBase, pPc, rt::EMetaMethod::Unm, tvValue, tvValue);
    }
    LJX_NEXT();
}
LJX_H(Len) {
    const TValue_t tvValue = pBase[uRd];
    if (tvValue.Is(EValueTag::String)) {
        pBase[uRa] = TValue_t::Number(
            static_cast<double>(static_cast<C_GcString*>(tvValue.AsGcPointer())->Length()));
    } else if (tvValue.Is(EValueTag::Table)) {
        pBase[uRa] = TValue_t::Number(static_cast<double>(
            static_cast<C_GcTable*>(tvValue.AsGcPointer())->Length(*pUni)));
    } else {
        ErrorAtPc(pUni, pBase, pPc, "attempt to get length of a %s value", TypeName(tvValue));
    }
    LJX_NEXT();
}

// --- arithmetic -------------------------------------------------------------

#define LJX_ARITH_VV(name, expr, mm)                                            \
    LJX_H(name##VV) {                                                           \
        const TValue_t tvB = pBase[uRd >> 8];                                   \
        const TValue_t tvC = pBase[uRd & 0xff];                                 \
        if (BothNumbers(tvB, tvC)) [[likely]] {                                 \
            const double flA = LoadNum(tvB), flB = LoadNum(tvC);                \
            pBase[uRa] = TValue_t::Number(expr);                                \
        } else {                                                                \
            pBase[uRa] = ArithSlow(pUni, pBase, pPc, rt::EMetaMethod::mm, tvB, tvC); \
        }                                                                       \
        LJX_NEXT();                                                             \
    }                                                                           \
    LJX_H(name##VN) {                                                           \
        const TValue_t tvB = pBase[uRd >> 8];                                   \
        const TValue_t tvC = pKBase[uRd & 0xff];                                \
        if (tvB.IsDouble()) [[likely]] {                                        \
            const double flA = LoadNum(tvB), flB = LoadNum(tvC);                \
            pBase[uRa] = TValue_t::Number(expr);                                \
        } else {                                                                \
            pBase[uRa] = ArithSlow(pUni, pBase, pPc, rt::EMetaMethod::mm, tvB, tvC); \
        }                                                                       \
        LJX_NEXT();                                                             \
    }                                                                           \
    LJX_H(name##NV) {                                                           \
        const TValue_t tvC = pBase[uRd >> 8];                                   \
        const TValue_t tvB = pKBase[uRd & 0xff];                                \
        if (tvC.IsDouble()) [[likely]] {                                        \
            const double flA = LoadNum(tvB), flB = LoadNum(tvC);                \
            pBase[uRa] = TValue_t::Number(expr);                                \
        } else {                                                                \
            pBase[uRa] = ArithSlow(pUni, pBase, pPc, rt::EMetaMethod::mm, tvB, tvC); \
        }                                                                       \
        LJX_NEXT();                                                             \
    }

LJX_ARITH_VV(Add, flA + flB, Add)
LJX_ARITH_VV(Sub, flA - flB, Sub)
LJX_ARITH_VV(Mul, flA* flB, Mul)
LJX_ARITH_VV(Div, flA / flB, Div)
LJX_ARITH_VV(Mod, flA - std::floor(flA / flB) * flB, Mod)

LJX_H(Pow) {
    const TValue_t tvB = pBase[uRd >> 8];
    const TValue_t tvC = pBase[uRd & 0xff];
    if (BothNumbers(tvB, tvC)) [[likely]] {
        pBase[uRa] = TValue_t::Number(std::pow(tvB.AsDouble(), tvC.AsDouble()));
    } else {
        pBase[uRa] = ArithSlow(pUni, pBase, pPc, rt::EMetaMethod::Pow, tvB, tvC);
    }
    LJX_NEXT();
}

// Fast number→text: integer path is hand-rolled (the overwhelmingly common
// case in concatenation); the general path falls back to %.14g.
LJX_FORCEINLINE std::uint32_t FormatNumber(char* pOut, double flValue) noexcept {
    std::int32_t nInt;
    if (core::NumToInt32Check(flValue, nInt)) {
        char* pCursor = pOut;
        std::uint32_t uAbs;
        if (nInt < 0) {
            *pCursor++ = '-';
            uAbs = static_cast<std::uint32_t>(-static_cast<std::int64_t>(nInt));
        } else {
            uAbs = static_cast<std::uint32_t>(nInt);
        }
        char vDigits[10];
        int nCount = 0;
        do {
            vDigits[nCount++] = static_cast<char>('0' + uAbs % 10);
            uAbs /= 10;
        } while (uAbs);
        while (nCount) *pCursor++ = vDigits[--nCount];
        return static_cast<std::uint32_t>(pCursor - pOut);
    }
    return static_cast<std::uint32_t>(std::snprintf(pOut, 40, "%.14g", flValue));
}

// Out of line: musttail cannot cross a scope with non-trivial destructors.
LJX_NOINLINE C_GcString* CatSlow(C_Universe* pUni, TValue_t* pBase, const BcIns_t* pPc,
                                 std::uint32_t uFirst, std::uint32_t uLast) {
    // Stack buffer covers virtually all concatenations; spill to the heap
    // only for large results. No std::string on this path.
    char vStack[512];
    char* pBuf = vStack;
    std::size_t uCapacity = sizeof vStack;
    std::size_t uLen = 0;
    char* pHeap = nullptr;
    for (std::uint32_t uI = uFirst; uI <= uLast; ++uI) {
        const TValue_t tvValue = pBase[uI];
        const char* pPiece;
        std::size_t uPieceLen;
        char vNumBuf[40];
        if (tvValue.Is(EValueTag::String)) {
            auto* pStr = static_cast<C_GcString*>(tvValue.AsGcPointer());
            pPiece = pStr->Data();
            uPieceLen = pStr->Length();
        } else if (tvValue.IsDouble()) {
            uPieceLen = FormatNumber(vNumBuf, tvValue.AsDouble());
            pPiece = vNumBuf;
        } else {
            std::free(pHeap);
            ErrorAtPc(pUni, pBase, pPc, "attempt to concatenate a %s value",
                      TypeName(tvValue));
        }
        if (uLen + uPieceLen > uCapacity) [[unlikely]] {
            uCapacity = (uLen + uPieceLen) * 2;
            char* pNew = static_cast<char*>(std::malloc(uCapacity));
            std::memcpy(pNew, pBuf, uLen);
            std::free(pHeap);
            pHeap = pBuf = pNew;
        }
        std::memcpy(pBuf + uLen, pPiece, uPieceLen);
        uLen += uPieceLen;
    }
    C_GcString* pResult = pUni->Interner().Intern(std::string_view(pBuf, uLen));
    std::free(pHeap);
    return pResult;
}

LJX_H(Cat) {
    if (pUni->Gc().NeedsStep()) [[unlikely]] GcCheck(pUni, pBase);
    pBase[uRa] = TValue_t::GcObject(
        EValueTag::String,
        CatSlow(pUni, pBase, pPc, static_cast<std::uint32_t>(uRd >> 8),
                static_cast<std::uint32_t>(uRd & 0xff)));
    LJX_NEXT();
}

// --- comparisons (each consumes the following Jmp) --------------------------

#define LJX_COMPARE_TAIL(bTaken)                                                \
    do {                                                                        \
        const BcIns_t insJmp{pPc->uRaw};                                        \
        pPc = (bTaken) ? pPc + 1 + insJmp.JumpTarget() : pPc + 1;               \
        LJX_NEXT();                                                             \
    } while (0)

LJX_NOINLINE bool CompareSlow(C_Universe* pUni, TValue_t* pBase, const BcIns_t* pPc,
                              TValue_t tvA, TValue_t tvB, bool bLessEqual) {
    if (tvA.Is(EValueTag::String) && tvB.Is(EValueTag::String)) {
        auto* pStrA = static_cast<C_GcString*>(tvA.AsGcPointer());
        auto* pStrB = static_cast<C_GcString*>(tvB.AsGcPointer());
        const std::uint32_t uMin =
            pStrA->Length() < pStrB->Length() ? pStrA->Length() : pStrB->Length();
        const int nCmp = std::memcmp(pStrA->Data(), pStrB->Data(), uMin);
        if (nCmp != 0) return bLessEqual ? nCmp < 0 : nCmp < 0;
        const bool bShorter = pStrA->Length() < pStrB->Length();
        return bLessEqual ? (bShorter || pStrA->Length() == pStrB->Length()) : bShorter;
    }
    ErrorAtPc(pUni, pBase, pPc, "attempt to compare %s values", TypeName(tvA));
}

LJX_H(IsLt) {
    const TValue_t tvA = pBase[uRa], tvB = pBase[uRd];
    bool bTaken;
    if (BothNumbers(tvA, tvB)) [[likely]]
        bTaken = LoadNum(tvA) < LoadNum(tvB);
    else
        bTaken = CompareSlow(pUni, pBase, pPc, tvA, tvB, false);
    LJX_COMPARE_TAIL(bTaken);
}
LJX_H(IsGe) {
    const TValue_t tvA = pBase[uRa], tvB = pBase[uRd];
    bool bTaken;
    if (BothNumbers(tvA, tvB)) [[likely]]
        bTaken = !(LoadNum(tvA) < LoadNum(tvB));  // NaN → taken
    else
        bTaken = !CompareSlow(pUni, pBase, pPc, tvA, tvB, false);
    LJX_COMPARE_TAIL(bTaken);
}
LJX_H(IsLe) {
    const TValue_t tvA = pBase[uRa], tvB = pBase[uRd];
    bool bTaken;
    if (BothNumbers(tvA, tvB)) [[likely]]
        bTaken = LoadNum(tvA) <= LoadNum(tvB);
    else
        bTaken = CompareSlow(pUni, pBase, pPc, tvA, tvB, true);
    LJX_COMPARE_TAIL(bTaken);
}
LJX_H(IsGt) {
    const TValue_t tvA = pBase[uRa], tvB = pBase[uRd];
    bool bTaken;
    if (BothNumbers(tvA, tvB)) [[likely]]
        bTaken = !(LoadNum(tvA) <= LoadNum(tvB));  // NaN → taken
    else
        bTaken = !CompareSlow(pUni, pBase, pPc, tvA, tvB, true);
    LJX_COMPARE_TAIL(bTaken);
}

LJX_FORCEINLINE bool ValueEquals(const TValue_t& tvA, const TValue_t& tvB) noexcept {
    if (tvA.uRaw == tvB.uRaw) return !(tvA.IsDouble() && tvA.AsDouble() != tvA.AsDouble());
    return tvA.IsDouble() && tvB.IsDouble() && tvA.AsDouble() == tvB.AsDouble();
}

LJX_H(IsEqV) { LJX_COMPARE_TAIL(ValueEquals(pBase[uRa], pBase[uRd])); }
LJX_H(IsNeV) { LJX_COMPARE_TAIL(!ValueEquals(pBase[uRa], pBase[uRd])); }
LJX_H(IsEqS) {
    const TValue_t tvKey = TValue_t::GcObject(
        EValueTag::String, KgcString(pUni, pKBase, static_cast<std::uint32_t>(uRd)));
    LJX_COMPARE_TAIL(pBase[uRa].uRaw == tvKey.uRaw);
}
LJX_H(IsNeS) {
    const TValue_t tvKey = TValue_t::GcObject(
        EValueTag::String, KgcString(pUni, pKBase, static_cast<std::uint32_t>(uRd)));
    LJX_COMPARE_TAIL(pBase[uRa].uRaw != tvKey.uRaw);
}
LJX_H(IsEqN) { LJX_COMPARE_TAIL(ValueEquals(pBase[uRa], pKBase[uRd])); }
LJX_H(IsNeN) { LJX_COMPARE_TAIL(!ValueEquals(pBase[uRa], pKBase[uRd])); }
LJX_H(IsEqP) {
    const TValue_t tvPrim = uRd == 0 ? TValue_t::Nil() : TValue_t::Boolean(uRd == 2);
    LJX_COMPARE_TAIL(pBase[uRa].uRaw == tvPrim.uRaw);
}
LJX_H(IsNeP) {
    const TValue_t tvPrim = uRd == 0 ? TValue_t::Nil() : TValue_t::Boolean(uRd == 2);
    LJX_COMPARE_TAIL(pBase[uRa].uRaw != tvPrim.uRaw);
}

// --- tests (value-preserving variants copy on the TAKEN path) ---------------

LJX_H(IsT) { LJX_COMPARE_TAIL(pBase[uRd].IsTruthy()); }
LJX_H(IsF) { LJX_COMPARE_TAIL(!pBase[uRd].IsTruthy()); }
LJX_H(IsTC) {
    const bool bTaken = pBase[uRd].IsTruthy();
    if (bTaken) pBase[uRa] = pBase[uRd];
    LJX_COMPARE_TAIL(bTaken);
}
LJX_H(IsFC) {
    const bool bTaken = !pBase[uRd].IsTruthy();
    if (bTaken) pBase[uRa] = pBase[uRd];
    LJX_COMPARE_TAIL(bTaken);
}

// --- upvalues & closures ----------------------------------------------------

LJX_FORCEINLINE TValue_t* UpvalSlot(C_Universe* pUni, C_GcUpvalue* pUpval) noexcept {
    return static_cast<TValue_t*>(core::RefToPtr(pUni->ArenaBase(), pUpval->m_rValue));
}

LJX_H(UGet) {
    C_GcFunction* pFn = FrameFunc(pBase);
    auto* pUpval = pUni->Deref<C_GcUpvalue>(pFn->UpvalRefs()[uRd]);
    pBase[uRa] = *UpvalSlot(pUni, pUpval);
    LJX_NEXT();
}
LJX_H(USetV) {
    C_GcFunction* pFn = FrameFunc(pBase);
    auto* pUpval = pUni->Deref<C_GcUpvalue>(pFn->UpvalRefs()[uRa]);
    *UpvalSlot(pUni, pUpval) = pBase[uRd];
    pUni->Gc().BarrierUpvalue(pUpval, pBase[uRd]);
    LJX_NEXT();
}
LJX_H(USetS) { ErrorAtPc(pUni, pBase, pPc, "USetS not emitted%s", ""); }
LJX_H(USetN) { ErrorAtPc(pUni, pBase, pPc, "USetN not emitted%s", ""); }
LJX_H(USetP) { ErrorAtPc(pUni, pBase, pPc, "USetP not emitted%s", ""); }

// Find-or-create an open upvalue for a stack slot (list sorted by address).
C_GcUpvalue* FindUpvalue(C_Universe* pUni, C_LuaThread* pThread, TValue_t* pSlot) {
    core::GcRef_t* pAnchor = &pThread->m_rOpenUpvals;
    while (!pAnchor->IsNull()) {
        auto* pUpval = pUni->Deref<C_GcUpvalue>(*pAnchor);
        TValue_t* pTarget = UpvalSlot(pUni, pUpval);
        if (pTarget == pSlot) return pUpval;
        if (pTarget < pSlot) break;  // list sorted descending
        pAnchor = reinterpret_cast<core::GcRef_t*>(&pUpval->m_tvClosed);
    }
    auto* pNew = static_cast<C_GcUpvalue*>(
        pUni->Gc().AllocObject(EGcObjectType::UpValue, sizeof(C_GcUpvalue)));
    pNew->m_Header.uExtra1 = 0;  // open
    pNew->m_rValue = core::PtrToRef(pUni->ArenaBase(), pSlot);
    pNew->m_uDHash = static_cast<std::uint32_t>(pUni->Prng().Next());
    *reinterpret_cast<core::GcRef_t*>(&pNew->m_tvClosed) = *pAnchor;  // list link
    *pAnchor = pUni->MakeRef(pNew);
    return pNew;
}

LJX_H(FNew) {
    if (pUni->Gc().NeedsStep()) [[unlikely]] GcCheck(pUni, pBase);
    const auto* pKgc = reinterpret_cast<const core::GcRef_t*>(pKBase);
    auto* pProto = pUni->Deref<C_GcProto>(pKgc[-static_cast<std::int32_t>(uRd + 1)]);
    const std::uint32_t uUpvals = pProto->m_uUpvalCount;
    auto* pFn = static_cast<C_GcFunction*>(pUni->Gc().AllocObject(
        EGcObjectType::Function, C_GcFunction::LuaAllocSize(uUpvals)));
    pFn->m_Header.uExtra1 = 0;  // Lua closure
    pFn->m_Header.uExtra2 = static_cast<std::uint8_t>(uUpvals);
    pFn->m_rEnv = pUni->MakeRef(pUni->Globals());
    pFn->m_pPc = pProto->Bytecode();
    C_GcFunction* pParent = FrameFunc(pBase);
    const auto* pDescs = static_cast<const std::uint16_t*>(
        core::RefToPtr(pUni->ArenaBase(), pProto->m_rUpvalDescs));
    for (std::uint32_t uI = 0; uI < uUpvals; ++uI) {
        const std::uint16_t uDesc = pDescs[uI];
        if (uDesc & 0x8000) {
            C_GcUpvalue* pUpval =
                FindUpvalue(pUni, pUni->MainThread(), pBase + (uDesc & 0xff));
            pFn->UpvalRefs()[uI] = pUni->MakeRef(pUpval);
        } else {
            pFn->UpvalRefs()[uI] = pParent->UpvalRefs()[uDesc];
        }
    }
    pBase[uRa] = TValue_t::GcObject(EValueTag::Function, pFn);
    LJX_NEXT();
}

LJX_H(UClo) {
    // Close upvalues pointing at slots >= pBase + uRa, then jump by D.
    C_LuaThread* pThread = pUni->MainThread();
    core::GcRef_t* pAnchor = &pThread->m_rOpenUpvals;
    TValue_t* pLimit = pBase + uRa;
    while (!pAnchor->IsNull()) {
        auto* pUpval = pUni->Deref<C_GcUpvalue>(*pAnchor);
        TValue_t* pTarget = UpvalSlot(pUni, pUpval);
        if (pTarget < pLimit) break;  // sorted descending: done
        *pAnchor = *reinterpret_cast<core::GcRef_t*>(&pUpval->m_tvClosed);  // unlink
        pUpval->m_tvClosed = *pTarget;
        pUpval->m_rValue =
            core::PtrToRef(pUni->ArenaBase(), &pUpval->m_tvClosed);
        pUpval->m_Header.uExtra1 = 1;  // closed
    }
    pPc += static_cast<std::int32_t>(uRd) - static_cast<std::int32_t>(kJumpBias);
    LJX_NEXT();
}

// --- globals & tables -------------------------------------------------------

LJX_H(GGet) {
    C_GcString* pKey = KgcString(pUni, pKBase, static_cast<std::uint32_t>(uRd));
    const TValue_t* pSlot = pUni->Globals()->GetStr(*pUni, pKey);
    pBase[uRa] = pSlot ? *pSlot : TValue_t::Nil();
    LJX_NEXT();
}
LJX_H(GSet) {
    C_GcString* pKey = KgcString(pUni, pKBase, static_cast<std::uint32_t>(uRd));
    *pUni->Globals()->Set(*pUni,
                          TValue_t::GcObject(EValueTag::String, pKey)) = pBase[uRa];
    pUni->Gc().BarrierBackTable(pUni->Globals());
    LJX_NEXT();
}

LJX_H(TNew) {
    if (pUni->Gc().NeedsStep()) [[unlikely]] GcCheck(pUni, pBase);
    const std::uint32_t uArrayHint = static_cast<std::uint32_t>(uRd) & 0x7ff;
    const std::uint32_t uHashLog = static_cast<std::uint32_t>(uRd) >> 11;
    pBase[uRa] = TValue_t::GcObject(
        EValueTag::Table, C_GcTable::New(*pUni, uArrayHint, uHashLog));
    LJX_NEXT();
}
LJX_H(TDup) { ErrorAtPc(pUni, pBase, pPc, "TDup not emitted%s", ""); }

LJX_H(TGetV) {
    const TValue_t tvTable = pBase[uRd >> 8];
    const TValue_t tvKey = pBase[uRd & 0xff];
    if (tvTable.Is(EValueTag::Table)) [[likely]] {
        auto* pTab = static_cast<C_GcTable*>(tvTable.AsGcPointer());
        // Inline array fast path (integer key in range).
        if (tvKey.IsDouble()) {
            const double flKey = tvKey.AsDouble();
            const auto uKey = static_cast<std::uint32_t>(static_cast<std::int64_t>(flKey));
            if (static_cast<double>(uKey) == flKey && uKey < pTab->m_uArraySize) {
                pBase[uRa] =
                    static_cast<TValue_t*>(core::RefToPtr(pUni->ArenaBase(), pTab->m_rArray))[uKey];
                LJX_NEXT();
            }
        }
        const TValue_t* pSlot = pTab->Get(*pUni, tvKey);
        if (pSlot && !pSlot->IsNil()) [[likely]] {
            pBase[uRa] = *pSlot;
            LJX_NEXT();
        }
        // Negative-cache fast miss: no __index → plain nil.
        if (pTab->m_rMetatable.IsNull()) {
            pBase[uRa] = TValue_t::Nil();
            LJX_NEXT();
        }
    }
    pBase[uRa] = IndexSlow(pUni, pBase, pPc, tvTable, tvKey);
    LJX_NEXT();
}
LJX_H(TGetS) {
    const TValue_t tvTable = pBase[uRd >> 8];
    C_GcString* pKey = KgcString(pUni, pKBase, static_cast<std::uint32_t>(uRd & 0xff));
    if (tvTable.Is(EValueTag::Table)) [[likely]] {
        auto* pTab = static_cast<C_GcTable*>(tvTable.AsGcPointer());
        const TValue_t* pSlot = pTab->GetStr(*pUni, pKey);
        if (pSlot && !pSlot->IsNil()) [[likely]] {
            pBase[uRa] = *pSlot;
            LJX_NEXT();
        }
        if (pTab->m_rMetatable.IsNull()) {
            pBase[uRa] = TValue_t::Nil();
            LJX_NEXT();
        }
        // Missed on the receiver: this is the method-lookup shape, so try the
        // inline cache before walking the metatable chain again.
        InlineCache_t* pCache = CacheLine(pUni, pBase, pPc);
        TValue_t tvCached;
        if (CacheProbe(pUni, pCache, pTab, tvCached)) [[likely]] {
            pBase[uRa] = tvCached;
            LJX_NEXT();
        }
        pBase[uRa] = IndexMissCached(pUni, pBase, pPc, pTab,
                                     TValue_t::GcObject(EValueTag::String, pKey), pCache);
        LJX_NEXT();
    }
    pBase[uRa] =
        IndexSlow(pUni, pBase, pPc, tvTable, TValue_t::GcObject(EValueTag::String, pKey));
    LJX_NEXT();
}
LJX_H(TGetB) {
    const TValue_t tvTable = pBase[uRd >> 8];
    const std::uint32_t uKey = static_cast<std::uint32_t>(uRd & 0xff);
    if (tvTable.Is(EValueTag::Table)) [[likely]] {
        auto* pTab = static_cast<C_GcTable*>(tvTable.AsGcPointer());
        const TValue_t* pSlot = pTab->GetInt(*pUni, uKey);
        if (pSlot && !pSlot->IsNil()) [[likely]] {
            pBase[uRa] = *pSlot;
            LJX_NEXT();
        }
        if (pTab->m_rMetatable.IsNull()) {
            pBase[uRa] = TValue_t::Nil();
            LJX_NEXT();
        }
    }
    pBase[uRa] = IndexSlow(pUni, pBase, pPc, tvTable,
                           TValue_t::Number(static_cast<double>(uKey)));
    LJX_NEXT();
}

LJX_FORCEINLINE bool ArrayStoreFast(C_Universe* pUni, const TValue_t& tvTable,
                                    const TValue_t& tvKey, const TValue_t& tvValue) noexcept {
    if (!tvTable.Is(EValueTag::Table) || !tvKey.IsDouble()) return false;
    auto* pTab = static_cast<C_GcTable*>(tvTable.AsGcPointer());
    const double flKey = tvKey.AsDouble();
    const auto uKey = static_cast<std::uint32_t>(static_cast<std::int64_t>(flKey));
    if (static_cast<double>(uKey) != flKey || uKey >= pTab->m_uArraySize) return false;
    static_cast<TValue_t*>(core::RefToPtr(pUni->ArenaBase(), pTab->m_rArray))[uKey] = tvValue;
    return true;
}

LJX_H(TSetV) {
    const TValue_t tvTable = pBase[uRd >> 8];
    const TValue_t tvKey = pBase[uRd & 0xff];
    if (ArrayStoreFast(pUni, tvTable, tvKey, pBase[uRa])) [[likely]] LJX_NEXT();
    if (tvTable.Is(EValueTag::Table)) {
        auto* pTab = static_cast<C_GcTable*>(tvTable.AsGcPointer());
        const TValue_t* pSlot = pTab->Get(*pUni, tvKey);
        if (pSlot && !pSlot->IsNil()) {          // existing key: raw store
            *const_cast<TValue_t*>(pSlot) = pBase[uRa];
            pTab->BumpVersion();
            pUni->Gc().BarrierBackTable(pTab);
            LJX_NEXT();
        }
    }
    NewIndexSlow(pUni, pBase, pPc, tvTable, tvKey, pBase[uRa]);
    LJX_NEXT();
}
LJX_H(TSetS) {
    const TValue_t tvTable = pBase[uRd >> 8];
    C_GcString* pKey = KgcString(pUni, pKBase, static_cast<std::uint32_t>(uRd & 0xff));
    if (tvTable.Is(EValueTag::Table)) [[likely]] {
        auto* pTab = static_cast<C_GcTable*>(tvTable.AsGcPointer());
        // An existing key is a raw store: no metamethod can intercept it.
        const TValue_t* pSlot = pTab->GetStr(*pUni, pKey);
        if (pSlot && !pSlot->IsNil()) [[likely]] {
            *const_cast<TValue_t*>(pSlot) = pBase[uRa];
            pTab->BumpVersion();   // a cache may have resolved through this table
            pUni->Gc().BarrierBackTable(pTab);
            LJX_NEXT();
        }
    }
    NewIndexSlow(pUni, pBase, pPc, tvTable,
                 TValue_t::GcObject(EValueTag::String, pKey), pBase[uRa]);
    LJX_NEXT();
}
LJX_H(TSetB) {
    const TValue_t tvTable = pBase[uRd >> 8];
    NewIndexSlow(pUni, pBase, pPc, tvTable,
                 TValue_t::Number(static_cast<double>(uRd & 0xff)), pBase[uRa]);
    LJX_NEXT();
}
LJX_H(TSetM) { ErrorAtPc(pUni, pBase, pPc, "TSetM not emitted%s", ""); }

// --- calls & returns --------------------------------------------------------

LJX_H(Call) {
    uRd = (uRd & 0xff) - 1;  // nargs from C field
    LJX_MUSTTAIL return CallDispatch(LJX_PASS_ARGS);
}
LJX_H(CallM) {
    uRd = (uRd & 0xff) + pUni->m_uMultRes;  // fixed args + expanded results
    LJX_MUSTTAIL return CallDispatch(LJX_PASS_ARGS);
}
LJX_H(CallT) {
    // Tailcall: move func+args down over the current frame; keep the link.
    const std::uint32_t uArgs = static_cast<std::uint32_t>(uRd) - 1;
    TValue_t* pFuncSlot = pBase + uRa;
    pBase[-2] = *pFuncSlot;
    for (std::uint32_t uI = 0; uI < uArgs; ++uI) pBase[uI] = pFuncSlot[2 + uI];
    if (!pBase[-2].Is(EValueTag::Function)) [[unlikely]]
        ErrorAtPc(pUni, pBase, pPc, "attempt to call a %s value", TypeName(pBase[-2]));
    auto* pFn = static_cast<C_GcFunction*>(pBase[-2].AsGcPointer());
    const auto* pCalleePc = reinterpret_cast<const BcIns_t*>(pFn->m_pPc);
    const BcIns_t insHeader{pCalleePc->uRaw};
    BcHandler_f fnHeader = pUni->Dispatch().Dynamic(insHeader.uRaw & 0xff);
    LJX_MUSTTAIL return fnHeader(pBase, pCalleePc + 1, insHeader.A(), uArgs, pUni, pKBase);
}
LJX_H(CallMT) {
    uRd = (uRd & 0xff) + pUni->m_uMultRes + 1;  // reuse CallT with adjusted count
    LJX_MUSTTAIL return OpCallT(LJX_PASS_ARGS);
}

// Ret handlers pass (uRa = first result slot, uRd = count); ReturnDispatch
// reads the link before copying, so results may overlap the link slot.
LJX_H(Ret0) {
    uRa = 0;
    uRd = 0;
    LJX_MUSTTAIL return ReturnDispatch(LJX_PASS_ARGS);
}
LJX_H(Ret1) {
    uRd = 1;
    LJX_MUSTTAIL return ReturnDispatch(LJX_PASS_ARGS);
}
LJX_H(Ret) {
    uRd = static_cast<std::uint64_t>(uRd) - 1;
    LJX_MUSTTAIL return ReturnDispatch(LJX_PASS_ARGS);
}
LJX_H(RetM) {
    uRd = static_cast<std::uint64_t>(uRd) + pUni->m_uMultRes;
    LJX_MUSTTAIL return ReturnDispatch(LJX_PASS_ARGS);
}

// --- loops ------------------------------------------------------------------

LJX_H(ForI) {
    TValue_t* pSlots = pBase + uRa;
    if (!(pSlots[0].IsDouble() && pSlots[1].IsDouble() && pSlots[2].IsDouble()))
        [[unlikely]]
        ErrorAtPc(pUni, pBase, pPc, "'for' initial value must be a number%s", "");
    // Hot counted loop → native code. On success it runs every iteration and
    // we resume past the loop; a failed entry guard (return 1) falls back to
    // the interpreter transparently.
    if (jit::C_LoopJit* pJit = pUni->LoopJit(); pJit && !pUni->m_uRecording) [[likely]] {
        jit::CompiledLoop_f fnLoop =
            pJit->LookupOrTick(pPc - 1, static_cast<std::uint8_t>(uRa));
        if (fnLoop) {
            const TValue_t* pKNum = pKBase;  // number constants base
            if (fnLoop(pBase, reinterpret_cast<const double*>(pKNum)) == 0) {
                const BcIns_t insSelf{pPc[-1].uRaw};
                pPc += insSelf.JumpTarget();  // loop complete: resume after it
                LJX_NEXT();
            }
            // Guard failed: fall through to the interpreter for this run.
        }
    }
    const double flIdx = pSlots[0].AsDouble();
    const double flStop = pSlots[1].AsDouble();
    const double flStep = pSlots[2].AsDouble();
    pSlots[3] = pSlots[0];
    const bool bExit = flStep >= 0 ? flIdx > flStop : flIdx < flStop;
    const BcIns_t insSelf{pPc[-1].uRaw};
    if (bExit) pPc += insSelf.JumpTarget();
    LJX_NEXT();
}
// Runs a compiled trace and reports where the interpreter picks up. The exit
// stub has already written every live value back to the Lua stack, so the only
// state to re-establish here is the frame base and the high-water mark that
// the collector uses to bound its stack scan (a trace can write slots above
// any frame the interpreter ever entered).
struct TraceResume_t {
    const BcIns_t* pPc;
    TValue_t* pBase;
};

LJX_NOINLINE TraceResume_t RunTrace(const jit::Trace_t* pTrace, TValue_t* pBase,
                                    C_Universe* pUni) {
    auto fnTrace = reinterpret_cast<jit::TraceEntry_f>(pTrace->pCode);
    const std::uint32_t uExit = fnTrace(pBase, pUni->ArenaBase());
    if (jit::TraceDebug()) {
        auto* pStats = const_cast<jit::Trace_t*>(pTrace);
        ++pStats->uEntries;
        if (pStats->vExitCounts.size() <= uExit) pStats->vExitCounts.resize(uExit + 1);
        ++pStats->vExitCounts[uExit];
        if (pStats->uEntries < 12)
            std::fprintf(stderr, "[trace] #%u entry %llu -> exit %u, resume %s @%p base%+d\n",
                         pTrace->uNumber,
                         static_cast<unsigned long long>(pStats->uEntries), uExit,
                         OpName(pTrace->vExits[uExit].pResumePc->Op()),
                         static_cast<const void*>(pTrace->vExits[uExit].pResumePc),
                         pTrace->vExits[uExit].nBaseOffset);
    }
    C_LuaThread* pThread = pUni->MainThread();
    TValue_t* pReach = pBase + pTrace->nTopSlot + 1;
    if (pReach > pThread->m_pHighWater) pThread->m_pHighWater = pReach;
    const jit::TraceExit_t& exit = pTrace->vExits[uExit];
    return TraceResume_t{exit.pResumePc, pBase + exit.nBaseOffset};
}

// Turns a loop op into its J-variant once a trace exists for it, so the entry
// test costs nothing on loops that were never compiled.
LJX_FORCEINLINE void PatchLoopOp(const BcIns_t* pIns, EBcOp eJitVariant) noexcept {
    auto* pMutable = const_cast<BcIns_t*>(pIns);
    pMutable->uRaw = (pMutable->uRaw & ~std::uint32_t{0xff}) |
                     static_cast<std::uint32_t>(eJitVariant);
}

LJX_H(ForL) {
    TValue_t* pSlots = pBase + uRa;
    const double flStep = pSlots[2].AsDouble();
    const double flIdx = pSlots[0].AsDouble() + flStep;
    const double flStop = pSlots[1].AsDouble();
    pSlots[0] = TValue_t::Number(flIdx);
    const bool bContinue = flStep >= 0 ? flIdx <= flStop : flIdx >= flStop;
    if (bContinue) {
        pSlots[3] = pSlots[0];
        const BcIns_t insSelf{pPc[-1].uRaw};
        pPc += insSelf.JumpTarget();
        // Three instructions on the hot path: the counter only borrows once
        // every kHotLoopThreshold iterations.
        if (pUni->HotCounts().DecrementLoop(pPc) && pUni->TraceJit() &&
            !pUni->m_uRecording) [[unlikely]] {
            pUni->HotCounts().Reset(pPc, C_HotCountTable::kArmedValue);
            const jit::Trace_t* pTrace = pUni->TraceJit()->OnLoopEdge(pPc, pBase, pKBase);
            if (pTrace) {
                PatchLoopOp(pPc - 1 - insSelf.JumpTarget(), EBcOp::JForL);
                const TraceResume_t res = RunTrace(pTrace, pBase, pUni);
                pBase = res.pBase;
                pPc = res.pPc;
                pKBase = KBaseOf(pUni, FrameProto(pUni, pBase));
            }
        }
    }
    LJX_NEXT();
}
LJX_H(JForI) { LJX_MUSTTAIL return OpForI(LJX_PASS_ARGS); }
LJX_H(IForL) { LJX_MUSTTAIL return OpForL(LJX_PASS_ARGS); }
LJX_H(JForL) {
    TValue_t* pSlots = pBase + uRa;
    const double flStep = pSlots[2].AsDouble();
    const double flIdx = pSlots[0].AsDouble() + flStep;
    const double flStop = pSlots[1].AsDouble();
    pSlots[0] = TValue_t::Number(flIdx);
    if (flStep >= 0 ? flIdx <= flStop : flIdx >= flStop) {
        pSlots[3] = pSlots[0];
        const BcIns_t insSelf{pPc[-1].uRaw};
        pPc += insSelf.JumpTarget();
        if (const jit::Trace_t* pTrace = pUni->TraceJit()->TraceAt(pPc)) [[likely]] {
            const TraceResume_t res = RunTrace(pTrace, pBase, pUni);
            pBase = res.pBase;
            pPc = res.pPc;
            pKBase = KBaseOf(pUni, FrameProto(pUni, pBase));
        }
    }
    LJX_NEXT();
}

LJX_H(IterC) {
    // R[A]=R[A-3], R[A+2]=R[A-2], R[A+3]=R[A-1]; then call R[A] with 2 args.
    TValue_t* pSlots = pBase + uRa;
    pSlots[0] = pSlots[-3];
    pSlots[2] = pSlots[-2];
    pSlots[3] = pSlots[-1];
    uRd = 2;
    LJX_MUSTTAIL return CallDispatch(LJX_PASS_ARGS);
}
LJX_H(IterN) { LJX_MUSTTAIL return OpIterC(LJX_PASS_ARGS); }
LJX_H(IterL) {
    TValue_t* pSlots = pBase + uRa;
    if (!pSlots[0].IsNil()) {
        pSlots[-1] = pSlots[0];  // update control variable
        const BcIns_t insSelf{pPc[-1].uRaw};
        pPc += insSelf.JumpTarget();
    }
    LJX_NEXT();
}
LJX_H(IIterL) { LJX_MUSTTAIL return OpIterL(LJX_PASS_ARGS); }
LJX_H(JIterL) { LJX_MUSTTAIL return OpIterL(LJX_PASS_ARGS); }
LJX_H(IsNext) { ErrorAtPc(pUni, pBase, pPc, "IsNext not emitted%s", ""); }

LJX_H(Loop) {
    if (pUni->HotCounts().DecrementLoop(pPc) && pUni->TraceJit() &&
        !pUni->m_uRecording) [[unlikely]] {
        pUni->HotCounts().Reset(pPc, C_HotCountTable::kArmedValue);
        const jit::Trace_t* pTrace = pUni->TraceJit()->OnLoopEdge(pPc - 1, pBase, pKBase);
        if (pTrace) {
            PatchLoopOp(pPc - 1, EBcOp::JLoop);
            const TraceResume_t res = RunTrace(pTrace, pBase, pUni);
            pBase = res.pBase;
            pPc = res.pPc;
            pKBase = KBaseOf(pUni, FrameProto(pUni, pBase));
        }
    }
    LJX_NEXT();
}
LJX_H(ILoop) { LJX_NEXT(); }
LJX_H(JLoop) {
    if (const jit::Trace_t* pTrace = pUni->TraceJit()->TraceAt(pPc - 1)) [[likely]] {
        const TraceResume_t res = RunTrace(pTrace, pBase, pUni);
        pBase = res.pBase;
        pPc = res.pPc;
        pKBase = KBaseOf(pUni, FrameProto(pUni, pBase));
    }
    LJX_NEXT();
}

LJX_H(Jmp) {
    pPc += static_cast<std::int32_t>(uRd) - static_cast<std::int32_t>(kJumpBias);
    LJX_NEXT();
}

LJX_H(VarG) { ErrorAtPc(pUni, pBase, pPc, "varargs not supported yet%s", ""); }
LJX_H(FuncV) { ErrorAtPc(pUni, pBase, pPc, "vararg functions not supported yet%s", ""); }
LJX_H(IFuncF) { LJX_MUSTTAIL return OpFuncF(LJX_PASS_ARGS); }
LJX_H(JFuncF) { LJX_MUSTTAIL return OpFuncF(LJX_PASS_ARGS); }
LJX_H(IFuncV) { LJX_MUSTTAIL return OpFuncV(LJX_PASS_ARGS); }
LJX_H(JFuncV) { LJX_MUSTTAIL return OpFuncV(LJX_PASS_ARGS); }

// Recording stub. Every dynamic entry points here while a trace is being
// recorded: it observes the instruction WITH ITS LIVE OPERANDS — which is what
// lets the recorder specialize on the types actually present — and then
// re-dispatches through the static table.
LJX_PRESERVE_NONE void RecordAndExecute(LJX_HANDLER_ARGS) {
    pUni->TraceJit()->RecordInstruction(pPc - 1, pBase, pKBase);
    BcHandler_f fnReal = pUni->Dispatch().Static(pPc[-1].uRaw & 0xff);
    LJX_MUSTTAIL return fnReal(LJX_PASS_ARGS);
}

}  // namespace

// ---------------------------------------------------------------------------
// Dispatch-table population & C++ entry points.
// ---------------------------------------------------------------------------

void C_DispatchTable::SetMode(EDispatchMode eMode) noexcept {
    const bool bRecord = HasMode(eMode, EDispatchMode::Recording);
    for (std::uint32_t uI = 0; uI < kEntries; ++uI)
        m_vDynamic[uI].store(bRecord ? &RecordAndExecute : m_vStatic[uI],
                             std::memory_order_release);
}

void C_Interpreter::SetRecordMode(C_Universe& uni, bool bOn) noexcept {
    uni.m_uRecording = bOn ? 1u : 0u;
    uni.Dispatch().SetMode(bOn ? EDispatchMode::Recording : EDispatchMode::Normal);
}

void C_Interpreter::InitDispatchTables(C_DispatchTable& dispatch) noexcept {
#define LJX_SET(name) \
    dispatch.SetStatic(static_cast<std::uint32_t>(EBcOp::name), &Op##name);
    LJX_BC_REGISTRY(LJX_SET)
#undef LJX_SET
    dispatch.SetMode(EDispatchMode::Normal);
}

std::int32_t C_Interpreter::Call(C_LuaThread* pThread, TValue_t* pFunc, std::int32_t nArgs,
                                 std::int32_t nResults) {
    // Frame: [func][link][args…]; link marks the C entry.
    C_Universe* pUniverse = pThread->m_pUniverse;
    // A nested interpreter run (a metamethod, a C function calling back) is
    // not part of the trace being recorded — and its bytecodes would otherwise
    // be appended to it with a bogus frame base.
    if (pUniverse->m_uRecording) pUniverse->TraceJit()->AbortForReentry();
    TValue_t* pBase = pFunc + 2;
    (void)pBase;
    pFunc[1] = TValue_t{FrameLink_t::FromDelta(0, EFrameType::C).uRaw};
    if (!pFunc[0].Is(EValueTag::Function))
        RaiseError(*pUniverse, "attempt to call a %s value", TypeName(pFunc[0]));
    auto* pFn = static_cast<C_GcFunction*>(pFunc[0].AsGcPointer());
    const auto* pCalleePc = reinterpret_cast<const BcIns_t*>(pFn->m_pPc);
    const BcIns_t insHeader{pCalleePc->uRaw};
    BcHandler_f fnHeader = pUniverse->Dispatch().Dynamic(insHeader.uRaw & 0xff);
    pUniverse->m_uEntryResults = 0;
    fnHeader(pBase, pCalleePc + 1, insHeader.A(), static_cast<std::uint64_t>(nArgs),
             pUniverse, nullptr);
    const auto nGot = static_cast<std::int32_t>(pUniverse->m_uEntryResults);
    // Adjust to the requested count (results start at pFunc).
    if (nResults >= 0)
        for (std::int32_t nI = nGot; nI < nResults; ++nI) pFunc[nI] = TValue_t::Nil();
    return nGot;
}

std::int32_t C_Interpreter::ProtectedCall(C_LuaThread* pThread, TValue_t* pFunc,
                                          std::int32_t nArgs, std::int32_t nResults,
                                          std::uint64_t) {
    C_Universe* pUni = pThread->m_pUniverse;
    ErrorFrame_t frame;
    frame.pPrev = pUni->m_pErrorTop;
    frame.pStackTop = pThread->m_pTop;
    pUni->m_pErrorTop = &frame;
    if (setjmp(frame.jb) != 0) {          // error landed here
        pUni->m_pErrorTop = frame.pPrev;
        pThread->m_pTop = frame.pStackTop;
        pFunc[0] = pUni->m_tvErrorValue;
        return -1;
    }
    const std::int32_t nGot = Call(pThread, pFunc, nArgs, nResults);
    pUni->m_pErrorTop = frame.pPrev;
    return nGot;
}

std::int32_t C_Interpreter::Resume(C_LuaThread*, std::int32_t) noexcept {
    return -1;  // coroutines land with the incremental GC phase
}

LJX_PRESERVE_NONE void C_Interpreter::ReenterFromExit(TValue_t*, const BcIns_t*, C_Universe*) {
    // JIT deopt re-entry: no traces exist yet.
}

}  // namespace ljx::vm
