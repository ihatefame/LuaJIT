// LJX — minimal standard library (v1: enough for real programs & benchmarks).
// C functions use the internal convention: args at base[0..n), results
// written back at base[0..], count returned. (The Lua 5.1 C-API compat shim
// wraps this later.)
#include <csetjmp>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <string>

#include "ljx/gc/GarbageCollector.hpp"
#include "ljx/vm/FastFunc.hpp"
#include "ljx/rt/Meta.hpp"
#include "ljx/rt/Pattern.hpp"
#include "ljx/rt/StringInterner.hpp"
#include "ljx/vm/Interpreter.hpp"

namespace ljx::vm {

bool TableNext(C_Universe& uni, C_GcTable* pTab, const TValue_t& tvKey, TValue_t& tvOutKey,
               TValue_t& tvOutVal);

namespace {

LJX_FORCEINLINE C_LuaThread* Th(lua_State* pState) noexcept {
    return reinterpret_cast<C_LuaThread*>(pState);
}
LJX_FORCEINLINE C_Universe* Uni(lua_State* pState) noexcept {
    return Th(pState)->m_pUniverse;
}
LJX_FORCEINLINE TValue_t* Args(lua_State* pState) noexcept { return Th(pState)->m_pBase; }
LJX_FORCEINLINE std::int32_t ArgCount(lua_State* pState) noexcept {
    return static_cast<std::int32_t>(Th(pState)->m_pTop - Th(pState)->m_pBase);
}

C_GcString* StrArg(lua_State* pState, std::int32_t nIdx, const char* sWho) {
    const TValue_t tvValue = Args(pState)[nIdx];
    if (!tvValue.Is(EValueTag::String))
        RaiseError(*Uni(pState), "bad argument #%d to '%s' (string expected)", nIdx + 1, sWho);
    return static_cast<C_GcString*>(tvValue.AsGcPointer());
}
double NumArg(lua_State* pState, std::int32_t nIdx, const char* sWho) {
    const TValue_t tvValue = Args(pState)[nIdx];
    if (!tvValue.IsDouble())
        RaiseError(*Uni(pState), "bad argument #%d to '%s' (number expected)", nIdx + 1, sWho);
    return tvValue.AsDouble();
}
C_GcTable* TabArg(lua_State* pState, std::int32_t nIdx, const char* sWho) {
    const TValue_t tvValue = Args(pState)[nIdx];
    if (!tvValue.Is(EValueTag::Table))
        RaiseError(*Uni(pState), "bad argument #%d to '%s' (table expected)", nIdx + 1, sWho);
    return static_cast<C_GcTable*>(tvValue.AsGcPointer());
}

void ToStringBuf(C_Universe& uni, const TValue_t& tvValue, std::string& sOut) {
    char vBuffer[48];
    if (tvValue.Is(EValueTag::String)) {
        auto* pStr = static_cast<C_GcString*>(tvValue.AsGcPointer());
        sOut.assign(pStr->Data(), pStr->Length());
    } else if (tvValue.IsDouble()) {
        const double flValue = tvValue.AsDouble();
        std::int32_t nInt;
        if (core::NumToInt32Check(flValue, nInt))
            std::snprintf(vBuffer, sizeof vBuffer, "%d", nInt);
        else
            std::snprintf(vBuffer, sizeof vBuffer, "%.14g", flValue);
        sOut = vBuffer;
    } else if (tvValue.IsNil()) {
        sOut = "nil";
    } else if (tvValue.Is(EValueTag::False)) {
        sOut = "false";
    } else if (tvValue.Is(EValueTag::True)) {
        sOut = "true";
    } else {
        const char* sType = tvValue.Is(EValueTag::Table)      ? "table"
                            : tvValue.Is(EValueTag::Function) ? "function"
                                                              : "value";
        std::snprintf(vBuffer, sizeof vBuffer, "%s: %p", sType, tvValue.AsGcPointer());
        sOut = vBuffer;
    }
    (void)uni;
}

// --- base library -----------------------------------------------------------

std::int32_t LibPrint(lua_State* pState) {
    const std::int32_t nArgs = ArgCount(pState);
    std::string sPart;
    for (std::int32_t nI = 0; nI < nArgs; ++nI) {
        if (nI) std::fputc('\t', stdout);
        ToStringBuf(*Uni(pState), Args(pState)[nI], sPart);
        std::fwrite(sPart.data(), 1, sPart.size(), stdout);
    }
    std::fputc('\n', stdout);
    return 0;
}

std::int32_t LibType(lua_State* pState) {
    const TValue_t tvValue = Args(pState)[0];
    const char* sName;
    if (tvValue.IsNil()) sName = "nil";
    else if (tvValue.IsNumber()) sName = "number";
    else if (tvValue.Is(EValueTag::False) || tvValue.Is(EValueTag::True)) sName = "boolean";
    else if (tvValue.Is(EValueTag::String)) sName = "string";
    else if (tvValue.Is(EValueTag::Table)) sName = "table";
    else if (tvValue.Is(EValueTag::Function)) sName = "function";
    else sName = "userdata";
    Args(pState)[0] =
        TValue_t::GcObject(EValueTag::String, Uni(pState)->Interner().Intern(sName));
    return 1;
}

std::int32_t LibToString(lua_State* pState) {
    std::string sOut;
    ToStringBuf(*Uni(pState), Args(pState)[0], sOut);
    Args(pState)[0] =
        TValue_t::GcObject(EValueTag::String, Uni(pState)->Interner().Intern(sOut));
    return 1;
}

std::int32_t LibToNumber(lua_State* pState) {
    const TValue_t tvValue = Args(pState)[0];
    if (tvValue.IsDouble()) return 1;
    if (tvValue.Is(EValueTag::String)) {
        auto* pStr = static_cast<C_GcString*>(tvValue.AsGcPointer());
        char* pEnd = nullptr;
        const double flValue = std::strtod(pStr->Data(), &pEnd);
        while (*pEnd == ' ' || *pEnd == '\t') ++pEnd;
        if (pEnd != pStr->Data() && *pEnd == '\0') {
            Args(pState)[0] = TValue_t::Number(flValue);
            return 1;
        }
    }
    Args(pState)[0] = TValue_t::Nil();
    return 1;
}

std::int32_t LibNext(lua_State* pState) {
    C_GcTable* pTab = TabArg(pState, 0, "next");
    const TValue_t tvKey = ArgCount(pState) > 1 ? Args(pState)[1] : TValue_t::Nil();
    TValue_t tvOutKey, tvOutVal;
    if (TableNext(*Uni(pState), pTab, tvKey, tvOutKey, tvOutVal)) {
        Args(pState)[0] = tvOutKey;
        Args(pState)[1] = tvOutVal;
        return 2;
    }
    Args(pState)[0] = TValue_t::Nil();
    return 1;
}

std::int32_t LibPairs(lua_State* pState) {
    C_GcTable* pTab = TabArg(pState, 0, "pairs");
    const TValue_t* pNextFn =
        Uni(pState)->Registry()->GetStr(*Uni(pState), Uni(pState)->Interner().Intern("next"));
    Args(pState)[0] = pNextFn ? *pNextFn : TValue_t::Nil();
    Args(pState)[1] = TValue_t::GcObject(EValueTag::Table, pTab);
    Args(pState)[2] = TValue_t::Nil();
    return 3;
}

std::int32_t LibIPairsIter(lua_State* pState) {
    C_GcTable* pTab = TabArg(pState, 0, "ipairs iterator");
    const double flIndex = NumArg(pState, 1, "ipairs iterator") + 1;
    const TValue_t* pSlot =
        pTab->GetInt(*Uni(pState), static_cast<std::uint32_t>(flIndex));
    if (!pSlot || pSlot->IsNil()) {
        Args(pState)[0] = TValue_t::Nil();
        return 1;
    }
    Args(pState)[0] = TValue_t::Number(flIndex);
    Args(pState)[1] = *pSlot;
    return 2;
}

std::int32_t LibIPairs(lua_State* pState) {
    C_GcTable* pTab = TabArg(pState, 0, "ipairs");
    const TValue_t* pIterFn = Uni(pState)->Registry()->GetStr(
        *Uni(pState), Uni(pState)->Interner().Intern("ipairs_iter"));
    Args(pState)[0] = pIterFn ? *pIterFn : TValue_t::Nil();
    Args(pState)[1] = TValue_t::GcObject(EValueTag::Table, pTab);
    Args(pState)[2] = TValue_t::Number(0.0);
    return 3;
}

std::int32_t LibSetMetatable(lua_State* pState) {
    C_GcTable* pTab = TabArg(pState, 0, "setmetatable");
    const TValue_t tvMeta = Args(pState)[1];
    if (tvMeta.IsNil()) {
        pTab->m_rMetatable = core::GcRef_t{};
    } else if (tvMeta.Is(EValueTag::Table)) {
        pTab->m_rMetatable = Uni(pState)->MakeRef(tvMeta.AsGcPointer());
        rt::C_MetaResolver::InvalidateNegativeCache(
            static_cast<C_GcTable*>(tvMeta.AsGcPointer()));
    } else {
        RaiseError(*Uni(pState), "bad argument #2 to 'setmetatable'");
    }
    return 1;  // the table itself
}

std::int32_t LibGetMetatable(lua_State* pState) {
    C_GcTable* pTab = TabArg(pState, 0, "getmetatable");
    auto* pMt = Uni(pState)->Deref<C_GcTable>(pTab->m_rMetatable);
    Args(pState)[0] =
        pMt ? TValue_t::GcObject(EValueTag::Table, pMt) : TValue_t::Nil();
    return 1;
}

std::int32_t LibSelect(lua_State* pState) {
    const std::int32_t nArgs = ArgCount(pState);
    if (nArgs < 1) RaiseError(*Uni(pState), "bad argument #1 to 'select'");
    const TValue_t tvSel = Args(pState)[0];
    if (tvSel.Is(EValueTag::String)) {
        auto* pStr = static_cast<C_GcString*>(tvSel.AsGcPointer());
        if (pStr->m_uLength == 1 && pStr->Data()[0] == '#') {
            Args(pState)[0] = TValue_t::Number(static_cast<double>(nArgs - 1));
            return 1;
        }
        RaiseError(*Uni(pState), "bad argument #1 to 'select' (number expected)");
    }
    const double flIdx = NumArg(pState, 0, "select");
    std::int32_t nIdx = static_cast<std::int32_t>(flIdx);
    if (nIdx < 0) nIdx += nArgs;   // select(-k, ...): the k-th from the end
    if (nIdx < 1)
        RaiseError(*Uni(pState), "bad argument #1 to 'select' (index out of range)");
    if (nIdx >= nArgs) return 0;
    const std::int32_t nOut = nArgs - nIdx;
    for (std::int32_t nI = 0; nI < nOut; ++nI)
        Args(pState)[nI] = Args(pState)[nIdx + nI];
    return nOut;
}

std::int32_t LibUnpack(lua_State* pState) {
    C_GcTable* pTab = TabArg(pState, 0, "unpack");
    C_Universe& uni = *Uni(pState);
    const std::int32_t nFrom =
        ArgCount(pState) > 1 ? static_cast<std::int32_t>(NumArg(pState, 1, "unpack")) : 1;
    const std::int32_t nTo = ArgCount(pState) > 2
                                 ? static_cast<std::int32_t>(NumArg(pState, 2, "unpack"))
                                 : static_cast<std::int32_t>(pTab->Length(uni));
    if (nFrom > nTo) return 0;
    const std::int32_t nCount = nTo - nFrom + 1;
    C_LuaThread* pThread = Th(pState);
    if (pThread->m_pBase + nCount + vm::kStackExtraSlots > pThread->m_pMaxStack)
        RaiseError(uni, "too many results to unpack");
    if (pThread->m_pBase + nCount > pThread->m_pHighWater)
        pThread->m_pHighWater = pThread->m_pBase + nCount;
    for (std::int32_t nI = 0; nI < nCount; ++nI) {
        const TValue_t* pSlot =
            pTab->Get(uni, TValue_t::Number(static_cast<double>(nFrom + nI)));
        Args(pState)[nI] = pSlot ? *pSlot : TValue_t::Nil();
    }
    return nCount;
}

std::int32_t LibRawGet(lua_State* pState) {
    C_GcTable* pTab = TabArg(pState, 0, "rawget");
    const TValue_t* pSlot = pTab->Get(*Uni(pState), Args(pState)[1]);
    Args(pState)[0] = pSlot ? *pSlot : TValue_t::Nil();
    return 1;
}

std::int32_t LibRawSet(lua_State* pState) {
    C_GcTable* pTab = TabArg(pState, 0, "rawset");
    *pTab->Set(*Uni(pState), Args(pState)[1]) = Args(pState)[2];
    return 1;
}

std::int32_t LibError(lua_State* pState) {
    C_Universe& uni = *Uni(pState);
    const TValue_t tvErr = ArgCount(pState) > 0 ? Args(pState)[0] : TValue_t::Nil();
    const bool bAddPos = !(ArgCount(pState) > 1 && Args(pState)[1].IsDouble() &&
                           Args(pState)[1].AsDouble() == 0.0);
    // error(message [,level]): a string message at level != 0 is prefixed
    // with the caller's position, as in the reference implementation.
    if (tvErr.Is(EValueTag::String) && bAddPos) {
        TValue_t* pBase = Args(pState);
        const FrameLink_t link{pBase[-1].uRaw};
        if (link.IsLua()) {
            const BcIns_t* pRetPc = link.ReturnPc();
            const BcIns_t insCall{pRetPc[-1].uRaw};
            const TValue_t* pCallerBase = pBase - 2 - insCall.A();
            const auto* pCaller =
                static_cast<const C_GcFunction*>(pCallerBase[-2].AsGcPointer());
            const C_GcProto* pProto = C_GcProto::FromBytecode(pCaller->m_pPc);
            const C_GcString* pChunk = uni.Deref<C_GcString>(pProto->m_rChunkName);
            core::BcLine_t uLine = 0;
            if (!pProto->m_rLineInfo.IsNull()) {
                const auto* pLines = static_cast<const core::BcLine_t*>(
                    core::RefToPtr(uni.ArenaBase(), pProto->m_rLineInfo));
                const auto uIdx = static_cast<std::size_t>(
                    pRetPc - 1 - reinterpret_cast<const BcIns_t*>(pProto->Bytecode()));
                if (uIdx < pProto->m_uBcCount) uLine = pLines[uIdx];
            }
            auto* pMsg = static_cast<C_GcString*>(tvErr.AsGcPointer());
            char vBuffer[512];
            std::snprintf(vBuffer, sizeof vBuffer, "%s:%u: %.*s",
                          pChunk ? pChunk->Data() : "?", uLine,
                          static_cast<int>(pMsg->Length()), pMsg->Data());
            RaiseErrorValue(uni, TValue_t::GcObject(EValueTag::String,
                                                    uni.Interner().Intern(vBuffer)));
        }
    }
    RaiseErrorValue(uni, tvErr);
}

std::int32_t LibAssert(lua_State* pState) {
    if (!Args(pState)[0].IsTruthy()) {
        if (ArgCount(pState) > 1) RaiseErrorValue(*Uni(pState), Args(pState)[1]);
        RaiseError(*Uni(pState), "assertion failed!");
    }
    return ArgCount(pState);
}

std::int32_t LibPCall(lua_State* pState) {
    const std::int32_t nArgs = ArgCount(pState) - 1;
    if (nArgs < 0) RaiseError(*Uni(pState), "bad argument #1 to 'pcall'");
    C_Universe* pUni = Uni(pState);
    C_LuaThread* pThread = Th(pState);
    TValue_t* pArgs0 = Args(pState);
    // Build a fresh frame above the current top: [f][link][args…].
    TValue_t* pFrame = pThread->m_pTop + 8;   // headroom above this C frame
    pFrame[0] = pArgs0[0];
    pFrame[1] = TValue_t::Nil();
    for (std::int32_t nI = 0; nI < nArgs; ++nI) pFrame[2 + nI] = pArgs0[1 + nI];
    TValue_t* pSavedBase = pThread->m_pBase;
    TValue_t* pSavedTop = pThread->m_pTop;

    ErrorFrame_t frame;
    frame.pPrev = pUni->m_pErrorTop;
    frame.pStackTop = pSavedTop;
    pUni->m_pErrorTop = &frame;
    if (setjmp(frame.jb) != 0) {
        pUni->m_pErrorTop = frame.pPrev;
        pThread->m_pBase = pSavedBase;
        pThread->m_pTop = pSavedTop;
        pArgs0[0] = TValue_t::Boolean(false);
        pArgs0[1] = pUni->m_tvErrorValue;
        return 2;
    }
    const std::int32_t nResults = C_Interpreter::Call(pThread, pFrame, nArgs, -1);
    pUni->m_pErrorTop = frame.pPrev;
    pThread->m_pBase = pSavedBase;
    pThread->m_pTop = pSavedTop;
    for (std::int32_t nI = nResults; nI-- > 0;) pArgs0[1 + nI] = pFrame[nI];
    pArgs0[0] = TValue_t::Boolean(true);
    return nResults + 1;
}

// --- math -------------------------------------------------------------------

#define LJX_MATH1(name, expr)                                       \
    std::int32_t LibMath##name(lua_State* pState) {                 \
        const double flX = NumArg(pState, 0, "math");               \
        Args(pState)[0] = TValue_t::Number(expr);                   \
        return 1;                                                   \
    }
LJX_MATH1(Floor, std::floor(flX))
LJX_MATH1(Ceil, std::ceil(flX))
LJX_MATH1(Sqrt, std::sqrt(flX))
LJX_MATH1(Abs, std::fabs(flX))
LJX_MATH1(Sin, std::sin(flX))
LJX_MATH1(Cos, std::cos(flX))
LJX_MATH1(Exp, std::exp(flX))
LJX_MATH1(Log, std::log(flX))
#undef LJX_MATH1

std::int32_t LibMathMax(lua_State* pState) {
    double flBest = NumArg(pState, 0, "max");
    for (std::int32_t nI = 1; nI < ArgCount(pState); ++nI)
        flBest = std::fmax(flBest, NumArg(pState, nI, "max"));
    Args(pState)[0] = TValue_t::Number(flBest);
    return 1;
}
std::int32_t LibMathMin(lua_State* pState) {
    double flBest = NumArg(pState, 0, "min");
    for (std::int32_t nI = 1; nI < ArgCount(pState); ++nI)
        flBest = std::fmin(flBest, NumArg(pState, nI, "min"));
    Args(pState)[0] = TValue_t::Number(flBest);
    return 1;
}
std::int32_t LibMathRandom(lua_State* pState) {
    const double flRand = Uni(pState)->Prng().NextDouble();
    if (ArgCount(pState) == 0) {
        Args(pState)[0] = TValue_t::Number(flRand);
    } else if (ArgCount(pState) == 1) {
        const double flHi = NumArg(pState, 0, "random");
        Args(pState)[0] = TValue_t::Number(std::floor(flRand * flHi) + 1);
    } else {
        const double flLo = NumArg(pState, 0, "random");
        const double flHi = NumArg(pState, 1, "random");
        Args(pState)[0] = TValue_t::Number(flLo + std::floor(flRand * (flHi - flLo + 1)));
    }
    return 1;
}

// --- string -----------------------------------------------------------------

std::int32_t LibStringLen(lua_State* pState) {
    Args(pState)[0] =
        TValue_t::Number(static_cast<double>(StrArg(pState, 0, "len")->Length()));
    return 1;
}

std::int32_t LibStringSub(lua_State* pState) {
    C_GcString* pStr = StrArg(pState, 0, "sub");
    const auto nLen = static_cast<std::int64_t>(pStr->Length());
    std::int64_t nFrom = static_cast<std::int64_t>(NumArg(pState, 1, "sub"));
    std::int64_t nTo =
        ArgCount(pState) > 2 ? static_cast<std::int64_t>(NumArg(pState, 2, "sub")) : -1;
    if (nFrom < 0) nFrom = nLen + nFrom + 1 > 1 ? nLen + nFrom + 1 : 1;
    else if (nFrom == 0) nFrom = 1;
    if (nTo < 0) nTo = nLen + nTo + 1;
    else if (nTo > nLen) nTo = nLen;
    std::string_view svResult;
    if (nFrom <= nTo)
        svResult = std::string_view(pStr->Data() + nFrom - 1,
                                    static_cast<std::size_t>(nTo - nFrom + 1));
    Args(pState)[0] =
        TValue_t::GcObject(EValueTag::String, Uni(pState)->Interner().Intern(svResult));
    return 1;
}

std::int32_t LibStringRep(lua_State* pState) {
    C_GcString* pStr = StrArg(pState, 0, "rep");
    const auto nCount = static_cast<std::int64_t>(NumArg(pState, 1, "rep"));
    std::string sResult;
    for (std::int64_t nI = 0; nI < nCount; ++nI) sResult.append(pStr->Data(), pStr->Length());
    Args(pState)[0] =
        TValue_t::GcObject(EValueTag::String, Uni(pState)->Interner().Intern(sResult));
    return 1;
}

// --- string patterns --------------------------------------------------------

// One capture as a Lua value; uIdx == level with level 0 means "whole match".
TValue_t CaptureValue(lua_State* pState, rt::C_PatternMatcher& matcher,
                      std::uint32_t uIdx, const char* pSrcInit, const char* pMatchS,
                      const char* pMatchE) {
    if (uIdx >= matcher.m_uLevel) {
        return TValue_t::GcObject(
            EValueTag::String,
            Uni(pState)->Interner().Intern(
                std::string_view(pMatchS, static_cast<std::size_t>(pMatchE - pMatchS))));
    }
    const rt::MatchCapture_t& cap = matcher.m_vCapture[uIdx];
    if (cap.nLen == rt::kCapUnfinished)
        RaiseError(*Uni(pState), "unfinished capture");
    if (cap.nLen == rt::kCapPosition)
        return TValue_t::Number(static_cast<double>(cap.pInit - pSrcInit + 1));
    return TValue_t::GcObject(
        EValueTag::String,
        Uni(pState)->Interner().Intern(
            std::string_view(cap.pInit, static_cast<std::size_t>(cap.nLen))));
}

// Write the captures (or the whole match) to pOut; returns the value count.
std::int32_t PushCaptures(lua_State* pState, rt::C_PatternMatcher& matcher,
                          const char* pSrcInit, const char* pMatchS, const char* pMatchE,
                          TValue_t* pOut) {
    const std::uint32_t uCount = matcher.m_uLevel ? matcher.m_uLevel : 1;
    for (std::uint32_t uI = 0; uI < uCount; ++uI)
        pOut[uI] = CaptureValue(pState, matcher, matcher.m_uLevel ? uI : 1,
                                pSrcInit, pMatchS, pMatchE);
    return static_cast<std::int32_t>(uCount);
}

// Clamped 1-based init argument -> byte offset.
std::size_t PatternInit(lua_State* pState, std::int32_t nArg, std::size_t uLen) {
    if (ArgCount(pState) <= nArg || Args(pState)[nArg].IsNil()) return 0;
    auto nInit = static_cast<std::int64_t>(NumArg(pState, nArg, "find"));
    if (nInit > 0) return static_cast<std::size_t>(nInit - 1) > uLen
                              ? uLen
                              : static_cast<std::size_t>(nInit - 1);
    if (nInit == 0) return 0;
    const auto nFromEnd = static_cast<std::int64_t>(uLen) + nInit;
    return nFromEnd < 0 ? 0 : static_cast<std::size_t>(nFromEnd);
}

std::int32_t StringFindCommon(lua_State* pState, bool bFind) {
    C_GcString* pSrc = StrArg(pState, 0, "find");
    C_GcString* pPat = StrArg(pState, 1, "find");
    const char* pS = pSrc->Data();
    const std::size_t uSLen = pSrc->Length();
    const char* pP = pPat->Data();
    const std::size_t uPLen = pPat->Length();
    const std::size_t uInit = PatternInit(pState, 2, uSLen);
    C_Universe& uni = *Uni(pState);

    if (bFind && ArgCount(pState) > 3 && Args(pState)[3].IsTruthy()) {
        // Plain find: raw substring search, no pattern meaning at all.
        if (uPLen == 0) {
            Args(pState)[0] = TValue_t::Number(static_cast<double>(uInit + 1));
            Args(pState)[1] = TValue_t::Number(static_cast<double>(uInit));
            return 2;
        }
        for (std::size_t uI = uInit; uI + uPLen <= uSLen; ++uI) {
            if (std::memcmp(pS + uI, pP, uPLen) == 0) {
                Args(pState)[0] = TValue_t::Number(static_cast<double>(uI + 1));
                Args(pState)[1] = TValue_t::Number(static_cast<double>(uI + uPLen));
                return 2;
            }
        }
        Args(pState)[0] = TValue_t::Nil();
        return 1;
    }

    const bool bAnchor = uPLen > 0 && *pP == '^';
    const char* pPatStart = pP + (bAnchor ? 1 : 0);
    rt::C_PatternMatcher matcher(uni, pS, pS + uSLen, pPatStart, pP + uPLen);
    const char* pTry = pS + uInit;
    do {
        matcher.m_uLevel = 0;
        if (const char* pEnd = matcher.Match(pTry, pPatStart)) {
            if (bFind) {
                Args(pState)[0] = TValue_t::Number(static_cast<double>(pTry - pS + 1));
                Args(pState)[1] = TValue_t::Number(static_cast<double>(pEnd - pS));
                return 2 + (matcher.m_uLevel
                                ? PushCaptures(pState, matcher, pS, pTry, pEnd,
                                               Args(pState) + 2)
                                : 0);
            }
            return PushCaptures(pState, matcher, pS, pTry, pEnd, Args(pState));
        }
    } while (pTry++ < pS + uSLen && !bAnchor);
    Args(pState)[0] = TValue_t::Nil();
    return 1;
}

std::int32_t LibStringFind(lua_State* pState) { return StringFindCommon(pState, true); }
std::int32_t LibStringMatch(lua_State* pState) { return StringFindCommon(pState, false); }

// gmatch iterator: a C closure whose upvalues hold (source, pattern, pos).
std::int32_t LibGMatchAux(lua_State* pState) {
    auto* pSelf = static_cast<C_GcFunction*>(Args(pState)[-2].AsGcPointer());
    TValue_t* pUp = pSelf->CUpvalues();
    auto* pSrc = static_cast<C_GcString*>(pUp[0].AsGcPointer());
    auto* pPat = static_cast<C_GcString*>(pUp[1].AsGcPointer());
    const char* pS = pSrc->Data();
    const std::size_t uSLen = pSrc->Length();
    const char* pP = pPat->Data();
    C_Universe& uni = *Uni(pState);
    rt::C_PatternMatcher matcher(uni, pS, pS + uSLen, pP, pP + pPat->Length());
    for (auto uPos = static_cast<std::size_t>(pUp[2].AsDouble()); uPos <= uSLen; ++uPos) {
        matcher.m_uLevel = 0;
        if (const char* pEnd = matcher.Match(pS + uPos, pP)) {
            // Next scan starts past this match; an empty match still advances.
            const auto uNext = static_cast<std::size_t>(pEnd - pS);
            pUp[2] = TValue_t::Number(
                static_cast<double>(uNext > uPos ? uNext : uPos + 1));
            return PushCaptures(pState, matcher, pS, pS + uPos, pEnd, Args(pState));
        }
    }
    Args(pState)[0] = TValue_t::Nil();
    return 1;
}

std::int32_t LibStringGMatch(lua_State* pState) {
    C_GcString* pSrc = StrArg(pState, 0, "gmatch");
    C_GcString* pPat = StrArg(pState, 1, "gmatch");
    C_Universe& uni = *Uni(pState);
    auto* pFn = static_cast<C_GcFunction*>(
        uni.Gc().AllocObject(EGcObjectType::Function, C_GcFunction::CAllocSize(3)));
    pFn->m_Header.uExtra1 = static_cast<std::uint8_t>(EFastFunc::C);
    pFn->m_Header.uExtra2 = 3;
    pFn->m_rEnv = uni.MakeRef(uni.Globals());
    pFn->m_pPc = &uni.m_insCFuncHeader.uRaw;
    pFn->CFunc() = &LibGMatchAux;
    pFn->CUpvalues()[0] = TValue_t::GcObject(EValueTag::String, pSrc);
    pFn->CUpvalues()[1] = TValue_t::GcObject(EValueTag::String, pPat);
    pFn->CUpvalues()[2] = TValue_t::Number(0);
    Args(pState)[0] = TValue_t::GcObject(EValueTag::Function, pFn);
    return 1;
}

std::int32_t LibStringGSub(lua_State* pState) {
    C_GcString* pSrc = StrArg(pState, 0, "gsub");
    C_GcString* pPat = StrArg(pState, 1, "gsub");
    const TValue_t tvRepl = Args(pState)[2];
    const bool bHasMax = ArgCount(pState) > 3 && !Args(pState)[3].IsNil();
    const auto nMax = bHasMax ? static_cast<std::int64_t>(NumArg(pState, 3, "gsub"))
                              : static_cast<std::int64_t>(-1);
    C_Universe& uni = *Uni(pState);
    C_LuaThread* pThread = Th(pState);

    const char* pS = pSrc->Data();
    const std::size_t uSLen = pSrc->Length();
    const char* pP = pPat->Data();
    const std::size_t uPLen = pPat->Length();
    const bool bAnchor = uPLen > 0 && *pP == '^';
    const char* pPatStart = pP + (bAnchor ? 1 : 0);

    rt::C_PatternMatcher matcher(uni, pS, pS + uSLen, pPatStart, pP + uPLen);
    std::string sOut;
    const char* pCur = pS;
    std::int64_t nCount = 0;
    while (nMax < 0 || nCount < nMax) {
        matcher.m_uLevel = 0;
        const char* pEnd = matcher.Match(pCur, pPatStart);
        if (pEnd) {
            ++nCount;
            // Produce the replacement for pCur..pEnd.
            if (tvRepl.Is(EValueTag::String)) {
                auto* pRepl = static_cast<C_GcString*>(tvRepl.AsGcPointer());
                const char* pR = pRepl->Data();
                for (std::size_t uI = 0; uI < pRepl->Length(); ++uI) {
                    if (pR[uI] != '%' || uI + 1 >= pRepl->Length()) {
                        sOut.push_back(pR[uI]);
                        continue;
                    }
                    const char cNext = pR[++uI];
                    if (cNext == '%') {
                        sOut.push_back('%');
                    } else if (cNext == '0') {
                        sOut.append(pCur, static_cast<std::size_t>(pEnd - pCur));
                    } else if (cNext >= '1' && cNext <= '9') {
                        const TValue_t tvCap = CaptureValue(
                            pState, matcher,
                            matcher.m_uLevel ? static_cast<std::uint32_t>(cNext - '1')
                                             : 1u,
                            pS, pCur, pEnd);
                        std::string sPart;
                        ToStringBuf(uni, tvCap, sPart);
                        sOut += sPart;
                    } else {
                        RaiseError(uni, "invalid use of '%%' in replacement string");
                    }
                }
            } else {
                TValue_t tvValue;
                if (tvRepl.Is(EValueTag::Table)) {
                    const TValue_t tvKey =
                        CaptureValue(pState, matcher, matcher.m_uLevel ? 0u : 1u, pS,
                                     pCur, pEnd);
                    auto* pTab = static_cast<C_GcTable*>(tvRepl.AsGcPointer());
                    const TValue_t* pSlot = pTab->Get(uni, tvKey);
                    tvValue = pSlot ? *pSlot : TValue_t::Nil();
                } else if (tvRepl.Is(EValueTag::Function)) {
                    // Call repl(captures...) in a frame above the current top.
                    TValue_t* pFrame = pThread->m_pTop + 8;
                    pFrame[0] = tvRepl;
                    pFrame[1] = TValue_t::Nil();
                    const std::int32_t nArgs =
                        PushCaptures(pState, matcher, pS, pCur, pEnd, pFrame + 2);
                    TValue_t* pSavedBase = pThread->m_pBase;
                    TValue_t* pSavedTop = pThread->m_pTop;
                    C_Interpreter::Call(pThread, pFrame, nArgs, 1);
                    pThread->m_pBase = pSavedBase;
                    pThread->m_pTop = pSavedTop;
                    tvValue = pFrame[0];
                } else {
                    RaiseError(uni, "bad argument #3 to 'gsub' (string/function/table)");
                }
                if (tvValue.IsNil() || tvValue.uRaw == TValue_t::Boolean(false).uRaw) {
                    sOut.append(pCur, static_cast<std::size_t>(pEnd - pCur));
                } else if (tvValue.Is(EValueTag::String) || tvValue.IsDouble()) {
                    std::string sPart;
                    ToStringBuf(uni, tvValue, sPart);
                    sOut += sPart;
                } else {
                    RaiseError(uni, "invalid replacement value in 'gsub'");
                }
            }
        }
        if (pEnd && pEnd > pCur) {
            pCur = pEnd;
        } else {
            // No match (or an empty one): copy one byte and move on.
            if (pCur >= pS + uSLen) break;
            sOut.push_back(*pCur++);
        }
        if (bAnchor) break;
    }
    sOut.append(pCur, static_cast<std::size_t>(pS + uSLen - pCur));
    Args(pState)[0] = TValue_t::GcObject(EValueTag::String, uni.Interner().Intern(sOut));
    Args(pState)[1] = TValue_t::Number(static_cast<double>(nCount));
    return 2;
}

std::int32_t LibStringFormat(lua_State* pState) {
    C_GcString* pFmt = StrArg(pState, 0, "format");
    C_Universe& uni = *Uni(pState);
    const char* pC = pFmt->Data();
    const char* pEnd = pC + pFmt->Length();
    std::string sOut;
    std::int32_t nArg = 1;
    char vSpec[32];   // "%" + flags/width/precision + conversion
    char vItem[256];
    while (pC < pEnd) {
        if (*pC != '%') {
            sOut.push_back(*pC++);
            continue;
        }
        ++pC;
        if (pC < pEnd && *pC == '%') {
            sOut.push_back('%');
            ++pC;
            continue;
        }
        // Collect flags, width and precision (bounded, snprintf does the rest).
        std::size_t uSpec = 0;
        vSpec[uSpec++] = '%';
        while (pC < pEnd && std::strchr("-+ #0", *pC) && uSpec < 6) vSpec[uSpec++] = *pC++;
        while (pC < pEnd && *pC >= '0' && *pC <= '9' && uSpec < 9) vSpec[uSpec++] = *pC++;
        if (pC < pEnd && *pC == '.') {
            vSpec[uSpec++] = *pC++;
            while (pC < pEnd && *pC >= '0' && *pC <= '9' && uSpec < 13) vSpec[uSpec++] = *pC++;
        }
        if (pC >= pEnd) RaiseError(uni, "invalid format string to 'format'");
        const char cConv = *pC++;
        switch (cConv) {
            case 'd': case 'i': {
                const auto nValue =
                    static_cast<long long>(NumArg(pState, nArg++, "format"));
                vSpec[uSpec] = '\0';
                std::snprintf(vItem, sizeof vItem,
                              (std::string(vSpec) + "lld").c_str(), nValue);
                sOut += vItem;
                break;
            }
            case 'u': case 'x': case 'X': case 'o': {
                const auto uValue = static_cast<unsigned long long>(
                    static_cast<long long>(NumArg(pState, nArg++, "format")));
                vSpec[uSpec] = '\0';
                std::snprintf(vItem, sizeof vItem,
                              (std::string(vSpec) + "ll" + cConv).c_str(), uValue);
                sOut += vItem;
                break;
            }
            case 'c': {
                sOut.push_back(static_cast<char>(
                    static_cast<int>(NumArg(pState, nArg++, "format"))));
                break;
            }
            case 'f': case 'F': case 'e': case 'E': case 'g': case 'G': {
                vSpec[uSpec++] = cConv;
                vSpec[uSpec] = '\0';
                std::snprintf(vItem, sizeof vItem, vSpec,
                              NumArg(pState, nArg++, "format"));
                sOut += vItem;
                break;
            }
            case 's': {
                std::string sPart;
                ToStringBuf(uni, Args(pState)[nArg++], sPart);
                if (uSpec == 1) {
                    sOut += sPart;   // plain %s: no copy through the buffer
                } else {
                    vSpec[uSpec++] = 's';
                    vSpec[uSpec] = '\0';
                    std::snprintf(vItem, sizeof vItem, vSpec, sPart.c_str());
                    sOut += vItem;
                }
                break;
            }
            case 'q': {
                C_GcString* pStr = StrArg(pState, nArg++, "format");
                sOut.push_back('"');
                for (std::uint32_t uI = 0; uI < pStr->Length(); ++uI) {
                    const char c = pStr->Data()[uI];
                    switch (c) {
                        case '"': sOut += "\\\""; break;
                        case '\\': sOut += "\\\\"; break;
                        case '\n': sOut += "\\\n"; break;
                        case '\r': sOut += "\\r"; break;
                        case '\0': sOut += "\\0"; break;
                        default: sOut.push_back(c); break;
                    }
                }
                sOut.push_back('"');
                break;
            }
            default:
                RaiseError(uni, "invalid option '%%%c' to 'format'", cConv);
        }
    }
    Args(pState)[0] = TValue_t::GcObject(EValueTag::String, uni.Interner().Intern(sOut));
    return 1;
}

std::int32_t LibStringByte(lua_State* pState) {
    C_GcString* pStr = StrArg(pState, 0, "byte");
    const std::int64_t nIdx =
        ArgCount(pState) > 1 ? static_cast<std::int64_t>(NumArg(pState, 1, "byte")) : 1;
    if (nIdx < 1 || nIdx > pStr->Length()) {
        Args(pState)[0] = TValue_t::Nil();
        return 1;
    }
    Args(pState)[0] = TValue_t::Number(
        static_cast<double>(static_cast<unsigned char>(pStr->Data()[nIdx - 1])));
    return 1;
}

std::int32_t LibStringChar(lua_State* pState) {
    std::string sResult;
    for (std::int32_t nI = 0; nI < ArgCount(pState); ++nI)
        sResult.push_back(static_cast<char>(NumArg(pState, nI, "char")));
    Args(pState)[0] =
        TValue_t::GcObject(EValueTag::String, Uni(pState)->Interner().Intern(sResult));
    return 1;
}

// --- table ------------------------------------------------------------------

std::int32_t LibTableConcat(lua_State* pState) {
    C_GcTable* pTab = TabArg(pState, 0, "concat");
    C_Universe& uni = *Uni(pState);
    const char* sSep = "";
    std::uint32_t uSepLen = 0;
    if (ArgCount(pState) >= 2) {
        const TValue_t tvSep = Args(pState)[1];
        if (!tvSep.Is(EValueTag::String))
            RaiseError(uni, "bad argument #2 to 'concat' (string expected)");
        auto* pSep = static_cast<C_GcString*>(tvSep.AsGcPointer());
        sSep = pSep->Data();
        uSepLen = pSep->Length();
    }
    const auto uFirst = ArgCount(pState) >= 3
                            ? static_cast<std::uint32_t>(NumArg(pState, 2, "concat"))
                            : 1u;
    const auto uLast = ArgCount(pState) >= 4
                           ? static_cast<std::uint32_t>(NumArg(pState, 3, "concat"))
                           : pTab->Length(uni);
    std::string sOut;
    for (std::uint32_t uI = uFirst; uI <= uLast; ++uI) {
        const TValue_t* pSlot = pTab->Get(uni, TValue_t::Number(uI));
        if (!pSlot || pSlot->IsNil() ||
            (!pSlot->Is(EValueTag::String) && !pSlot->IsNumber()))
            RaiseError(uni, "invalid value (at index %d) in table for 'concat'", uI);
        std::string sPiece;
        ToStringBuf(uni, *pSlot, sPiece);
        sOut += sPiece;
        if (uI != uLast) sOut.append(sSep, uSepLen);
    }
    Args(pState)[0] = TValue_t::GcObject(EValueTag::String, uni.Interner().Intern(sOut));
    return 1;
}

std::int32_t LibTableInsert(lua_State* pState) {
    C_GcTable* pTab = TabArg(pState, 0, "insert");
    const std::uint32_t uLen = pTab->Length(*Uni(pState));
    if (ArgCount(pState) == 2) {
        *pTab->Set(*Uni(pState), TValue_t::Number(uLen + 1)) = Args(pState)[1];
    } else {
        const auto uPos = static_cast<std::uint32_t>(NumArg(pState, 1, "insert"));
        for (std::uint32_t uI = uLen + 1; uI > uPos; --uI)
            *pTab->Set(*Uni(pState), TValue_t::Number(uI)) =
                *pTab->Get(*Uni(pState), TValue_t::Number(uI - 1));
        *pTab->Set(*Uni(pState), TValue_t::Number(uPos)) = Args(pState)[2];
    }
    return 0;
}

std::int32_t LibTableRemove(lua_State* pState) {
    C_GcTable* pTab = TabArg(pState, 0, "remove");
    const std::uint32_t uLen = pTab->Length(*Uni(pState));
    if (uLen == 0) {
        Args(pState)[0] = TValue_t::Nil();
        return 1;
    }
    const std::uint32_t uPos =
        ArgCount(pState) > 1 ? static_cast<std::uint32_t>(NumArg(pState, 1, "remove")) : uLen;
    const TValue_t tvRemoved = *pTab->Get(*Uni(pState), TValue_t::Number(uPos));
    for (std::uint32_t uI = uPos; uI < uLen; ++uI)
        *pTab->Set(*Uni(pState), TValue_t::Number(uI)) =
            *pTab->Get(*Uni(pState), TValue_t::Number(uI + 1));
    *pTab->Set(*Uni(pState), TValue_t::Number(uLen)) = TValue_t::Nil();
    Args(pState)[0] = tvRemoved;
    return 1;
}

// --- os / io ----------------------------------------------------------------

std::int32_t LibOsClock(lua_State* pState) {
    Args(pState)[0] = TValue_t::Number(static_cast<double>(std::clock()) / CLOCKS_PER_SEC);
    return 1;
}
std::int32_t LibOsTime(lua_State* pState) {
    Args(pState)[0] = TValue_t::Number(static_cast<double>(std::time(nullptr)));
    return 1;
}
std::int32_t LibIoWrite(lua_State* pState) {
    std::string sPart;
    for (std::int32_t nI = 0; nI < ArgCount(pState); ++nI) {
        ToStringBuf(*Uni(pState), Args(pState)[nI], sPart);
        std::fwrite(sPart.data(), 1, sPart.size(), stdout);
    }
    return 0;
}
std::int32_t LibCollectGarbage(lua_State* pState) {
    Uni(pState)->Gc().CollectNow();
    Args(pState)[0] = TValue_t::Number(
        static_cast<double>(Uni(pState)->Gc().Stats().uTotalBytes) / 1024.0);
    return 1;
}

// --- registration -----------------------------------------------------------

C_GcFunction* NewCFunction(C_Universe& uni, CFunction_f fnImpl,
                           EFastFunc eFfid = EFastFunc::C) {
    auto* pFn = static_cast<C_GcFunction*>(
        uni.Gc().AllocObject(EGcObjectType::Function, C_GcFunction::CAllocSize(0)));
    // The ffid is what lets the trace recorder recognize a builtin and emit
    // its machine instruction instead of refusing the call.
    pFn->m_Header.uExtra1 = static_cast<std::uint8_t>(eFfid);
    pFn->m_Header.uExtra2 = 0;
    pFn->m_rEnv = uni.MakeRef(uni.Globals());
    pFn->m_pPc = &uni.m_insCFuncHeader.uRaw;
    pFn->CFunc() = fnImpl;
    return pFn;
}

void SetField(C_Universe& uni, C_GcTable* pTab, const char* sName, const TValue_t& tvValue) {
    C_GcString* pKey = uni.Interner().Intern(sName);
    *pTab->Set(uni, TValue_t::GcObject(EValueTag::String, pKey)) = tvValue;
}

void RegisterFn(C_Universe& uni, C_GcTable* pTab, const char* sName, CFunction_f fnImpl,
                EFastFunc eFfid = EFastFunc::C) {
    SetField(uni, pTab, sName,
             TValue_t::GcObject(EValueTag::Function, NewCFunction(uni, fnImpl, eFfid)));
}

}  // namespace

void OpenStdLib(C_Universe& uni) {
    C_GcTable* pGlobals = uni.Globals();
    RegisterFn(uni, pGlobals, "print", &LibPrint);
    RegisterFn(uni, pGlobals, "type", &LibType);
    RegisterFn(uni, pGlobals, "tostring", &LibToString, EFastFunc::ToString);
    RegisterFn(uni, pGlobals, "tonumber", &LibToNumber);
    RegisterFn(uni, pGlobals, "next", &LibNext, EFastFunc::Next);
    RegisterFn(uni, pGlobals, "pairs", &LibPairs);
    RegisterFn(uni, pGlobals, "ipairs", &LibIPairs);
    RegisterFn(uni, pGlobals, "setmetatable", &LibSetMetatable);
    RegisterFn(uni, pGlobals, "getmetatable", &LibGetMetatable);
    RegisterFn(uni, pGlobals, "select", &LibSelect);
    RegisterFn(uni, pGlobals, "unpack", &LibUnpack);
    RegisterFn(uni, pGlobals, "rawget", &LibRawGet);
    RegisterFn(uni, pGlobals, "rawset", &LibRawSet);
    RegisterFn(uni, pGlobals, "error", &LibError);
    RegisterFn(uni, pGlobals, "assert", &LibAssert);
    RegisterFn(uni, pGlobals, "pcall", &LibPCall);
    RegisterFn(uni, pGlobals, "collectgarbage", &LibCollectGarbage);

    // pairs/ipairs helper functions parked in the registry.
    *uni.Registry()->Set(
        uni, TValue_t::GcObject(EValueTag::String, uni.Interner().Intern("next"))) =
        TValue_t::GcObject(EValueTag::Function, NewCFunction(uni, &LibNext, EFastFunc::Next));
    *uni.Registry()->Set(
        uni, TValue_t::GcObject(EValueTag::String, uni.Interner().Intern("ipairs_iter"))) =
        TValue_t::GcObject(EValueTag::Function,
                           NewCFunction(uni, &LibIPairsIter, EFastFunc::IPairsAux));

    C_GcTable* pMath = C_GcTable::New(uni, 0, 4);
    SetField(uni, pGlobals, "math", TValue_t::GcObject(EValueTag::Table, pMath));
    RegisterFn(uni, pMath, "floor", &LibMathFloor, EFastFunc::MathFloor);
    RegisterFn(uni, pMath, "ceil", &LibMathCeil, EFastFunc::MathCeil);
    RegisterFn(uni, pMath, "sqrt", &LibMathSqrt, EFastFunc::MathSqrt);
    RegisterFn(uni, pMath, "abs", &LibMathAbs, EFastFunc::MathAbs);
    RegisterFn(uni, pMath, "sin", &LibMathSin);
    RegisterFn(uni, pMath, "cos", &LibMathCos);
    RegisterFn(uni, pMath, "exp", &LibMathExp);
    RegisterFn(uni, pMath, "log", &LibMathLog);
    RegisterFn(uni, pMath, "max", &LibMathMax, EFastFunc::MathMax);
    RegisterFn(uni, pMath, "min", &LibMathMin, EFastFunc::MathMin);
    RegisterFn(uni, pMath, "random", &LibMathRandom);
    SetField(uni, pMath, "huge", TValue_t::Number(HUGE_VAL));
    SetField(uni, pMath, "pi", TValue_t::Number(3.14159265358979323846));

    C_GcTable* pString = C_GcTable::New(uni, 0, 4);
    SetField(uni, pGlobals, "string", TValue_t::GcObject(EValueTag::Table, pString));
    RegisterFn(uni, pString, "len", &LibStringLen, EFastFunc::StringLen);
    RegisterFn(uni, pString, "sub", &LibStringSub, EFastFunc::StringSub);
    RegisterFn(uni, pString, "find", &LibStringFind);
    RegisterFn(uni, pString, "match", &LibStringMatch);
    RegisterFn(uni, pString, "gmatch", &LibStringGMatch);
    RegisterFn(uni, pString, "gsub", &LibStringGSub);
    RegisterFn(uni, pString, "format", &LibStringFormat);
    RegisterFn(uni, pString, "rep", &LibStringRep);
    RegisterFn(uni, pString, "byte", &LibStringByte);
    RegisterFn(uni, pString, "char", &LibStringChar, EFastFunc::StringChar);

    // String methods via the base-type metatable (s:len() etc.).
    C_GcTable* pStringMeta = C_GcTable::New(uni, 0, 1);
    SetField(uni, pStringMeta, "__index", TValue_t::GcObject(EValueTag::Table, pString));
    uni.Meta().SetBaseMetatable(EValueTag::String, pStringMeta);

    C_GcTable* pTable = C_GcTable::New(uni, 0, 2);
    SetField(uni, pGlobals, "table", TValue_t::GcObject(EValueTag::Table, pTable));
    RegisterFn(uni, pTable, "insert", &LibTableInsert, EFastFunc::TableInsert);
    RegisterFn(uni, pTable, "remove", &LibTableRemove);
    RegisterFn(uni, pTable, "concat", &LibTableConcat, EFastFunc::TableConcat);
    RegisterFn(uni, pTable, "unpack", &LibUnpack);

    C_GcTable* pOs = C_GcTable::New(uni, 0, 1);
    SetField(uni, pGlobals, "os", TValue_t::GcObject(EValueTag::Table, pOs));
    RegisterFn(uni, pOs, "clock", &LibOsClock);
    RegisterFn(uni, pOs, "time", &LibOsTime);

    C_GcTable* pIo = C_GcTable::New(uni, 0, 1);
    SetField(uni, pGlobals, "io", TValue_t::GcObject(EValueTag::Table, pIo));
    RegisterFn(uni, pIo, "write", &LibIoWrite);

    SetField(uni, pGlobals, "_G", TValue_t::GcObject(EValueTag::Table, pGlobals));
}

}  // namespace ljx::vm
