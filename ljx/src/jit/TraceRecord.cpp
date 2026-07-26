// LJX — trace recorder: executed bytecodes → typed SSA IR.
//
// The recorder runs alongside the interpreter (the dispatch table is swapped
// so every instruction passes through RecordAndExecute first). It sees the
// live operand values, so it can specialize on the types actually present and
// emit a guard rather than a test — and it can resolve a metatable lookup at
// record time and keep only the guards that make the answer a constant.
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include <sys/mman.h>

#include "ljx/gc/GarbageCollector.hpp"
#include "ljx/jit/TraceJit.hpp"
#include "ljx/rt/Meta.hpp"
#include "ljx/vm/FastFunc.hpp"
#include "ljx/vm/Frame.hpp"
#include "ljx/vm/Interpreter.hpp"
#include "ljx/vm/Object.hpp"

namespace ljx::jit {

using vm::BcIns_t;
using vm::C_GcFunction;
using vm::C_GcProto;
using vm::C_GcString;
using vm::C_GcTable;
using vm::C_GcUpvalue;
using vm::EBcOp;
using vm::EValueTag;
using vm::TValue_t;

bool TraceDebug() noexcept {
    static const bool bOn = std::getenv("LJX_TRACEDEBUG") != nullptr;
    return bOn;
}

namespace {

const char* kIrOpNames[] = {
#define LJX_IR_NAME(name) #name,
    LJX_IR_OPS(LJX_IR_NAME)
#undef LJX_IR_NAME
};

// Byte offsets the trace bakes into address arithmetic. Every one of these is
// pinned by the frozen-layout static_asserts in vm/Object.hpp.
constexpr std::int32_t kOfsTabArray = static_cast<std::int32_t>(offsetof(C_GcTable, m_rArray));
constexpr std::int32_t kOfsTabMeta = static_cast<std::int32_t>(offsetof(C_GcTable, m_rMetatable));
constexpr std::int32_t kOfsTabNodes = static_cast<std::int32_t>(offsetof(C_GcTable, m_rNodes));
constexpr std::int32_t kOfsTabAsize = static_cast<std::int32_t>(offsetof(C_GcTable, m_uArraySize));
constexpr std::int32_t kOfsTabHmask = static_cast<std::int32_t>(offsetof(C_GcTable, m_uHashMask));
constexpr std::int32_t kOfsTabVersion = static_cast<std::int32_t>(offsetof(C_GcTable, m_uVersion));
constexpr std::int32_t kOfsNodeKey = static_cast<std::int32_t>(offsetof(vm::TableNode_t, tvKey));
constexpr std::int32_t kOfsNodeNext = static_cast<std::int32_t>(offsetof(vm::TableNode_t, rNext));
constexpr std::int32_t kNodeSize = static_cast<std::int32_t>(sizeof(vm::TableNode_t));

constexpr std::int32_t kOfsStrSid = static_cast<std::int32_t>(offsetof(C_GcString, m_uSid));

constexpr std::uint32_t kMaxChainWalk = 4;

// True when a double is an exact 32-bit integer — the precondition under which
// the induction argument in GuardEntryInt32 holds.
bool IsInt32Value(double flValue) noexcept {
    return flValue == std::floor(flValue) && flValue >= -2147483648.0 &&
           flValue <= 2147483647.0;
}

}  // namespace

const char* IrOpName(EIrOp eOp) noexcept {
    const auto uIdx = static_cast<std::uint32_t>(eOp);
    return uIdx < static_cast<std::uint32_t>(EIrOp::Count_) ? kIrOpNames[uIdx] : "?";
}

C_TraceJit::~C_TraceJit() {
    if (TraceDebug())
        for (const Trace_t* pTrace : m_vTraces) {
            std::fprintf(stderr, "[trace] #%u entered %llu times\n", pTrace->uNumber,
                         static_cast<unsigned long long>(pTrace->uEntries));
            for (std::size_t uE = 0; uE < pTrace->vExits.size(); ++uE)
                if (pTrace->vExits[uE].uCount)
                    std::fprintf(stderr, "[trace]   exit %zu: %u%s\n", uE,
                                 pTrace->vExits[uE].uCount,
                                 pTrace->vExits[uE].pChild ? " -> child" : "");
        }
    for (Trace_t* pTrace : m_vTraces) delete pTrace;
    if (m_pCodeArena) munmap(m_pCodeArena, kCodeArenaSize);
}

// ---------------------------------------------------------------------------
// IR construction
// ---------------------------------------------------------------------------

EIrType C_TraceJit::ObservedType(const TValue_t& tvValue) const noexcept {
    if (tvValue.IsDouble()) return EIrType::Num;
    switch (static_cast<EValueTag>(tvValue.TagBits())) {
        case EValueTag::Nil: return EIrType::Nil;
        case EValueTag::False: return EIrType::False;
        case EValueTag::True: return EIrType::True;
        case EValueTag::String: return EIrType::Str;
        case EValueTag::Table: return EIrType::Tab;
        case EValueTag::Function: return EIrType::Func;
        default: return EIrType::Nothing;
    }
}

EIrType C_TraceJit::TypeOf(IrRef rRef) const noexcept {
    if (rRef == kIrNone) return EIrType::Nothing;
    if (IsConstRef(rRef)) return m_vConst[kIrBias - 1 - rRef].eType;
    return m_vIns[rRef - kIrBias].eType;
}

TValue_t C_TraceJit::ConstValue(IrRef rRef) const noexcept {
    return TValue_t{m_vConst[kIrBias - 1 - rRef].uValue};
}

IrRef C_TraceJit::Emit(EIrOp eOp, EIrType eType, IrRef rOp1, IrRef rOp2) {
    if (m_eState != ETraceState::Recording) return kIrNone;
    // Common-subexpression elimination over this opcode's chain. Operands are
    // always defined before their use, so the walk is bounded by the newer of
    // the two operand refs — anything older cannot match.
    const IrRef rLimit = rOp1 > rOp2 ? rOp1 : rOp2;
    if (IsPureOp(eOp)) {
        for (IrRef r = m_vChain[static_cast<std::size_t>(eOp)];
             r != kIrNone && r > rLimit;) {
            const IrIns_t& ins = m_vIns[r - kIrBias];
            if (ins.rOp1 == rOp1 && ins.rOp2 == rOp2 && ins.eType == eType) return r;
            r = ins.rPrev;
        }
    }
    if (m_vIns.size() >= kMaxIrIns) {
        AbortRecording(EAbort::TooLong, "IR overflow");
        return kIrNone;
    }
    const auto rNew = static_cast<IrRef>(kIrBias + m_vIns.size());
    m_vIns.push_back(IrIns_t{rOp1, rOp2, eOp, eType,
                             m_vChain[static_cast<std::size_t>(eOp)]});
    m_vInsSnap.push_back(0xffff);
    m_vIntegral.push_back(0);
    m_vForceHoist.push_back(0);
    m_vChain[static_cast<std::size_t>(eOp)] = rNew;
    // A store invalidates the load chains: a later load must not be forwarded
    // across it. Truncating the chain is the whole alias analysis.
    if (eOp == EIrOp::StoreTV || eOp == EIrOp::SStore) {
        m_vChain[static_cast<std::size_t>(EIrOp::LoadTV)] = kIrNone;
        m_vChain[static_cast<std::size_t>(EIrOp::LoadU32)] = kIrNone;
    }
    return rNew;
}

IrRef C_TraceJit::EmitGuard(EIrOp eOp, IrRef rOp1, IrRef rOp2, const BcIns_t* pResumePc) {
    const std::size_t uBefore = m_vIns.size();
    const IrRef rIns = Emit(eOp, EIrType::Nothing, rOp1, rOp2);
    if (rIns == kIrNone || m_vIns.size() == uBefore) return rIns;  // CSE'd: already proven
    m_vInsSnap[rIns - kIrBias] = static_cast<std::uint16_t>(TakeSnapshot(pResumePc));
    return rIns;
}

IrRef C_TraceJit::EmitSnapped(EIrOp eOp, EIrType eType, IrRef rOp1, IrRef rOp2,
                              const BcIns_t* pResumePc) {
    const std::size_t uBefore = m_vIns.size();
    const IrRef rIns = Emit(eOp, eType, rOp1, rOp2);
    if (rIns != kIrNone && m_vIns.size() != uBefore)
        m_vInsSnap[rIns - kIrBias] = static_cast<std::uint16_t>(TakeSnapshot(pResumePc));
    return rIns;
}

bool C_TraceJit::IsIntegralNum(IrRef rRef) const noexcept {
    if (rRef == kIrNone) return false;
    if (IsConstRef(rRef)) {
        const IrConst_t& k = m_vConst[kIrBias - 1 - rRef];
        return k.eType == EIrType::Num && IsInt32Value(std::bit_cast<double>(k.uValue));
    }
    return m_vIntegral[rRef - kIrBias] != 0;
}

void C_TraceJit::MarkIntegral(IrRef rRef) noexcept {
    if (rRef != kIrNone && !IsConstRef(rRef)) m_vIntegral[rRef - kIrBias] = 1;
}

// Narrowing. The IR keeps Lua numbers as doubles, so an array index costs a
// cvttsd2si plus a round-trip exactness guard on EVERY use — the biggest
// per-iteration tax the trace tier pays next to LuaJIT's integer IR. The
// escape is an induction argument: if a value is an exact int32 at trace
// ENTRY, and everything added to it is integral with the loop guard bounding
// its growth, it stays exact forever. So this emits ONE hoisted 32-bit
// round-trip check in the preamble (out-of-range and NaN both fail it), marks
// the ref integral, and every downstream ToInt drops its per-iteration guard.
bool C_TraceJit::GuardEntryInt32(IrRef rRef, const BcIns_t* pResumePc) {
    if (rRef == kIrNone) return false;
    if (IsIntegralNum(rRef)) return true;
    if (IsConstRef(rRef)) return false;
    const std::size_t uBefore = m_vIns.size();
    const IrRef rChk = Emit(EIrOp::ChkInt32, EIrType::Nothing, rRef, kIrNone);
    if (rChk == kIrNone) return false;
    if (m_vIns.size() != uBefore) {
        m_vInsSnap[rChk - kIrBias] = static_cast<std::uint16_t>(TakeSnapshot(pResumePc));
        m_vForceHoist[rChk - kIrBias] = 1;
    }
    MarkIntegral(rRef);
    return m_eState == ETraceState::Recording;
}

// A compiled trace outlives any Lua reference to the objects it specialized
// on: it compares against their addresses and, for the metatable fold, reads
// their version words directly. Anchoring them in the registry — an ordinary
// GC root — keeps those addresses valid for the trace's lifetime without
// teaching the collector anything about traces.
void C_TraceJit::PinValue(const TValue_t& tvValue) {
    if (!tvValue.IsGcObject()) return;
    *m_pUniverse->Registry()->Set(*m_pUniverse, tvValue) = TValue_t::Boolean(true);
}

IrRef C_TraceJit::Constant(const TValue_t& tvValue) {
    PinValue(tvValue);
    const EIrType eType = ObservedType(tvValue);
    for (std::size_t uI = 0; uI < m_vConst.size(); ++uI)
        if (m_vConst[uI].uValue == tvValue.uRaw && m_vConst[uI].eType == eType)
            return static_cast<IrRef>(kIrBias - 1 - uI);
    if (m_vConst.size() >= kIrBias - 1) { AbortRecording(EAbort::TooLong, "constants"); return kIrNone; }
    m_vConst.push_back(IrConst_t{tvValue.uRaw, eType});
    return static_cast<IrRef>(kIrBias - m_vConst.size());
}

IrRef C_TraceJit::ConstantNum(double flValue) { return Constant(TValue_t::Number(flValue)); }

// Operands of real machine instructions must live in registers. Wrapping a
// constant in KLoad lets CSE share it and lets the hoisting pass lift the
// materialization out of the loop, instead of rebuilding the immediate on
// every iteration.
IrRef C_TraceJit::Materialize(IrRef rRef) {
    if (rRef == kIrNone || !IsConstRef(rRef)) return rRef;
    const bool bIntegral = IsIntegralNum(rRef);
    const IrRef rNew = Emit(EIrOp::KLoad, TypeOf(rRef), rRef, kIrNone);
    if (bIntegral) MarkIntegral(rNew);
    return rNew;
}

IrRef C_TraceJit::ConstantInt(std::int64_t nValue) {
    const auto uRaw = static_cast<std::uint64_t>(nValue);
    for (std::size_t uI = 0; uI < m_vConst.size(); ++uI)
        if (m_vConst[uI].uValue == uRaw && m_vConst[uI].eType == EIrType::Int)
            return static_cast<IrRef>(kIrBias - 1 - uI);
    if (m_vConst.size() >= kIrBias - 1) { AbortRecording(EAbort::TooLong, "constants"); return kIrNone; }
    m_vConst.push_back(IrConst_t{uRaw, EIrType::Int});
    return static_cast<IrRef>(kIrBias - m_vConst.size());
}

IrRef C_TraceJit::ConstantPtr(const void* pPtr) {
    const auto uRaw = reinterpret_cast<std::uint64_t>(pPtr);
    for (std::size_t uI = 0; uI < m_vConst.size(); ++uI)
        if (m_vConst[uI].uValue == uRaw && m_vConst[uI].eType == EIrType::Ptr)
            return static_cast<IrRef>(kIrBias - 1 - uI);
    if (m_vConst.size() >= kIrBias - 1) { AbortRecording(EAbort::TooLong, "constants"); return kIrNone; }
    m_vConst.push_back(IrConst_t{uRaw, EIrType::Ptr});
    return static_cast<IrRef>(kIrBias - m_vConst.size());
}

// ---------------------------------------------------------------------------
// Slot map: a stack slot's current IR value, loaded on demand with a guard.
//
// A slot that is READ before it is written gets an entry SLoad, is carried in
// a register across the back edge, and appears in every snapshot. A slot that
// is only ever WRITTEN is written through to the Lua stack at the point of
// assignment, so the interpreter always finds it current after a side exit —
// which is what keeps deoptimization exact without a loop-carry analysis.
// ---------------------------------------------------------------------------

IrRef C_TraceJit::SlotRef(std::int32_t nSlot) {
    if (nSlot < -kSlotBias || nSlot >= kMaxSlot) {
        AbortRecording(EAbort::Unsupported, "slot out of range");
        return kIrNone;
    }
    const auto uIdx = static_cast<std::size_t>(nSlot + kSlotBias);
    if (nSlot > m_nTopSlot) m_nTopSlot = nSlot;
    if (m_vSlotValue[uIdx] != kIrNone) return m_vSlotValue[uIdx];
    const TValue_t tvLive = m_pEntryBase[nSlot];
    const EIrType eType = ObservedType(tvLive);
    if (eType == EIrType::Nothing) {
        AbortRecording(EAbort::BadType, "unsupported value type in slot");
        return kIrNone;
    }
    const IrRef rLoad = Emit(EIrOp::SLoad, eType, static_cast<IrRef>(uIdx), kIrNone);
    if (rLoad == kIrNone) return kIrNone;
    // The entry guard exits to the trace head with nothing to restore.
    m_vInsSnap[rLoad - kIrBias] = 0;
    m_vSlotValue[uIdx] = rLoad;
    m_vSlotEntry[uIdx] = rLoad;
    return rLoad;
}

void C_TraceJit::SetSlot(std::int32_t nSlot, IrRef rValue) {
    if (nSlot < -kSlotBias || nSlot >= kMaxSlot) {
        AbortRecording(EAbort::Unsupported, "slot out of range");
        return;
    }
    if (rValue == kIrNone) { AbortRecording(EAbort::Unsupported, "no value"); return; }
    const auto uIdx = static_cast<std::size_t>(nSlot + kSlotBias);
    if (nSlot > m_nTopSlot) m_nTopSlot = nSlot;
    m_vSlotValue[uIdx] = rValue;
}

std::uint32_t C_TraceJit::TakeSnapshot(const BcIns_t* pResumePc) {
    Snapshot_t snap;
    snap.uFirstSlot = static_cast<std::uint32_t>(m_vSnapSlots.size());
    snap.pResumePc = pResumePc;
    snap.nBaseOffset = m_nBaseOffset;
    for (std::size_t uI = 0; uI < m_vSlotValue.size(); ++uI) {
        if (m_vSlotValue[uI] == kIrNone) continue;
        m_vSnapSlots.push_back(SnapSlot_t{static_cast<std::int32_t>(uI) - kSlotBias,
                                          m_vSlotValue[uI]});
    }
    snap.uSlotCount = static_cast<std::uint32_t>(m_vSnapSlots.size()) - snap.uFirstSlot;
    m_vSnapshots.push_back(snap);
    return static_cast<std::uint32_t>(m_vSnapshots.size() - 1);
}

// ---------------------------------------------------------------------------
// Recording lifecycle
// ---------------------------------------------------------------------------

void C_TraceJit::StartRecording(const BcIns_t* pPc, TValue_t* pBase, const TValue_t* pKBase) {
    m_eState = ETraceState::Recording;
    m_pStartPc = pPc;
    m_pEntryBase = pBase;
    m_pBase = pBase;
    m_pKBase = pKBase;
    m_uRecorded = 0;
    m_pOriginTrace = nullptr;
    m_bPendingCFunc = false;
    m_nBaseOffset = 0;
    m_nTopSlot = 0;
    m_vIns.clear();
    m_vInsSnap.clear();
    m_vIntegral.clear();
    m_vForceHoist.clear();
    m_vConst.clear();
    m_vSnapshots.clear();
    m_vSnapSlots.clear();
    m_vFrames.clear();
    std::memset(m_vChain, 0, sizeof m_vChain);
    m_vSlotValue.assign(static_cast<std::size_t>(kMaxSlot + kSlotBias), kIrNone);
    m_vSlotEntry.assign(static_cast<std::size_t>(kMaxSlot + kSlotBias), kIrNone);
    // Exit 0 is the entry guard: nothing live, resume at the trace head.
    (void)TakeSnapshot(pPc);
    // A head that follows a ForI is a numeric for-loop. When the control
    // triple is integral int32, guard exactly that once in the preamble and
    // the whole loop narrows: entry-exact + integral step + the loop bound
    // keeps the index exact on every iteration (see GuardEntryInt32).
    const BcIns_t insPrev{pPc[-1].uRaw};
    if (insPrev.Op() == EBcOp::ForI || insPrev.Op() == EBcOp::JForI) {
        const std::uint32_t uCtl = insPrev.A();
        bool bAllInt = true;
        for (std::uint32_t uI = 0; uI < 3; ++uI) {
            const TValue_t tvCtl = pBase[uCtl + uI];
            bAllInt &= tvCtl.IsDouble() && IsInt32Value(tvCtl.AsDouble());
        }
        if (bAllInt)
            for (std::uint32_t uI = 0; uI < 4; ++uI)
                (void)GuardEntryInt32(SlotRef(static_cast<std::int32_t>(uCtl + uI)), pPc);
    }
    vm::C_Interpreter::SetRecordMode(*m_pUniverse, true);
    if (TraceDebug())
        std::fprintf(stderr, "[trace] start @%p\n", static_cast<const void*>(pPc));
}

void C_TraceJit::AbortRecording(EAbort eReason, const char* sDetail) {
    if (m_eState != ETraceState::Recording) return;
    m_eState = ETraceState::Idle;
    // Penalize the site so a region that cannot be recorded is not retried on
    // every entry; the penalty blacklists it for good. A side recording
    // penalizes the EXIT that spawned it — its start PC may well be a loop
    // head that a future root recording can still handle.
    if (m_pOriginTrace)
        m_pOriginTrace->vExits[m_uOriginExit].uCount = kBlacklistCount;
    else if (m_pStartPc)
        m_mapHot[m_pStartPc] = kBlacklistCount;
    m_pOriginTrace = nullptr;
    if (TraceDebug())
        std::fprintf(stderr, "[trace] abort (%u): %s\n",
                     static_cast<unsigned>(eReason), sDetail);
    vm::C_Interpreter::SetRecordMode(*m_pUniverse, false);
}

// A snapshot taken BEFORE a slot's first read does not mention that slot,
// because at that point in the recorded iteration nothing had touched it. For
// one straight-line iteration that is correct — the Lua stack still holds the
// entry value. For a LOOP it is not: the slot is carried in a register that
// the back edge keeps current, so after N iterations the stack copy is N
// iterations stale, and an exit through such a snapshot resumes the
// interpreter at an old index.
//
// So once the loop is closed and the full set of loop-carried slots is known,
// every snapshot is refilled with the ones it lacks, naming the ENTRY load —
// which is exactly the register holding that slot's value at the loop top.
// Snapshot 0 is left alone: it is the entry exit, taken before the preamble
// has loaded anything.
void C_TraceJit::FinalizeSnapshots() {
    std::vector<SnapSlot_t> vNew;
    vNew.reserve(m_vSnapSlots.size() * 2);
    for (std::size_t uS = 0; uS < m_vSnapshots.size(); ++uS) {
        Snapshot_t& snap = m_vSnapshots[uS];
        const auto uFirst = static_cast<std::uint32_t>(vNew.size());
        for (std::uint32_t uI = 0; uI < snap.uSlotCount; ++uI)
            vNew.push_back(m_vSnapSlots[snap.uFirstSlot + uI]);
        if (uS != 0) {
            for (std::size_t uSlot = 0; uSlot < m_vSlotEntry.size(); ++uSlot) {
                if (m_vSlotEntry[uSlot] == kIrNone) continue;
                bool bPresent = false;
                for (std::uint32_t uI = 0; uI < snap.uSlotCount; ++uI)
                    if (m_vSnapSlots[snap.uFirstSlot + uI].nSlot ==
                        static_cast<std::int32_t>(uSlot) - kSlotBias)
                        bPresent = true;
                if (bPresent) continue;
                vNew.push_back(SnapSlot_t{static_cast<std::int32_t>(uSlot) - kSlotBias,
                                          m_vSlotEntry[uSlot]});
            }
        }
        snap.uFirstSlot = uFirst;
        snap.uSlotCount = static_cast<std::uint32_t>(vNew.size()) - uFirst;
    }
    m_vSnapSlots = std::move(vNew);
}

// Human-readable IR listing (LJX_TRACEIR=1). Constants print as ~N, the slot
// map as the entry/exit pair per slot.
void C_TraceJit::DumpIr() const {
    std::fprintf(stderr, "---- trace IR (%zu ins, %zu const) ----\n", m_vIns.size(),
                 m_vConst.size());
    for (std::size_t uI = 0; uI < m_vIns.size(); ++uI) {
        const IrIns_t& ins = m_vIns[uI];
        std::fprintf(stderr, "%4zu %-10s t%u ", uI, IrOpName(ins.eOp),
                     static_cast<unsigned>(ins.eType));
        for (int nK = 0; nK < 2; ++nK) {
            const IrRef r = nK == 0 ? ins.rOp1 : ins.rOp2;
            if (r == kIrNone) continue;
            if (ins.eOp == EIrOp::SLoad || ins.eOp == EIrOp::SStore) {
                if (nK == 0) { std::fprintf(stderr, "slot%d ", static_cast<int>(r) - kSlotBias); continue; }
            }
            if (IsConstRef(r))
                std::fprintf(stderr, "K%lld(0x%llx) ",
                             static_cast<long long>(kIrBias - 1 - r),
                             static_cast<unsigned long long>(m_vConst[kIrBias - 1 - r].uValue));
            else
                std::fprintf(stderr, "%zu ", static_cast<std::size_t>(r - kIrBias));
        }
        if (m_vInsSnap[uI] != 0xffff) std::fprintf(stderr, " [snap %u]", m_vInsSnap[uI]);
        std::fprintf(stderr, "\n");
    }
    for (std::size_t uS = 0; uS < m_vSlotValue.size(); ++uS)
        if (m_vSlotValue[uS] != kIrNone || m_vSlotEntry[uS] != kIrNone)
            std::fprintf(stderr, "  slot%-4d entry=%d value=%d\n",
                         static_cast<int>(uS) - kSlotBias,
                         m_vSlotEntry[uS] ? m_vSlotEntry[uS] - kIrBias : -1,
                         m_vSlotValue[uS] ? m_vSlotValue[uS] - kIrBias : -1);
}

void C_TraceJit::CloseLoop() {
    if (!m_vFrames.empty() || m_nBaseOffset != 0) {
        AbortRecording(EAbort::LeftFrame, "loop closed in an inlined frame");
        return;
    }
    // Slots the trace writes but never reads are not carried in registers, so
    // the Lua stack must hold their value at the loop top: a guard early in an
    // iteration can exit before that slot is assigned again, and the
    // interpreter has to see what the PREVIOUS iteration left there. One store
    // at the back edge is enough — a guard after the assignment is covered by
    // the snapshot instead.
    for (std::size_t uI = 0; uI < m_vSlotValue.size(); ++uI)
        if (m_vSlotValue[uI] != kIrNone && m_vSlotEntry[uI] == kIrNone)
            (void)Emit(EIrOp::SStore, TypeOf(m_vSlotValue[uI]), static_cast<IrRef>(uI),
                       m_vSlotValue[uI]);
    (void)Emit(EIrOp::Loop, EIrType::Nothing, kIrNone, kIrNone);
    FinalizeSnapshots();
    if (std::getenv("LJX_TRACEIR")) DumpIr();
    FinishTrace(Assemble(), nullptr);
}

void C_TraceJit::CloseLink(Trace_t* pTarget) {
    // Everything the trace computed goes back to the Lua stack (the End's
    // snapshot), and the chain enters the target, whose preamble re-loads and
    // re-guards its entry state from there. No register contract between the
    // two traces is needed, which is what makes any trace linkable to any
    // other.
    const IrRef rEnd = Emit(EIrOp::End, EIrType::Nothing, kIrNone, kIrNone);
    if (m_eState != ETraceState::Recording) return;   // Emit may have aborted
    m_vInsSnap[rEnd - kIrBias] =
        static_cast<std::uint16_t>(TakeSnapshot(pTarget->pStartPc));
    if (TraceDebug())
        std::fprintf(stderr, "[trace] closing with link -> #%u (start %p)\n",
                     pTarget->uNumber, static_cast<const void*>(pTarget->pStartPc));
    FinalizeSnapshots();
    if (std::getenv("LJX_TRACEIR")) DumpIr();
    FinishTrace(Assemble(), pTarget);
}

void C_TraceJit::FinishTrace(Trace_t* pTrace, Trace_t* pLinkTarget) {
    Trace_t* pOrigin = m_pOriginTrace;
    const std::uint32_t uOriginExit = m_uOriginExit;
    m_pOriginTrace = nullptr;
    m_eState = ETraceState::Idle;
    vm::C_Interpreter::SetRecordMode(*m_pUniverse, false);
    if (!pTrace) {
        if (pOrigin)
            pOrigin->vExits[uOriginExit].uCount = kBlacklistCount;
        else if (m_pStartPc)
            m_mapHot[m_pStartPc] = kBlacklistCount;
        return;
    }
    pTrace->pStartPc = m_pStartPc;
    pTrace->nTopSlot = m_nTopSlot;
    pTrace->uDepth = pOrigin ? pOrigin->uDepth + 1 : 0;
    pTrace->uNumber = static_cast<std::uint32_t>(m_vTraces.size());
    m_vTraces.push_back(pTrace);
    // Registered by start PC even for side traces: a later recording that
    // reaches this PC links here instead of recording the region again.
    m_mapTraces[m_pStartPc] = pTrace;
    if (pTrace->nLinkExit >= 0 && pLinkTarget)
        pTrace->vExits[static_cast<std::size_t>(pTrace->nLinkExit)].pChild = pLinkTarget;
    if (pOrigin) pOrigin->vExits[uOriginExit].pChild = pTrace;
    if (TraceDebug())
        std::fprintf(stderr, "[trace] #%u compiled: %zu ins, %zu exits%s%s\n",
                     pTrace->uNumber, m_vIns.size(), pTrace->vExits.size(),
                     pOrigin ? " (side)" : "", pLinkTarget ? " (linked)" : "");
}

void C_TraceJit::OnHotExit(Trace_t* pParent, std::uint32_t uExit, TValue_t* pBase,
                           const TValue_t* pKBase) {
    if (m_eState != ETraceState::Idle) return;
    TraceExit_t& exit = pParent->vExits[uExit];
    // A trace already compiled at the resume point links directly — unless it
    // is the parent itself. That case is an entry-type exit (exit 0 resumes at
    // the parent's own head): linking it to itself would spin without
    // progress, so record a NEW trace instead, specialized to the types
    // present NOW — a type-polymorphic variant chained off the old one.
    // A chain that keeps sprouting variants is a specialization that does not
    // hold — a per-iteration closure identity, alternating types. Cap the
    // depth; beyond it the exit goes back to the interpreter for good.
    if (pParent->uDepth >= kMaxSideDepth) {
        exit.uCount = kBlacklistCount;
        return;
    }
    if (const auto it = m_mapTraces.find(exit.pResumePc);
        it != m_mapTraces.end() && it->second != pParent) {
        exit.pChild = it->second;
        if (TraceDebug())
            std::fprintf(stderr, "[trace] #%u exit %u linked to #%u\n", pParent->uNumber,
                         uExit, it->second->uNumber);
        return;
    }
    StartRecording(exit.pResumePc, pBase, pKBase);
    if (m_eState != ETraceState::Recording) return;
    m_pOriginTrace = pParent;
    m_uOriginExit = uExit;
    if (TraceDebug())
        std::fprintf(stderr, "[trace] side recording from #%u exit %u\n",
                     pParent->uNumber, uExit);
}

Trace_t* C_TraceJit::OnLoopEdge(const BcIns_t* pHeadPc, TValue_t* pBase,
                                const TValue_t* pKBase) {
    // LJX_NOJIT=1 forces everything through the interpreter — the reference
    // semantics the differential test compares compiled output against.
    static const bool bDisabled = std::getenv("LJX_NOJIT") != nullptr;
    if (bDisabled || m_eState == ETraceState::Recording) return nullptr;
    if (auto it = m_mapTraces.find(pHeadPc); it != m_mapTraces.end()) return it->second;
    // The caller only gets here when the hashed hot counter BORROWED, which
    // already means kHotLoopThreshold iterations of this back edge. Counting
    // again here would multiply the two thresholds together — 56 x 53 ~ 3000
    // iterations before a trace is even attempted, which is more than most
    // real loops ever run. This map is the blacklist, not a second counter.
    if (const auto it = m_mapHot.find(pHeadPc);
        it != m_mapHot.end() && it->second >= kBlacklistCount)
        return nullptr;
    StartRecording(pHeadPc, pBase, pKBase);
    return nullptr;
}

// ---------------------------------------------------------------------------
// Per-bytecode recording
// ---------------------------------------------------------------------------

void C_TraceJit::RecordInstruction(const BcIns_t* pPc, TValue_t* pBase, const TValue_t* pKBase) {
    if (m_eState != ETraceState::Recording) return;
    m_pBase = pBase;
    m_pKBase = pKBase;
    const std::ptrdiff_t nOffset = pBase - m_pEntryBase;
    if (nOffset < 0 || nOffset > kMaxSlot) {
        AbortRecording(EAbort::LeftFrame, "frame left the trace");
        return;
    }
    m_nBaseOffset = static_cast<std::int32_t>(nOffset);
    if (pPc == m_pStartPc && m_uRecorded != 0) { CloseLoop(); return; }
    // Reaching a PC that already has a compiled trace ends this recording by
    // LINKING into it. This is what stitches side traces back into their loop,
    // lets an outer loop's stem enter an inner loop's trace, and — crucially —
    // stops the recorder before a J-variant op could run the other trace
    // natively underneath it, which would hide those bytecodes and leave the
    // slot map stale.
    if (m_uRecorded != 0) {
        if (const auto itLink = m_mapTraces.find(pPc); itLink != m_mapTraces.end()) {
            CloseLink(itLink->second);
            return;
        }
    }
    if (++m_uRecorded > kMaxRecordedIns) {
        AbortRecording(EAbort::TooLong, "recorded instruction budget");
        return;
    }
    const BcIns_t ins{pPc->uRaw};
    if (!RecordOne(ins, pPc + 1) && m_eState == ETraceState::Recording)
        AbortRecording(EAbort::Unsupported, vm::OpName(ins.Op()));
}

bool C_TraceJit::RecordOne(const BcIns_t& ins, const BcIns_t* pNext) {
    const std::int32_t nB = m_nBaseOffset;
    const std::uint32_t uA = ins.A();
    switch (ins.Op()) {
        case EBcOp::Mov: {
            const IrRef r = SlotRef(nB + ins.D());
            if (r == kIrNone) return false;
            SetSlot(nB + uA, r);
            return true;
        }
        case EBcOp::KShort:
            SetSlot(nB + uA, ConstantNum(static_cast<double>(
                                 static_cast<std::int16_t>(ins.D()))));
            return true;
        case EBcOp::KNum:
            SetSlot(nB + uA, Constant(m_pKBase[ins.D()]));
            return true;
        case EBcOp::KStr: {
            const auto* pKgc = reinterpret_cast<const core::GcRef_t*>(m_pKBase);
            auto* pStr = m_pUniverse->Deref<C_GcString>(
                pKgc[-static_cast<std::int32_t>(ins.D() + 1)]);
            SetSlot(nB + uA, Constant(TValue_t::GcObject(EValueTag::String, pStr)));
            return true;
        }
        case EBcOp::KPri:
            SetSlot(nB + uA, Constant(ins.D() == 0 ? TValue_t::Nil()
                                                   : TValue_t::Boolean(ins.D() == 2)));
            return true;
        case EBcOp::KNil: {
            const IrRef rNil = Constant(TValue_t::Nil());
            for (std::uint32_t uI = uA; uI <= ins.D(); ++uI)
                SetSlot(nB + static_cast<std::int32_t>(uI), rNil);
            return true;
        }
        case EBcOp::Unm: {
            const IrRef r = SlotRef(nB + ins.D());
            if (TypeOf(r) != EIrType::Num) return false;
            SetSlot(nB + uA, Emit(EIrOp::Neg, EIrType::Num, r, kIrNone));
            return true;
        }

        case EBcOp::AddVV: return RecordArith(ins, EIrOp::Add, 0);
        case EBcOp::SubVV: return RecordArith(ins, EIrOp::Sub, 0);
        case EBcOp::MulVV: return RecordArith(ins, EIrOp::Mul, 0);
        case EBcOp::DivVV: return RecordArith(ins, EIrOp::Div, 0);
        case EBcOp::ModVV: return RecordArith(ins, EIrOp::Mod, 0);
        case EBcOp::AddVN: return RecordArith(ins, EIrOp::Add, 1);
        case EBcOp::SubVN: return RecordArith(ins, EIrOp::Sub, 1);
        case EBcOp::MulVN: return RecordArith(ins, EIrOp::Mul, 1);
        case EBcOp::DivVN: return RecordArith(ins, EIrOp::Div, 1);
        case EBcOp::ModVN: return RecordArith(ins, EIrOp::Mod, 1);
        case EBcOp::AddNV: return RecordArith(ins, EIrOp::Add, 2);
        case EBcOp::SubNV: return RecordArith(ins, EIrOp::Sub, 2);
        case EBcOp::MulNV: return RecordArith(ins, EIrOp::Mul, 2);
        case EBcOp::DivNV: return RecordArith(ins, EIrOp::Div, 2);
        case EBcOp::ModNV: return RecordArith(ins, EIrOp::Mod, 2);

        case EBcOp::IsLt: case EBcOp::IsGe: case EBcOp::IsLe: case EBcOp::IsGt:
        case EBcOp::IsEqV: case EBcOp::IsNeV: case EBcOp::IsEqS: case EBcOp::IsNeS:
        case EBcOp::IsEqN: case EBcOp::IsNeN: case EBcOp::IsEqP: case EBcOp::IsNeP:
            return RecordCompare(ins, pNext);
        case EBcOp::IsT: case EBcOp::IsF: case EBcOp::IsTC: case EBcOp::IsFC:
            return RecordTest(ins, pNext);

        case EBcOp::Jmp:
        case EBcOp::Loop: case EBcOp::ILoop: case EBcOp::JLoop:
        case EBcOp::FuncF: case EBcOp::IFuncF: case EBcOp::JFuncF:
            return true;
        case EBcOp::FuncC: case EBcOp::FuncCW:
            return SkipCFuncHeader();

        case EBcOp::ForI: case EBcOp::JForI:
            return RecordForI(ins, pNext);
        case EBcOp::ForL: case EBcOp::IForL: case EBcOp::JForL:
            return RecordForL(ins, pNext);

        case EBcOp::TGetV: {
            const std::int32_t nTab = nB + (ins.D() >> 8);
            const std::int32_t nKey = nB + (ins.D() & 0xff);
            const TValue_t tvKey = m_pBase[ins.D() & 0xff];
            return RecordTableGet(ins, pNext, tvKey, nTab, SlotRef(nKey));
        }
        case EBcOp::TGetS: {
            const auto* pKgc = reinterpret_cast<const core::GcRef_t*>(m_pKBase);
            auto* pStr = m_pUniverse->Deref<C_GcString>(
                pKgc[-static_cast<std::int32_t>((ins.D() & 0xff) + 1)]);
            return RecordTableGet(ins, pNext, TValue_t::GcObject(EValueTag::String, pStr),
                                  nB + (ins.D() >> 8), kIrNone);
        }
        case EBcOp::TGetB:
            return RecordTableGet(ins, pNext,
                                  TValue_t::Number(static_cast<double>(ins.D() & 0xff)),
                                  nB + (ins.D() >> 8), kIrNone);
        case EBcOp::TSetV: {
            const TValue_t tvKey = m_pBase[ins.D() & 0xff];
            return RecordTableSet(ins, pNext, tvKey, nB + (ins.D() >> 8));
        }
        case EBcOp::TSetS: {
            const auto* pKgc = reinterpret_cast<const core::GcRef_t*>(m_pKBase);
            auto* pStr = m_pUniverse->Deref<C_GcString>(
                pKgc[-static_cast<std::int32_t>((ins.D() & 0xff) + 1)]);
            return RecordTableSet(ins, pNext, TValue_t::GcObject(EValueTag::String, pStr),
                                  nB + (ins.D() >> 8));
        }

        case EBcOp::UGet: {
            const IrRef rFunc = SlotRef(nB - 2);
            if (TypeOf(rFunc) != EIrType::Func) return false;
            auto* pFn = static_cast<C_GcFunction*>(m_pBase[-2].AsGcPointer());
            if (!IsConstRef(rFunc))
                (void)EmitGuard(EIrOp::GuardEq, rFunc,
                                Materialize(Constant(
                                    TValue_t::GcObject(EValueTag::Function, pFn))),
                                pNext - 1);
            auto* pUpval = m_pUniverse->Deref<C_GcUpvalue>(pFn->UpvalRefs()[ins.D()]);
            PinValue(TValue_t::GcObject(EValueTag::UpValue, pUpval));
            // The cell can migrate when the upvalue is closed: guard it, then
            // the value's address is a compile-time constant.
            (void)EmitGuard(EIrOp::GuardEqI,
                            Emit(EIrOp::LoadU32, EIrType::Int,
                                 ConstantPtr(&pUpval->m_rValue), kIrNone),
                            ConstantInt(pUpval->m_rValue.uIndex), pNext - 1);
            auto* pCell = static_cast<TValue_t*>(
                core::RefToPtr(m_pUniverse->ArenaBase(), pUpval->m_rValue));
            const EIrType eType = ObservedType(*pCell);
            if (eType == EIrType::Nothing) return false;
            const IrRef rVal =
                EmitSnapped(EIrOp::LoadTV, eType, ConstantPtr(pCell), kIrNone, pNext - 1);
            if (rVal == kIrNone) return false;
            SetSlot(nB + uA, rVal);
            return true;
        }

        case EBcOp::GGet: {
            const auto* pKgc = reinterpret_cast<const core::GcRef_t*>(m_pKBase);
            auto* pStr = m_pUniverse->Deref<C_GcString>(
                pKgc[-static_cast<std::int32_t>(ins.D() + 1)]);
            C_GcTable* pGlobals = m_pUniverse->Globals();
            PinValue(TValue_t::GcObject(EValueTag::Table, pGlobals));
            bool bAbsent = false;
            const IrRef rNode = HashNodeRef(ConstantPtr(pGlobals), pGlobals,
                                            TValue_t::GcObject(EValueTag::String, pStr),
                                            kIrNone, pNext - 1, bAbsent);
            if (m_eState != ETraceState::Recording) return false;
            if (bAbsent) { SetSlot(nB + uA, Constant(TValue_t::Nil())); return true; }
            if (rNode == kIrNone) return false;
            const TValue_t* pSlot = pGlobals->GetStr(*m_pUniverse, pStr);
            const EIrType eType = ObservedType(*pSlot);
            if (eType == EIrType::Nothing) return false;
            const IrRef rVal = EmitSnapped(EIrOp::LoadTV, eType, rNode, kIrNone, pNext - 1);
            if (rVal == kIrNone) return false;
            SetSlot(nB + uA, rVal);
            return true;
        }

        case EBcOp::IterC: case EBcOp::IterN: {
            const TValue_t tvIter = m_pBase[uA - 3];
            if (!tvIter.Is(EValueTag::Function)) return false;
            auto* pIter = static_cast<C_GcFunction*>(tvIter.AsGcPointer());
            if (static_cast<vm::EFastFunc>(pIter->m_Header.uExtra1) !=
                vm::EFastFunc::IPairsAux)
                return false;
            const IrRef rIter = SlotRef(nB + static_cast<std::int32_t>(uA) - 3);
            if (rIter == kIrNone) return false;
            if (!IsConstRef(rIter))
                (void)EmitGuard(EIrOp::GuardEq, rIter, Materialize(Constant(tvIter)),
                                pNext - 1);
            if (m_eState != ETraceState::Recording) return false;
            if (ins.B() != 3) return false;   // `for i, v in ipairs(t)` exactly
            SetSlot(nB + static_cast<std::int32_t>(uA), rIter);
            if (!RecordIPairsIter(ins, pNext - 1)) return false;
            m_bPendingCFunc = true;
            return true;
        }
        case EBcOp::IterL: case EBcOp::IIterL: case EBcOp::JIterL: {
            // The control slot is non-nil on the recorded path — the iterator's
            // own guards proved it — so the branch folds and only the copy of
            // the control variable remains.
            const IrRef rCtl = SlotRef(nB + static_cast<std::int32_t>(uA));
            if (rCtl == kIrNone) return false;
            const EIrType eCtl = TypeOf(rCtl);
            if (eCtl == EIrType::Nil || eCtl == EIrType::Nothing) return false;
            SetSlot(nB + static_cast<std::int32_t>(uA) - 1, rCtl);
            return true;
        }

        case EBcOp::Call: return RecordCall(ins, pNext);
        case EBcOp::Ret0: return RecordReturn(ins, 0, 0);
        case EBcOp::Ret1: return RecordReturn(ins, uA, 1);
        case EBcOp::Ret: return RecordReturn(ins, uA, ins.D() - 1u);

        default:
            return false;
    }
}

bool C_TraceJit::RecordArith(const BcIns_t& ins, EIrOp eOp, int nKind) {
    const std::int32_t nB = m_nBaseOffset;
    IrRef rLeft, rRight;
    if (nKind == 0) {
        rLeft = SlotRef(nB + (ins.D() >> 8));
        rRight = SlotRef(nB + (ins.D() & 0xff));
    } else if (nKind == 1) {
        rLeft = SlotRef(nB + (ins.D() >> 8));
        rRight = Constant(m_pKBase[ins.D() & 0xff]);
    } else {
        rLeft = Constant(m_pKBase[ins.D() & 0xff]);
        rRight = SlotRef(nB + (ins.D() >> 8));
    }
    if (TypeOf(rLeft) != EIrType::Num || TypeOf(rRight) != EIrType::Num) return false;
    const IrRef rRes = Emit(eOp, EIrType::Num, Materialize(rLeft), Materialize(rRight));
    if (rRes == kIrNone) return false;
    // Sum of two exact int32 is exact (< 2^33 << 2^53); products can round.
    if ((eOp == EIrOp::Add || eOp == EIrOp::Sub) && IsIntegralNum(rLeft) &&
        IsIntegralNum(rRight))
        MarkIntegral(rRes);
    SetSlot(nB + ins.A(), rRes);
    return true;
}

bool C_TraceJit::RecordCompare(const BcIns_t& ins, const BcIns_t* pNext) {
    const std::int32_t nB = m_nBaseOffset;
    const BcIns_t insJmp{pNext->uRaw};
    IrRef rLeft = kIrNone, rRight = kIrNone;
    bool bTaken = false;
    const EBcOp eOp = ins.Op();
    const TValue_t tvA = m_pBase[ins.A()];

    if (eOp == EBcOp::IsLt || eOp == EBcOp::IsGe || eOp == EBcOp::IsLe || eOp == EBcOp::IsGt) {
        rLeft = SlotRef(nB + ins.A());
        rRight = SlotRef(nB + ins.D());
        if (TypeOf(rLeft) != EIrType::Num || TypeOf(rRight) != EIrType::Num) return false;
        const double flA = m_pBase[ins.A()].AsDouble();
        const double flB = m_pBase[ins.D()].AsDouble();
        EIrOp eGuard;
        switch (eOp) {
            case EBcOp::IsLt: bTaken = flA < flB; eGuard = bTaken ? EIrOp::GuardLt : EIrOp::GuardGe; break;
            case EBcOp::IsGe: bTaken = !(flA < flB); eGuard = bTaken ? EIrOp::GuardGe : EIrOp::GuardLt; break;
            case EBcOp::IsLe: bTaken = flA <= flB; eGuard = bTaken ? EIrOp::GuardLe : EIrOp::GuardGt; break;
            default:          bTaken = !(flA <= flB); eGuard = bTaken ? EIrOp::GuardGt : EIrOp::GuardLe; break;
        }
        const BcIns_t* pExitPc = bTaken ? pNext + 1 : pNext + 1 + insJmp.JumpTarget();
        (void)EmitGuard(eGuard, Materialize(rLeft), Materialize(rRight), pExitPc);
        return m_eState == ETraceState::Recording;
    }

    // Equality forms: a single 64-bit word compare after canonicalization.
    TValue_t tvB;
    switch (eOp) {
        case EBcOp::IsEqV: case EBcOp::IsNeV:
            rRight = SlotRef(nB + ins.D());
            tvB = m_pBase[ins.D()];
            break;
        case EBcOp::IsEqS: case EBcOp::IsNeS: {
            const auto* pKgc = reinterpret_cast<const core::GcRef_t*>(m_pKBase);
            auto* pStr = m_pUniverse->Deref<C_GcString>(
                pKgc[-static_cast<std::int32_t>(ins.D() + 1)]);
            tvB = TValue_t::GcObject(EValueTag::String, pStr);
            rRight = Constant(tvB);
            break;
        }
        case EBcOp::IsEqN: case EBcOp::IsNeN:
            tvB = m_pKBase[ins.D()];
            rRight = Constant(tvB);
            break;
        default:
            tvB = ins.D() == 0 ? TValue_t::Nil() : TValue_t::Boolean(ins.D() == 2);
            rRight = Constant(tvB);
            break;
    }
    rLeft = SlotRef(nB + ins.A());
    if (rLeft == kIrNone || rRight == kIrNone) return false;
    const bool bIsEq = eOp == EBcOp::IsEqV || eOp == EBcOp::IsEqS ||
                       eOp == EBcOp::IsEqN || eOp == EBcOp::IsEqP;
    const EIrType eLeft = TypeOf(rLeft), eRight = TypeOf(rRight);
    bool bEqual;
    EIrOp eGuardOp;
    if (eLeft == EIrType::Num && eRight == EIrType::Num) {
        // Numbers compare by VALUE — raw bits are wrong for -0.0 == 0.0, and
        // the operands live in xmm registers, so the guard is a ucomisd.
        bEqual = tvA.AsDouble() == tvB.AsDouble();
        eGuardOp = bEqual ? EIrOp::GuardFEq : EIrOp::GuardFNe;
    } else if (eLeft != eRight) {
        // Different guarded types can never be equal: the outcome is static
        // under the operands' own type guards, no comparison needed.
        return m_eState == ETraceState::Recording;
    } else {
        // Same GC/primitive type: identity IS equality, one raw compare.
        bEqual = tvA.uRaw == tvB.uRaw;
        eGuardOp = bEqual ? EIrOp::GuardEq : EIrOp::GuardNe;
    }
    bTaken = bIsEq ? bEqual : !bEqual;
    const BcIns_t* pExitPc = bTaken ? pNext + 1 : pNext + 1 + insJmp.JumpTarget();
    (void)EmitGuard(eGuardOp, Materialize(rLeft), Materialize(rRight), pExitPc);
    return m_eState == ETraceState::Recording;
}

bool C_TraceJit::RecordTest(const BcIns_t& ins, const BcIns_t* pNext) {
    const std::int32_t nB = m_nBaseOffset;
    const BcIns_t insJmp{pNext->uRaw};
    const IrRef rVal = SlotRef(nB + ins.D());
    if (rVal == kIrNone) return false;
    const EIrType eType = TypeOf(rVal);
    // Truthiness follows from the guarded type alone — no run-time test at all.
    const bool bTruthy = eType != EIrType::Nil && eType != EIrType::False;
    const bool bTaken = (ins.Op() == EBcOp::IsT || ins.Op() == EBcOp::IsTC) ? bTruthy : !bTruthy;
    if (bTaken && (ins.Op() == EBcOp::IsTC || ins.Op() == EBcOp::IsFC))
        SetSlot(nB + ins.A(), rVal);
    (void)insJmp;
    return true;
}

// ForI is a loop ENTRY, not a back edge: type-check the control triple, guard
// the direction actually taken (enter or skip), and set the visible variable.
// Recording one makes traces that CROSS an inner numeric loop possible — the
// inner head is reached next, and if it has a compiled trace, CloseLink
// stitches into it.
bool C_TraceJit::RecordForI(const BcIns_t& ins, const BcIns_t* pNext) {
    const std::int32_t nB = m_nBaseOffset;
    const std::int32_t nSlot = nB + static_cast<std::int32_t>(ins.A());
    const IrRef rIdx = SlotRef(nSlot);
    const IrRef rStop = SlotRef(nSlot + 1);
    const IrRef rStep = SlotRef(nSlot + 2);
    if (TypeOf(rIdx) != EIrType::Num || TypeOf(rStop) != EIrType::Num ||
        TypeOf(rStep) != EIrType::Num)
        return false;
    const double flIdx = m_pBase[ins.A()].AsDouble();
    const double flStop = m_pBase[ins.A() + 1].AsDouble();
    const double flStep = m_pBase[ins.A() + 2].AsDouble();
    // The interpreter copies the visible variable before the entry test.
    SetSlot(nSlot + 3, rIdx);
    const bool bEnter = flStep >= 0 ? flIdx <= flStop : flIdx >= flStop;
    const BcIns_t* pSkip = pNext + ins.JumpTarget();
    if (bEnter)
        (void)EmitGuard(flStep >= 0 ? EIrOp::GuardLe : EIrOp::GuardGe,
                        Materialize(rIdx), Materialize(rStop), pSkip);
    else
        (void)EmitGuard(flStep >= 0 ? EIrOp::GuardGt : EIrOp::GuardLt,
                        Materialize(rIdx), Materialize(rStop), pNext);
    return m_eState == ETraceState::Recording;
}

bool C_TraceJit::RecordForL(const BcIns_t& ins, const BcIns_t* pNext) {
    const std::int32_t nB = m_nBaseOffset;
    const std::int32_t nSlot = nB + static_cast<std::int32_t>(ins.A());
    const IrRef rIdx = SlotRef(nSlot);
    const IrRef rStop = SlotRef(nSlot + 1);
    const IrRef rStep = SlotRef(nSlot + 2);
    if (TypeOf(rIdx) != EIrType::Num || TypeOf(rStop) != EIrType::Num ||
        TypeOf(rStep) != EIrType::Num)
        return false;
    const double flStep = m_pBase[ins.A() + 2].AsDouble();
    const double flNew = m_pBase[ins.A()].AsDouble() + flStep;
    const double flStop = m_pBase[ins.A() + 1].AsDouble();
    const IrRef rNew = Emit(EIrOp::Add, EIrType::Num, rIdx, rStep);
    if (rNew == kIrNone) return false;
    if (IsIntegralNum(rIdx) && IsIntegralNum(rStep)) MarkIntegral(rNew);
    // The index update, and the visible copy, must be in the guard's snapshot
    // BEFORE the guard: the interpreter resuming on either side of the back
    // edge expects them current. (The copy is dead on the loop-exit side —
    // the loop variable's scope ends with the loop — so writing it there too
    // is unobservable and keeps the two directions uniform.)
    SetSlot(nSlot, rNew);
    SetSlot(nSlot + 3, rNew);
    const bool bContinue = flStep >= 0 ? flNew <= flStop : flNew >= flStop;
    if (bContinue) {
        // Guard that the loop keeps going; the exit resumes past the back edge.
        (void)EmitGuard(flStep >= 0 ? EIrOp::GuardLe : EIrOp::GuardGe, rNew,
                        Materialize(rStop), pNext);
    } else {
        // The recorded iteration was the loop's LAST: assert the exit
        // direction, and resume at the body start if the loop continues after
        // all. Without this split, a trace recorded on a final iteration
        // would carry a continue-guard in front of post-loop code.
        (void)EmitGuard(flStep >= 0 ? EIrOp::GuardGt : EIrOp::GuardLt, rNew,
                        Materialize(rStop), pNext + ins.JumpTarget());
    }
    return m_eState == ETraceState::Recording;
}

// ---------------------------------------------------------------------------
// Table access
// ---------------------------------------------------------------------------

IrRef C_TraceJit::HashNodeRef(IrRef rTabPtr, const C_GcTable* pTab, TValue_t tvKey,
                              IrRef rKeyRef, const BcIns_t* pResumePc, bool& bAbsent) {
    bAbsent = false;
    if (!tvKey.Is(EValueTag::String)) return kIrNone;   // only string keys, for now
    const std::uintptr_t uArena = m_pUniverse->ArenaBase();
    const auto* pStr = static_cast<const C_GcString*>(tvKey.AsGcPointer());

    // The main position is computed the way the runtime computes it — mask
    // loaded, string id loaded when the key is not a constant — so the trace
    // works for ANY key and survives a rehash instead of guarding the mask.
    const IrRef rHmask = Emit(EIrOp::LoadU32, EIrType::Int,
                              Emit(EIrOp::AddK, EIrType::Ptr, rTabPtr, ConstantInt(kOfsTabHmask)),
                              kIrNone);
    IrRef rSid;
    IrRef rKeyTagged;
    if (rKeyRef == kIrNone || IsConstRef(rKeyRef)) {
        rSid = ConstantInt(pStr->m_uSid);
        rKeyTagged = Constant(tvKey);
    } else {
        if (TypeOf(rKeyRef) != EIrType::Str) return kIrNone;
        rSid = Emit(EIrOp::LoadU32, EIrType::Int,
                    Emit(EIrOp::AddK, EIrType::Ptr,
                         Emit(EIrOp::TabPtr, EIrType::Ptr, rKeyRef, kIrNone),
                         ConstantInt(kOfsStrSid)),
                    kIrNone);
        rKeyTagged = rKeyRef;
    }
    const IrRef rMain = Emit(EIrOp::AndInt, EIrType::Int, Materialize(rSid), rHmask);
    // A node is three granules wide, so the byte offset is an IdxPtr scale.
    const IrRef rGranule = Emit(EIrOp::MulK, EIrType::Int, rMain,
                                ConstantInt(kNodeSize / 8));
    const IrRef rNodes = Emit(EIrOp::RefPtr, EIrType::Ptr,
                              Emit(EIrOp::LoadU32, EIrType::Int,
                                   Emit(EIrOp::AddK, EIrType::Ptr, rTabPtr,
                                        ConstantInt(kOfsTabNodes)),
                                   kIrNone),
                              kIrNone);
    IrRef rNode = Emit(EIrOp::IdxPtr, EIrType::Ptr, rNodes, rGranule);
    if (rNode == kIrNone) return kIrNone;

    const auto* pNode = static_cast<const vm::TableNode_t*>(core::RefToPtr(uArena, pTab->m_rNodes)) +
                        (pStr->m_uSid & pTab->m_uHashMask);
    for (std::uint32_t uStep = 0; uStep < kMaxChainWalk; ++uStep) {
        const IrRef rNodeKey = Emit(EIrOp::LoadTV, EIrType::Int,
                                    Emit(EIrOp::AddK, EIrType::Ptr, rNode,
                                         ConstantInt(kOfsNodeKey)),
                                    kIrNone);
        if (pNode->tvKey == tvKey) {
            (void)EmitGuard(EIrOp::GuardEq, rNodeKey, Materialize(rKeyTagged), pResumePc);
            return m_eState == ETraceState::Recording ? rNode : kIrNone;
        }
        (void)EmitGuard(EIrOp::GuardNe, rNodeKey, Materialize(rKeyTagged), pResumePc);
        if (m_eState != ETraceState::Recording) return kIrNone;
        const IrRef rNextIdx = Emit(EIrOp::LoadU32, EIrType::Int,
                                    Emit(EIrOp::AddK, EIrType::Ptr, rNode,
                                         ConstantInt(kOfsNodeNext)),
                                    kIrNone);
        if (pNode->rNext.IsNull()) {
            (void)EmitGuard(EIrOp::GuardEqI, rNextIdx, ConstantInt(0), pResumePc);
            bAbsent = m_eState == ETraceState::Recording;
            return kIrNone;
        }
        rNode = Emit(EIrOp::RefPtr, EIrType::Ptr, rNextIdx, kIrNone);
        pNode = static_cast<const vm::TableNode_t*>(core::RefToPtr(uArena, pNode->rNext));
        if (rNode == kIrNone) return kIrNone;
    }
    return kIrNone;   // chain too long: abort
}

bool C_TraceJit::RecordTableGet(const BcIns_t& ins, const BcIns_t* pNext, TValue_t tvKey,
                                std::int32_t nTabSlot, IrRef rKeyDynamic) {
    const std::int32_t nB = m_nBaseOffset;
    const IrRef rTab = SlotRef(nTabSlot);
    if (TypeOf(rTab) != EIrType::Tab) return false;
    auto* pTab = static_cast<C_GcTable*>(m_pEntryBase[nTabSlot].AsGcPointer());
    const IrRef rTabPtr = Emit(EIrOp::TabPtr, EIrType::Ptr, rTab, kIrNone);
    const BcIns_t* pResumePc = pNext - 1;

    // --- array part ---------------------------------------------------------
    if (tvKey.IsDouble()) {
        const double flKey = tvKey.AsDouble();
        const auto nKey = static_cast<std::int64_t>(flKey);
        if (static_cast<double>(nKey) != flKey) return false;
        if (static_cast<std::uint64_t>(nKey) >= pTab->m_uArraySize) return false;
        IrRef rIdx;
        if (rKeyDynamic != kIrNone) {
            if (TypeOf(rKeyDynamic) != EIrType::Num) return false;
            rIdx = IsIntegralNum(rKeyDynamic)
                       ? Emit(EIrOp::ToInt, EIrType::Int, rKeyDynamic, kIrNone)
                       : EmitSnapped(EIrOp::ToInt, EIrType::Int, rKeyDynamic, kIrNone,
                                     pResumePc);
            if (rIdx == kIrNone) return false;
        } else {
            rIdx = ConstantInt(nKey);
        }
        const IrRef rSize = Emit(EIrOp::LoadU32, EIrType::Int,
                                 Emit(EIrOp::AddK, EIrType::Ptr, rTabPtr,
                                      ConstantInt(kOfsTabAsize)),
                                 kIrNone);
        (void)EmitGuard(EIrOp::GuardBelow, Materialize(rIdx), rSize, pResumePc);
        const IrRef rArr = Emit(EIrOp::RefPtr, EIrType::Ptr,
                                Emit(EIrOp::LoadU32, EIrType::Int,
                                     Emit(EIrOp::AddK, EIrType::Ptr, rTabPtr,
                                          ConstantInt(kOfsTabArray)),
                                     kIrNone),
                                kIrNone);
        const IrRef rElem = Emit(EIrOp::IdxPtr, EIrType::Ptr, rArr, Materialize(rIdx));
        const TValue_t* pElem =
            static_cast<TValue_t*>(core::RefToPtr(m_pUniverse->ArenaBase(), pTab->m_rArray)) + nKey;
        const EIrType eType = ObservedType(*pElem);
        if (eType == EIrType::Nothing) return false;
        const IrRef rVal = EmitSnapped(EIrOp::LoadTV, eType, rElem, kIrNone, pResumePc);
        if (rVal == kIrNone) return false;
        SetSlot(nB + ins.A(), rVal);
        return true;
    }

    if (!tvKey.Is(EValueTag::String)) return false;

    // --- hash part of the receiver -----------------------------------------
    bool bAbsent = false;
    const IrRef rNode = HashNodeRef(rTabPtr, pTab, tvKey, rKeyDynamic, pResumePc, bAbsent);
    if (m_eState != ETraceState::Recording) return false;
    if (!bAbsent && rNode == kIrNone) return false;
    if (!bAbsent) {
        const TValue_t* pSlot = pTab->Get(*m_pUniverse, tvKey);
        if (!pSlot || pSlot->IsNil()) return false;
        const EIrType eType = ObservedType(*pSlot);
        if (eType == EIrType::Nothing) return false;
        const IrRef rVal = EmitSnapped(EIrOp::LoadTV, eType, rNode, kIrNone, pResumePc);
        if (rVal == kIrNone) return false;
        SetSlot(nB + ins.A(), rVal);
        return true;
    }

    // --- the key is absent: plain nil, or the metatable chain ---------------
    if (pTab->m_rMetatable.IsNull()) {
        (void)EmitGuard(EIrOp::GuardEqI,
                        Emit(EIrOp::LoadU32, EIrType::Int,
                             Emit(EIrOp::AddK, EIrType::Ptr, rTabPtr, ConstantInt(kOfsTabMeta)),
                             kIrNone),
                        ConstantInt(0), pResumePc);
        SetSlot(nB + ins.A(), Constant(TValue_t::Nil()));
        return true;
    }

    // This is the method-lookup shape. Guard the metatable's identity and the
    // STRUCTURAL versions of the two tables the chain went through: those pin
    // the two slot addresses involved, which fold to constants. The values in
    // them are then re-checked/loaded — a plain store bumps nothing, and this
    // is exactly what makes that correct: the new value flows out of the same
    // slot. A call site stays monomorphic through RecordCall's identity guard
    // on the loaded function.
    (void)EmitGuard(EIrOp::GuardEqI,
                    Emit(EIrOp::LoadU32, EIrType::Int,
                         Emit(EIrOp::AddK, EIrType::Ptr, rTabPtr, ConstantInt(kOfsTabMeta)),
                         kIrNone),
                    ConstantInt(pTab->m_rMetatable.uIndex), pResumePc);
    auto* pMeta = m_pUniverse->Deref<C_GcTable>(pTab->m_rMetatable);
    const TValue_t* pIndex = m_pUniverse->Meta().Lookup(pMeta, rt::EMetaMethod::Index);
    if (!pIndex || !pIndex->Is(EValueTag::Table)) return false;
    auto* pIndexTab = static_cast<C_GcTable*>(pIndex->AsGcPointer());
    PinValue(TValue_t::GcObject(EValueTag::Table, pMeta));
    PinValue(*pIndex);
    const TValue_t* pSlot = pIndexTab->Get(*m_pUniverse, tvKey);
    if (!pSlot || pSlot->IsNil()) return false;
    if (ObservedType(*pSlot) == EIrType::Nothing) return false;
    (void)EmitGuard(EIrOp::GuardEqI,
                    Emit(EIrOp::LoadU32, EIrType::Int, ConstantPtr(&pMeta->m_uVersion), kIrNone),
                    ConstantInt(pMeta->m_uVersion), pResumePc);
    (void)EmitGuard(EIrOp::GuardEqI,
                    Emit(EIrOp::LoadU32, EIrType::Int, ConstantPtr(&pIndexTab->m_uVersion),
                         kIrNone),
                    ConstantInt(pIndexTab->m_uVersion), pResumePc);
    // The __index binding itself is a VALUE at a (now pinned) slot address:
    // `mt.__index = other` is a plain store the structural version cannot see.
    (void)EmitGuard(EIrOp::GuardEq,
                    Emit(EIrOp::LoadTV, EIrType::Int, Materialize(ConstantPtr(pIndex)),
                         kIrNone),
                    Materialize(Constant(*pIndex)), pResumePc);
    if (m_eState != ETraceState::Recording) return false;
    const IrRef rVal = EmitSnapped(EIrOp::LoadTV, ObservedType(*pSlot),
                                   Materialize(ConstantPtr(pSlot)), kIrNone, pResumePc);
    if (rVal == kIrNone) return false;
    SetSlot(nB + ins.A(), rVal);
    return true;
}

bool C_TraceJit::RecordTableSet(const BcIns_t& ins, const BcIns_t* pNext, TValue_t tvKey,
                                std::int32_t nTabSlot) {
    const std::int32_t nB = m_nBaseOffset;
    const IrRef rTab = SlotRef(nTabSlot);
    if (TypeOf(rTab) != EIrType::Tab) return false;
    auto* pTab = static_cast<C_GcTable*>(m_pEntryBase[nTabSlot].AsGcPointer());
    const IrRef rValue = SlotRef(nB + ins.A());
    if (rValue == kIrNone) return false;
    const IrRef rTabPtr = Emit(EIrOp::TabPtr, EIrType::Ptr, rTab, kIrNone);
    const BcIns_t* pResumePc = pNext - 1;

    if (tvKey.IsDouble()) {
        const double flKey = tvKey.AsDouble();
        const auto nKey = static_cast<std::int64_t>(flKey);
        if (static_cast<double>(nKey) != flKey) return false;
        if (static_cast<std::uint64_t>(nKey) >= pTab->m_uArraySize) return false;
        IrRef rIdx;
        const std::uint32_t uKeySlot = ins.D() & 0xff;
        if (ins.Op() == EBcOp::TSetV) {
            const IrRef rKey = SlotRef(nB + static_cast<std::int32_t>(uKeySlot));
            if (TypeOf(rKey) != EIrType::Num) return false;
            rIdx = IsIntegralNum(rKey)
                       ? Emit(EIrOp::ToInt, EIrType::Int, rKey, kIrNone)
                       : EmitSnapped(EIrOp::ToInt, EIrType::Int, rKey, kIrNone, pResumePc);
            if (rIdx == kIrNone) return false;
        } else {
            rIdx = ConstantInt(nKey);
        }
        const IrRef rSize = Emit(EIrOp::LoadU32, EIrType::Int,
                                 Emit(EIrOp::AddK, EIrType::Ptr, rTabPtr,
                                      ConstantInt(kOfsTabAsize)),
                                 kIrNone);
        (void)EmitGuard(EIrOp::GuardBelow, Materialize(rIdx), rSize, pResumePc);
        const IrRef rArr = Emit(EIrOp::RefPtr, EIrType::Ptr,
                                Emit(EIrOp::LoadU32, EIrType::Int,
                                     Emit(EIrOp::AddK, EIrType::Ptr, rTabPtr,
                                          ConstantInt(kOfsTabArray)),
                                     kIrNone),
                                kIrNone);
        const IrRef rElem = Emit(EIrOp::IdxPtr, EIrType::Ptr, rArr, Materialize(rIdx));
        (void)Emit(EIrOp::StoreTV, TypeOf(rValue), rElem, rValue);
        return m_eState == ETraceState::Recording;
    }

    if (!tvKey.Is(EValueTag::String)) return false;
    IrRef rKeyRef = kIrNone;
    if (ins.Op() == EBcOp::TSetV) {
        rKeyRef = SlotRef(nB + static_cast<std::int32_t>(ins.D() & 0xff));
        if (rKeyRef == kIrNone) return false;
    }
    bool bAbsent = false;
    const IrRef rNode = HashNodeRef(rTabPtr, pTab, tvKey, rKeyRef, pResumePc, bAbsent);
    if (m_eState != ETraceState::Recording) return false;
    // Creating a key would resize and allocate — not something a trace does.
    if (bAbsent || rNode == kIrNone) return false;
    const TValue_t* pSlot = pTab->Get(*m_pUniverse, tvKey);
    if (!pSlot || pSlot->IsNil()) return false;
    // No version bump: the structural version does not see value stores, and
    // both the interpreter's caches and other traces re-read values through
    // slot addresses, so the store is visible to them by construction.
    (void)Emit(EIrOp::StoreTV, TypeOf(rValue), rNode, rValue);
    return m_eState == ETraceState::Recording;
}

// ---------------------------------------------------------------------------
// Calls: recording continues THROUGH a call whose callee is constant, so a
// trace spans many Lua frames and the call overhead disappears entirely.
// ---------------------------------------------------------------------------

// A C function is normally the end of a trace: it can allocate, it can call
// back into the interpreter, and the recorder cannot see inside it. The
// builtins below are the exceptions — pure, allocation-free, numbers in and
// numbers out — and each one is a single SSE instruction, so the trace emits
// the instruction instead of refusing the call.
bool C_TraceJit::RecordBuiltin(vm::EFastFunc eFfid, const BcIns_t& ins, const BcIns_t* pNext) {
    const std::int32_t nB = m_nBaseOffset;
    const std::int32_t nArg = nB + static_cast<std::int32_t>(ins.A()) + 2;
    const std::uint32_t uArgs = static_cast<std::uint32_t>(ins.C()) - 1u;
    const std::uint32_t uWant = ins.B();          // nresults + 1; 0 = all
    auto Unary = [&](EIrOp eOp, IrRef rMode) -> bool {
        if (uArgs != 1 || uWant != 2) return false;   // exactly one result wanted
        const IrRef rArg = SlotRef(nArg);
        if (TypeOf(rArg) != EIrType::Num) return false;
        const IrRef rRes = Emit(eOp, EIrType::Num, Materialize(rArg), rMode);
        if (rRes == kIrNone) return false;
        SetSlot(nB + static_cast<std::int32_t>(ins.A()), rRes);
        return true;
    };
    switch (eFfid) {
        case vm::EFastFunc::MathFloor: return Unary(EIrOp::Round, ConstantInt(0x09));
        case vm::EFastFunc::MathCeil:  return Unary(EIrOp::Round, ConstantInt(0x0a));
        case vm::EFastFunc::MathSqrt:  return Unary(EIrOp::Sqrt, kIrNone);
        case vm::EFastFunc::MathAbs:   return Unary(EIrOp::Abs, kIrNone);
        default: return false;
    }
    (void)pNext;
}

// The interpreter still dispatches the builtin's own FuncC header after the
// call; the recorder has already emitted the instruction that replaces it, so
// that header — and nothing else — is skipped.
bool C_TraceJit::SkipCFuncHeader() {
    if (!m_bPendingCFunc) return false;
    m_bPendingCFunc = false;
    return true;
}

// `for i, v in ipairs(t)` compiles to IterC (call the iterator) + IterL (the
// back edge). The iterator is a C function whose whole body is "bump the
// index, read the array slot, stop on nil" — recorded here as the array access
// it is, with the bound guard doubling as the loop-exit guard.
bool C_TraceJit::RecordIPairsIter(const BcIns_t& ins, const BcIns_t* pPc) {
    const std::int32_t nB = m_nBaseOffset;
    const std::int32_t nA = nB + static_cast<std::int32_t>(ins.A());
    const IrRef rTab = SlotRef(nA - 2);
    const IrRef rIdx = SlotRef(nA - 1);
    if (TypeOf(rTab) != EIrType::Tab || TypeOf(rIdx) != EIrType::Num) return false;
    auto* pTab = static_cast<C_GcTable*>(m_pEntryBase[nA - 2].AsGcPointer());
    const double flNext = m_pEntryBase[nA - 1].AsDouble() + 1.0;
    const auto nNext = static_cast<std::int64_t>(flNext);
    if (static_cast<double>(nNext) != flNext) return false;
    if (static_cast<std::uint64_t>(nNext) >= pTab->m_uArraySize) return false;

    if (IsInt32Value(m_pEntryBase[nA - 1].AsDouble()))
        (void)GuardEntryInt32(rIdx, pPc);   // control counter: narrow it
    if (m_eState != ETraceState::Recording) return false;
    const IrRef rNext = Emit(EIrOp::Add, EIrType::Num, rIdx, Materialize(ConstantNum(1.0)));
    if (IsIntegralNum(rIdx)) MarkIntegral(rNext);
    IrRef rInt;
    if (IsIntegralNum(rNext)) {
        rInt = Emit(EIrOp::ToInt, EIrType::Int, rNext, kIrNone);
    } else {
        rInt = EmitSnapped(EIrOp::ToInt, EIrType::Int, rNext, kIrNone, pPc);
    }
    if (rInt == kIrNone) return false;
    const IrRef rTabPtr = Emit(EIrOp::TabPtr, EIrType::Ptr, rTab, kIrNone);
    const IrRef rSize = Emit(EIrOp::LoadU32, EIrType::Int,
                             Emit(EIrOp::AddK, EIrType::Ptr, rTabPtr, ConstantInt(kOfsTabAsize)),
                             kIrNone);
    // Falling off the end of the array part is the loop's exit: the guard
    // resumes the interpreter at this very IterC, which runs the real iterator.
    (void)EmitGuard(EIrOp::GuardBelow, rInt, rSize, pPc);
    const IrRef rArr = Emit(EIrOp::RefPtr, EIrType::Ptr,
                            Emit(EIrOp::LoadU32, EIrType::Int,
                                 Emit(EIrOp::AddK, EIrType::Ptr, rTabPtr,
                                      ConstantInt(kOfsTabArray)),
                                 kIrNone),
                            kIrNone);
    const TValue_t* pElem =
        static_cast<TValue_t*>(core::RefToPtr(m_pUniverse->ArenaBase(), pTab->m_rArray)) + nNext;
    const EIrType eType = ObservedType(*pElem);
    // A nil element also ends the iteration; the type guard is what catches it.
    if (eType == EIrType::Nothing || eType == EIrType::Nil) return false;
    const IrRef rElem = EmitSnapped(EIrOp::LoadTV, eType,
                                    Emit(EIrOp::IdxPtr, EIrType::Ptr, rArr, rInt), kIrNone,
                                    pPc);
    if (rElem == kIrNone) return false;
    SetSlot(nA, rNext);
    SetSlot(nA + 1, rElem);
    return m_eState == ETraceState::Recording;
}

bool C_TraceJit::RecordCall(const BcIns_t& ins, const BcIns_t* pNext) {
    if (m_vFrames.size() >= kMaxInlineDepth) return false;
    const std::int32_t nB = m_nBaseOffset;
    const std::uint32_t uA = ins.A();
    const std::uint32_t uArgs = static_cast<std::uint32_t>(ins.C()) - 1u;
    const TValue_t tvFunc = m_pBase[uA];
    if (!tvFunc.Is(EValueTag::Function)) return false;
    auto* pFn = static_cast<C_GcFunction*>(tvFunc.AsGcPointer());
    if (!pFn->IsLua()) {
        const auto eFfid = static_cast<vm::EFastFunc>(pFn->m_Header.uExtra1);
        if (eFfid == vm::EFastFunc::C) return false;
        const IrRef rFn = SlotRef(nB + static_cast<std::int32_t>(uA));
        if (rFn == kIrNone) return false;
        if (!IsConstRef(rFn))
            (void)EmitGuard(EIrOp::GuardEq, rFn, Materialize(Constant(tvFunc)), pNext - 1);
        if (m_eState != ETraceState::Recording) return false;
        if (!RecordBuiltin(eFfid, ins, pNext)) return false;
        m_bPendingCFunc = true;
        return true;
    }
    const auto* pProto = C_GcProto::FromBytecode(pFn->m_pPc);
    if (pProto->m_uFlags & static_cast<std::uint8_t>(vm::EProtoFlag::IsVararg)) return false;
    if (uArgs != pProto->ParamCount()) return false;

    const IrRef rFunc = SlotRef(nB + static_cast<std::int32_t>(uA));
    if (rFunc == kIrNone) return false;
    if (!IsConstRef(rFunc))
        (void)EmitGuard(EIrOp::GuardEq, rFunc, Constant(tvFunc), pNext - 1);
    if (m_eState != ETraceState::Recording) return false;

    // The frame link is a constant: the return PC is fixed for this call site.
    SetSlot(nB + static_cast<std::int32_t>(uA) + 1,
            Constant(TValue_t{vm::FrameLink_t::FromReturnPc(pNext).uRaw}));
    m_vFrames.push_back(InlineFrame_t{nB, pNext, m_pKBase});
    return true;
}

bool C_TraceJit::RecordReturn(const BcIns_t&, std::uint32_t uFirst, std::uint32_t uCount) {
    if (m_vFrames.empty()) {
        AbortRecording(EAbort::LeftFrame, "returned out of the trace's entry frame");
        return false;
    }
    const InlineFrame_t frame = m_vFrames.back();
    m_vFrames.pop_back();
    const std::int32_t nB = m_nBaseOffset;
    const BcIns_t insCall{frame.pReturnPc[-1].uRaw};
    const std::uint32_t uExpected = insCall.B();   // nresults+1; 0 = all
    if (uExpected == 0) return false;              // multi-result: not on a trace
    for (std::uint32_t uI = 0; uI < uCount; ++uI) {
        const IrRef r = SlotRef(nB + static_cast<std::int32_t>(uFirst + uI));
        if (r == kIrNone) return false;
        SetSlot(nB - 2 + static_cast<std::int32_t>(uI), r);
    }
    const IrRef rNil = Constant(TValue_t::Nil());
    for (std::uint32_t uI = uCount; uI + 1 < uExpected; ++uI)
        SetSlot(nB - 2 + static_cast<std::int32_t>(uI), rNil);
    m_nBaseOffset = frame.nBaseOffset;
    m_pKBase = frame.pKBase;
    return m_eState == ETraceState::Recording;
}

}  // namespace ljx::jit
