// LJX — trace recorder: executed bytecodes → typed SSA IR.
//
// The recorder runs alongside the interpreter (the dispatch table is swapped
// so every instruction passes through RecordAndExecute first). It sees the
// live operand values, so it can specialize on the types actually present and
// emit a guard rather than a test — and it can resolve a metatable lookup at
// record time and keep only the guards that make the answer a constant.
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include <sys/mman.h>

#include "ljx/gc/GarbageCollector.hpp"
#include "ljx/jit/TraceJit.hpp"
#include "ljx/rt/Meta.hpp"
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
            for (std::size_t uE = 0; uE < pTrace->vExitCounts.size(); ++uE)
                if (pTrace->vExitCounts[uE])
                    std::fprintf(stderr, "[trace]   exit %zu: %llu\n", uE,
                                 static_cast<unsigned long long>(pTrace->vExitCounts[uE]));
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

// A compiled trace outlives any Lua reference to the objects it specialized
// on: it compares against their addresses and, for the metatable fold, reads
// their version words directly. Anchoring them in the registry — an ordinary
// GC root — keeps those addresses valid for the trace's lifetime without
// teaching the collector anything about traces.
void C_TraceJit::PinValue(const TValue_t& tvValue) {
    if (!tvValue.IsGcObject()) return;
    *m_pUniverse->Registry()->Set(*m_pUniverse, tvValue) = TValue_t::Boolean(true);
    m_pUniverse->Registry()->BumpVersion();
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
    if (m_vSlotEntry[uIdx] == kIrNone)
        (void)Emit(EIrOp::SStore, TypeOf(rValue), static_cast<IrRef>(uIdx), rValue);
}

std::uint32_t C_TraceJit::TakeSnapshot(const BcIns_t* pResumePc) {
    Snapshot_t snap;
    snap.uFirstSlot = static_cast<std::uint32_t>(m_vSnapSlots.size());
    snap.pResumePc = pResumePc;
    snap.nBaseOffset = m_nBaseOffset;
    for (std::size_t uI = 0; uI < m_vSlotValue.size(); ++uI) {
        if (m_vSlotValue[uI] == kIrNone || m_vSlotEntry[uI] == kIrNone) continue;
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
    m_nBaseOffset = 0;
    m_nTopSlot = 0;
    m_vIns.clear();
    m_vInsSnap.clear();
    m_vConst.clear();
    m_vSnapshots.clear();
    m_vSnapSlots.clear();
    m_vFrames.clear();
    std::memset(m_vChain, 0, sizeof m_vChain);
    m_vSlotValue.assign(static_cast<std::size_t>(kMaxSlot + kSlotBias), kIrNone);
    m_vSlotEntry.assign(static_cast<std::size_t>(kMaxSlot + kSlotBias), kIrNone);
    // Exit 0 is the entry guard: nothing live, resume at the trace head.
    (void)TakeSnapshot(pPc);
    vm::C_Interpreter::SetRecordMode(*m_pUniverse, true);
    if (TraceDebug())
        std::fprintf(stderr, "[trace] start @%p\n", static_cast<const void*>(pPc));
}

void C_TraceJit::AbortRecording(EAbort eReason, const char* sDetail) {
    if (m_eState != ETraceState::Recording) return;
    m_eState = ETraceState::Idle;
    // Penalize the site so a loop that cannot be recorded is not retried on
    // every entry; the penalty blacklists it for good.
    if (m_pStartPc) m_mapHot[m_pStartPc] = kBlacklistCount;
    if (TraceDebug())
        std::fprintf(stderr, "[trace] abort (%u): %s\n",
                     static_cast<unsigned>(eReason), sDetail);
    vm::C_Interpreter::SetRecordMode(*m_pUniverse, false);
}

void C_TraceJit::CloseLoop() {
    if (!m_vFrames.empty() || m_nBaseOffset != 0) {
        AbortRecording(EAbort::LeftFrame, "loop closed in an inlined frame");
        return;
    }
    (void)Emit(EIrOp::Loop, EIrType::Nothing, kIrNone, kIrNone);
    Trace_t* pTrace = Assemble();
    m_eState = ETraceState::Idle;
    vm::C_Interpreter::SetRecordMode(*m_pUniverse, false);
    if (!pTrace) {
        if (m_pStartPc) m_mapHot[m_pStartPc] = kBlacklistCount;
        return;
    }
    pTrace->pStartPc = m_pStartPc;
    pTrace->nTopSlot = m_nTopSlot;
    pTrace->uNumber = static_cast<std::uint32_t>(m_vTraces.size());
    m_vTraces.push_back(pTrace);
    m_mapTraces[m_pStartPc] = pTrace;
    if (TraceDebug())
        std::fprintf(stderr, "[trace] #%u compiled: %zu ins, %zu exits\n",
                     pTrace->uNumber, m_vIns.size(), pTrace->vExits.size());
}

const Trace_t* C_TraceJit::OnLoopEdge(const BcIns_t* pHeadPc, TValue_t* pBase,
                                      const TValue_t* pKBase) {
    // LJX_NOJIT=1 forces everything through the interpreter — the reference
    // semantics the differential test compares compiled output against.
    static const bool bDisabled = std::getenv("LJX_NOJIT") != nullptr;
    if (bDisabled || m_eState == ETraceState::Recording) return nullptr;
    if (auto it = m_mapTraces.find(pHeadPc); it != m_mapTraces.end()) return it->second;
    std::uint32_t& uCount = m_mapHot[pHeadPc];
    if (uCount >= kBlacklistCount) return nullptr;
    if (++uCount < kHotLoopThreshold) return nullptr;
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
                                Constant(TValue_t::GcObject(EValueTag::Function, pFn)),
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
            const IrRef rVal = Emit(EIrOp::LoadTV, eType, ConstantPtr(pCell), kIrNone);
            if (rVal == kIrNone) return false;
            m_vInsSnap[rVal - kIrBias] = static_cast<std::uint16_t>(TakeSnapshot(pNext - 1));
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
            const IrRef rVal = Emit(EIrOp::LoadTV, eType, rNode, kIrNone);
            if (rVal == kIrNone) return false;
            m_vInsSnap[rVal - kIrBias] = static_cast<std::uint16_t>(TakeSnapshot(pNext - 1));
            SetSlot(nB + uA, rVal);
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
    const IrRef rRes = Emit(eOp, EIrType::Num, rLeft, rRight);
    if (rRes == kIrNone) return false;
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
        (void)EmitGuard(eGuard, rLeft, rRight, pExitPc);
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
    // NaN would make the raw compare disagree with Lua equality; a guarded
    // number that is NaN is vanishingly rare, so reject it outright.
    if (tvA.IsDouble() && tvA.AsDouble() != tvA.AsDouble()) return false;
    const bool bEqual = tvA.uRaw == tvB.uRaw;
    const bool bIsEq = eOp == EBcOp::IsEqV || eOp == EBcOp::IsEqS ||
                       eOp == EBcOp::IsEqN || eOp == EBcOp::IsEqP;
    bTaken = bIsEq ? bEqual : !bEqual;
    const BcIns_t* pExitPc = bTaken ? pNext + 1 : pNext + 1 + insJmp.JumpTarget();
    (void)EmitGuard(bEqual ? EIrOp::GuardEq : EIrOp::GuardNe, rLeft, rRight, pExitPc);
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
    const IrRef rNew = Emit(EIrOp::Add, EIrType::Num, rIdx, rStep);
    if (rNew == kIrNone) return false;
    // The loop continues on the recorded path; the guard's exit resumes at the
    // instruction after the back edge, i.e. the loop's exit.
    // The interpreter stores the new index unconditionally and the visible
    // variable only when the loop continues, so the index update must be
    // visible to the guard's snapshot and the copy must not be.
    SetSlot(nSlot, rNew);
    (void)EmitGuard(flStep >= 0 ? EIrOp::GuardLe : EIrOp::GuardGe, rNew, rStop, pNext);
    if (m_eState != ETraceState::Recording) return false;
    SetSlot(nSlot + 3, rNew);
    return true;
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
    const IrRef rMain = Emit(EIrOp::AndInt, EIrType::Int, rSid, rHmask);
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
            (void)EmitGuard(EIrOp::GuardEq, rNodeKey, rKeyTagged, pResumePc);
            return m_eState == ETraceState::Recording ? rNode : kIrNone;
        }
        (void)EmitGuard(EIrOp::GuardNe, rNodeKey, rKeyTagged, pResumePc);
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
            rIdx = Emit(EIrOp::ToInt, EIrType::Int, rKeyDynamic, kIrNone);
            if (rIdx == kIrNone) return false;
            m_vInsSnap[rIdx - kIrBias] = static_cast<std::uint16_t>(TakeSnapshot(pResumePc));
        } else {
            rIdx = ConstantInt(nKey);
        }
        const IrRef rSize = Emit(EIrOp::LoadU32, EIrType::Int,
                                 Emit(EIrOp::AddK, EIrType::Ptr, rTabPtr,
                                      ConstantInt(kOfsTabAsize)),
                                 kIrNone);
        (void)EmitGuard(EIrOp::GuardBelow, rIdx, rSize, pResumePc);
        const IrRef rArr = Emit(EIrOp::RefPtr, EIrType::Ptr,
                                Emit(EIrOp::LoadU32, EIrType::Int,
                                     Emit(EIrOp::AddK, EIrType::Ptr, rTabPtr,
                                          ConstantInt(kOfsTabArray)),
                                     kIrNone),
                                kIrNone);
        const IrRef rElem = Emit(EIrOp::IdxPtr, EIrType::Ptr, rArr, rIdx);
        const TValue_t* pElem =
            static_cast<TValue_t*>(core::RefToPtr(m_pUniverse->ArenaBase(), pTab->m_rArray)) + nKey;
        const EIrType eType = ObservedType(*pElem);
        if (eType == EIrType::Nothing) return false;
        const IrRef rVal = Emit(EIrOp::LoadTV, eType, rElem, kIrNone);
        if (rVal == kIrNone) return false;
        m_vInsSnap[rVal - kIrBias] = static_cast<std::uint16_t>(TakeSnapshot(pResumePc));
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
        const IrRef rVal = Emit(EIrOp::LoadTV, eType, rNode, kIrNone);
        if (rVal == kIrNone) return false;
        m_vInsSnap[rVal - kIrBias] = static_cast<std::uint16_t>(TakeSnapshot(pResumePc));
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
    // versions of the two tables the chain went through; under those guards
    // the resolved value is a compile-time CONSTANT, which is what makes the
    // following call site monomorphic and inlinable.
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
    if (m_eState != ETraceState::Recording) return false;
    SetSlot(nB + ins.A(), Constant(*pSlot));
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
            rIdx = Emit(EIrOp::ToInt, EIrType::Int, rKey, kIrNone);
            if (rIdx == kIrNone) return false;
            m_vInsSnap[rIdx - kIrBias] = static_cast<std::uint16_t>(TakeSnapshot(pResumePc));
        } else {
            rIdx = ConstantInt(nKey);
        }
        const IrRef rSize = Emit(EIrOp::LoadU32, EIrType::Int,
                                 Emit(EIrOp::AddK, EIrType::Ptr, rTabPtr,
                                      ConstantInt(kOfsTabAsize)),
                                 kIrNone);
        (void)EmitGuard(EIrOp::GuardBelow, rIdx, rSize, pResumePc);
        const IrRef rArr = Emit(EIrOp::RefPtr, EIrType::Ptr,
                                Emit(EIrOp::LoadU32, EIrType::Int,
                                     Emit(EIrOp::AddK, EIrType::Ptr, rTabPtr,
                                          ConstantInt(kOfsTabArray)),
                                     kIrNone),
                                kIrNone);
        const IrRef rElem = Emit(EIrOp::IdxPtr, EIrType::Ptr, rArr, rIdx);
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
    (void)Emit(EIrOp::StoreTV, TypeOf(rValue), rNode, rValue);
    // An interpreter inline cache may have resolved through this table, so the
    // store must invalidate it exactly as C_GcTable::BumpVersion would.
    (void)Emit(EIrOp::IncU32, EIrType::Nothing,
               Emit(EIrOp::AddK, EIrType::Ptr, rTabPtr, ConstantInt(kOfsTabVersion)), kIrNone);
    return m_eState == ETraceState::Recording;
}

// ---------------------------------------------------------------------------
// Calls: recording continues THROUGH a call whose callee is constant, so a
// trace spans many Lua frames and the call overhead disappears entirely.
// ---------------------------------------------------------------------------

bool C_TraceJit::RecordCall(const BcIns_t& ins, const BcIns_t* pNext) {
    if (m_vFrames.size() >= kMaxInlineDepth) return false;
    const std::int32_t nB = m_nBaseOffset;
    const std::uint32_t uA = ins.A();
    const std::uint32_t uArgs = static_cast<std::uint32_t>(ins.C()) - 1u;
    const TValue_t tvFunc = m_pBase[uA];
    if (!tvFunc.Is(EValueTag::Function)) return false;
    auto* pFn = static_cast<C_GcFunction*>(tvFunc.AsGcPointer());
    if (!pFn->IsLua()) return false;   // C functions can allocate and re-enter
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
