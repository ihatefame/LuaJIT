// LJX — trace recorder: executed bytecodes → typed SSA IR.
//
// The recorder runs alongside the interpreter (the dispatch table is swapped
// so every instruction passes through RecordAndExecute first). It sees the
// live operand values, so it can specialize on the types actually present and
// emit a guard rather than a test.
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "ljx/jit/TraceJit.hpp"
#include "ljx/rt/Meta.hpp"
#include "ljx/vm/Interpreter.hpp"
#include "ljx/vm/Object.hpp"

namespace ljx::jit {

using vm::BcIns_t;
using vm::EBcOp;
using vm::EValueTag;
using vm::TValue_t;

namespace {

bool TraceDebug() {
    static const bool bOn = std::getenv("LJX_TRACEDEBUG") != nullptr;
    return bOn;
}

// Slot indices are relative to the trace's entry base and may reach a little
// below it (a frame's function and link slots live at base-2 and base-1).
constexpr std::int32_t kSlotBias = 4;

const char* kIrOpNames[] = {
#define LJX_IR_NAME(name) #name,
    LJX_IR_OPS(LJX_IR_NAME)
#undef LJX_IR_NAME
};

}  // namespace

const char* IrOpName(EIrOp eOp) noexcept {
    const auto uIdx = static_cast<std::uint32_t>(eOp);
    return uIdx < static_cast<std::uint32_t>(EIrOp::Count_) ? kIrOpNames[uIdx] : "?";
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

IrRef C_TraceJit::Emit(EIrOp eOp, EIrType eType, IrRef rOp1, IrRef rOp2) {
    // Common-subexpression elimination over this opcode's chain. Operands are
    // always defined before their use, so the walk is bounded by the newer of
    // the two operand refs — anything older cannot match.
    const IrRef rLimit = rOp1 > rOp2 ? rOp1 : rOp2;
    const bool bPure = eOp != EIrOp::AStore && eOp != EIrOp::HStore &&
                       eOp != EIrOp::SLoad && eOp != EIrOp::Loop &&
                       eOp != EIrOp::Phi && eOp != EIrOp::ALoad &&
                       eOp != EIrOp::HLoad;
    if (bPure) {
        for (IrRef r = m_vChain[static_cast<std::size_t>(eOp)];
             r != kIrNone && r > rLimit;) {
            const IrIns_t& ins = m_vIns[r - kIrBias];
            if (ins.rOp1 == rOp1 && ins.rOp2 == rOp2 && ins.eType == eType) return r;
            r = ins.rPrev;
        }
    }
    if (m_vIns.size() + kIrBias >= 0xfff0 || m_vIns.size() >= kMaxIrIns) {
        AbortRecording(EAbort::TooLong, "IR overflow");
        return kIrNone;
    }
    const auto rNew = static_cast<IrRef>(kIrBias + m_vIns.size());
    m_vIns.push_back(IrIns_t{rOp1, rOp2, eOp, eType,
                             m_vChain[static_cast<std::size_t>(eOp)]});
    m_vChain[static_cast<std::size_t>(eOp)] = rNew;
    return rNew;
}

IrRef C_TraceJit::Constant(const TValue_t& tvValue) {
    const EIrType eType = ObservedType(tvValue);
    for (std::size_t uI = 0; uI < m_vConst.size(); ++uI)
        if (m_vConst[uI].uValue == tvValue.uRaw && m_vConst[uI].eType == eType)
            return static_cast<IrRef>(kIrBias - 1 - uI);
    m_vConst.push_back(IrConst_t{tvValue.uRaw, eType});
    return static_cast<IrRef>(kIrBias - m_vConst.size());
}

IrRef C_TraceJit::ConstantNum(double flValue) {
    return Constant(TValue_t::Number(flValue));
}

// ---------------------------------------------------------------------------
// Slot map: a stack slot's current IR value, loaded on demand with a guard.
// ---------------------------------------------------------------------------

IrRef C_TraceJit::SlotRef(std::int32_t nSlot) {
    const auto uIdx = static_cast<std::size_t>(nSlot + kSlotBias);
    if (uIdx >= m_vSlotValue.size()) {
        AbortRecording(EAbort::Unsupported, "slot out of range");
        return kIrNone;
    }
    if (m_vSlotValue[uIdx] != kIrNone) return m_vSlotValue[uIdx];
    // First read of this slot on the trace: emit the load together with the
    // guard that it still holds the type we observed while recording.
    const TValue_t tvLive = m_pEntryBase[nSlot];
    const EIrType eType = ObservedType(tvLive);
    if (eType == EIrType::Nothing) {
        AbortRecording(EAbort::BadType, "unsupported value type in slot");
        return kIrNone;
    }
    const IrRef rSlotConst = Constant(TValue_t::Integer(nSlot));
    const IrRef rLoad = Emit(EIrOp::SLoad, eType, rSlotConst, kIrNone);
    m_vSlotValue[uIdx] = rLoad;
    m_vSlotType[uIdx] = eType;
    return rLoad;
}

void C_TraceJit::SetSlot(std::int32_t nSlot, IrRef rValue) {
    const auto uIdx = static_cast<std::size_t>(nSlot + kSlotBias);
    if (uIdx >= m_vSlotValue.size()) {
        AbortRecording(EAbort::Unsupported, "slot out of range");
        return;
    }
    m_vSlotValue[uIdx] = rValue;
    m_vSlotType[uIdx] = rValue == kIrNone
                            ? EIrType::Nothing
                            : (IsConstRef(rValue) ? m_vConst[kIrBias - 1 - rValue].eType
                                                  : m_vIns[rValue - kIrBias].eType);
}

std::uint32_t C_TraceJit::TakeSnapshot(std::uint32_t uResumePc) {
    Snapshot_t snap;
    snap.uFirstSlot = static_cast<std::uint32_t>(m_vSnapSlots.size());
    snap.uResumePc = uResumePc;
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

void C_TraceJit::StartRecording(const BcIns_t* pPc, TValue_t* pBase) {
    m_eState = ETraceState::Recording;
    m_pStartPc = pPc;
    m_pEntryBase = pBase;
    m_uRecorded = 0;
    m_nBaseOffset = 0;
    m_vIns.clear();
    m_vConst.clear();
    m_vSnapshots.clear();
    m_vSnapSlots.clear();
    m_vFrames.clear();
    std::memset(m_vChain, 0, sizeof m_vChain);
    m_vSlotValue.assign(kMaxSlots + kSlotBias, kIrNone);
    m_vSlotType.assign(kMaxSlots + kSlotBias, EIrType::Nothing);
    if (TraceDebug())
        std::fprintf(stderr, "[trace] start @%p\n", static_cast<const void*>(pPc));
}

void C_TraceJit::AbortRecording(EAbort eReason, const char* sDetail) {
    if (m_eState != ETraceState::Recording) return;
    m_eState = ETraceState::Idle;
    // Penalize the site so a loop that cannot be recorded is not retried on
    // every entry; a large enough count blacklists it for good.
    if (m_pStartPc) m_mapHot[m_pStartPc] = 0x40000000u;
    if (TraceDebug())
        std::fprintf(stderr, "[trace] abort (%u): %s\n",
                     static_cast<unsigned>(eReason), sDetail);
    vm::C_Interpreter::SetRecordMode(*m_pUniverse, false);
}

}  // namespace ljx::jit
