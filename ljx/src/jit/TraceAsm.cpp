// LJX — trace backend: typed SSA IR → x86-64.
//
// Layout of a compiled trace:
//
//     prologue                 save callee-saved regs, pin pBase / arena base
//     preamble                 every SLoad + its entry type guard, once
//   loop_top:
//     body                     the recorded iteration, fully register-resident
//     back-edge copies         loop-carried values back into their entry regs
//     jmp loop_top
//   exit stubs                 write the snapshot back to the Lua stack
//     epilogue                 restore and return the exit number
//
// Because SLoads are hoisted into the preamble and the back edge restores
// their registers, one iteration executes with no dispatch, no type test that
// was already proven, and no memory traffic for values that stay in registers.
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <sys/mman.h>

#include "ljx/jit/TraceJit.hpp"
#include "ljx/vm/Object.hpp"
#include "X64Emitter.hpp"

namespace ljx::jit {

namespace {

// Pinned registers.
constexpr std::uint8_t kRegBase = 3;    // rbx — Lua stack base of the entry frame
constexpr std::uint8_t kRegArena = 5;   // rbp — GC arena base (compressed refs)
// Scratch, never allocated.
constexpr std::uint8_t kGprScratch[2] = {10, 11};
constexpr std::uint8_t kXmmScratch[2] = {13, 14};
constexpr std::uint8_t kXmmTemp = 15;

constexpr std::uint8_t kAllocGpr[] = {0, 1, 2, 6, 7, 8, 9, 12, 13, 14, 15};
constexpr std::uint8_t kAllocXmm[] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
constexpr std::uint32_t kNumGpr = sizeof kAllocGpr / sizeof kAllocGpr[0];
constexpr std::uint32_t kNumXmm = sizeof kAllocXmm / sizeof kAllocXmm[0];

constexpr std::uint8_t kNoReg = 0xff;
constexpr std::uint32_t kNoPos = 0xffff'ffffu;

// Tag value a guard compares against, per IR type.
std::uint32_t TagFor(EIrType eType) noexcept {
    switch (eType) {
        case EIrType::Nil: return static_cast<std::uint32_t>(vm::EValueTag::Nil);
        case EIrType::False: return static_cast<std::uint32_t>(vm::EValueTag::False);
        case EIrType::True: return static_cast<std::uint32_t>(vm::EValueTag::True);
        case EIrType::Str: return static_cast<std::uint32_t>(vm::EValueTag::String);
        case EIrType::Tab: return static_cast<std::uint32_t>(vm::EValueTag::Table);
        case EIrType::Func: return static_cast<std::uint32_t>(vm::EValueTag::Function);
        default: return 0;
    }
}

// Where a value lives at a given program point, captured for an exit stub.
struct ExitLoc_t {
    std::int32_t nSlot;
    std::uint8_t uReg;      // kNoReg when the value is a constant or spilled
    bool bFloat;
    bool bSpilled;
    std::uint32_t uSpillIdx;
    std::uint64_t uConst;   // valid when uReg == kNoReg && !bSpilled
    bool bConst;
};

struct PendingExit_t {
    std::vector<std::size_t> vPatches;
    std::vector<ExitLoc_t> vLocs;
    const vm::BcIns_t* pResumePc;
    std::int32_t nBaseOffset;
    std::size_t uIrIdx = 0;
};

class C_TraceAsm {
public:
    C_TraceAsm(const std::vector<IrIns_t>& vIns, const std::vector<std::uint16_t>& vInsSnap,
               const std::vector<IrConst_t>& vConst, const std::vector<Snapshot_t>& vSnap,
               const std::vector<SnapSlot_t>& vSnapSlots, const std::vector<IrRef>& vSlotValue,
               const std::vector<IrRef>& vSlotEntry, const std::vector<std::uint8_t>& vForce,
               std::int32_t nSlotBias) noexcept
        : m_vIns(vIns), m_vInsSnap(vInsSnap), m_vConst(vConst), m_vSnap(vSnap),
          m_vSnapSlots(vSnapSlots), m_vSlotValue(vSlotValue), m_vSlotEntry(vSlotEntry),
          m_vForceHoist(vForce), m_nSlotBias(nSlotBias) {}

    [[nodiscard]] bool Run();

private:
    [[nodiscard]] bool RunOnce();

public:

    X64Emitter m_Emit;
    std::vector<PendingExit_t> m_vExits;

private:
    // --- helpers ------------------------------------------------------------
    [[nodiscard]] EIrType TypeOf(IrRef r) const noexcept {
        return IsConstRef(r) ? m_vConst[kIrBias - 1 - r].eType : m_vIns[r - kIrBias].eType;
    }
    [[nodiscard]] std::uint64_t ConstOf(IrRef r) const noexcept {
        return m_vConst[kIrBias - 1 - r].uValue;
    }
    [[nodiscard]] std::int32_t SlotDisp(IrRef rRaw) const noexcept {
        return (static_cast<std::int32_t>(rRaw) - m_nSlotBias) * 8;
    }
    [[nodiscard]] std::int32_t SpillDisp(std::size_t uIdx) const noexcept {
        return static_cast<std::int32_t>(uIdx * 8);
    }

    void ComputeInvariance();
    void ComputeLiveness();
    [[nodiscard]] bool Allocate(std::size_t uIdx, bool bFloat, std::uint8_t& uOut);
    [[nodiscard]] std::uint8_t OperandGpr(IrRef r, int nScratch);
    [[nodiscard]] std::uint8_t OperandXmm(IrRef r, int nScratch);
    void MaterializeConstGpr(std::uint8_t uReg, std::uint64_t u);
    void MaterializeConstXmm(std::uint8_t uXmm, std::uint64_t u);
    void FreeDead(std::uint32_t uPos);
    void Pin(std::uint8_t uReg, bool bFloat);
    void UnpinAll();
    [[nodiscard]] std::uint32_t BeginExit(std::uint16_t uSnap);
    void AddPatch(std::uint32_t uExit, std::size_t uPatch);
    [[nodiscard]] bool EmitOne(std::size_t uIdx);
    void EmitBackEdge();

    const std::vector<IrIns_t>& m_vIns;
    const std::vector<std::uint16_t>& m_vInsSnap;
    const std::vector<IrConst_t>& m_vConst;
    const std::vector<Snapshot_t>& m_vSnap;
    const std::vector<SnapSlot_t>& m_vSnapSlots;
    const std::vector<IrRef>& m_vSlotValue;
    const std::vector<IrRef>& m_vSlotEntry;
    const std::vector<std::uint8_t>& m_vForceHoist;
    std::int32_t m_nSlotBias;

    std::vector<std::size_t> m_vOrder;      // emission order (SLoads hoisted)
    std::vector<std::uint32_t> m_vLastUse;  // per IR index, position in m_vOrder
    std::vector<std::uint8_t> m_vReg;       // per IR index
    std::vector<std::uint8_t> m_vSpilled;
    std::vector<std::uint8_t> m_vNoSpill;
    std::vector<std::uint8_t> m_vInvariant;
    // Pre-roll values that must live in memory for the whole loop (see Run()).
    std::vector<std::uint8_t> m_vMemOnly;
    bool m_bNewMemOnly = false;
    std::uint32_t m_vGprOwner[16]{};        // IR index + 1, 0 = free
    std::uint32_t m_vXmmOwner[16]{};
    std::uint32_t m_uGprPinned = 0;
    std::uint32_t m_uXmmPinned = 0;
    std::uint32_t m_uPos = 0;
    std::size_t m_uCurIdx = 0;
    bool m_bFailed = false;
    bool m_bLooping = false;
    std::size_t m_uLoopTop = 0;
    std::size_t m_uHoisted = 0;
};

// Loop-invariant code motion.
//
// Everything a field access costs except the value load itself is invariant
// when the receiver is: the mask and node-array loads, the main-position
// arithmetic, the node addresses, and the node KEY loads with their guards
// (a raw store rewrites a node's value word, never its key). Hoisting them
// leaves a loop body that is essentially load, guard the type, compute, store
// — which is the shape LuaJIT's hoisted pre-roll produces.
//
// Guards that move into the pre-roll are re-pointed at snapshot 0, the entry
// exit: nothing has been modified when they run, so resuming at the trace head
// with the interpreter's own state is always correct.
void C_TraceAsm::ComputeInvariance() {
    const std::size_t uCount = m_vIns.size();
    m_vInvariant.assign(uCount, 0);
    // A stack slot is loop-invariant exactly when the recorded iteration left
    // it holding the value it was entered with.
    std::vector<std::uint8_t> vSlotFixed(m_vSlotValue.size(), 0);
    for (std::size_t uS = 0; uS < m_vSlotValue.size(); ++uS)
        vSlotFixed[uS] = m_vSlotEntry[uS] != kIrNone && m_vSlotValue[uS] == m_vSlotEntry[uS];

    auto Inv = [&](IrRef r) {
        return r == kIrNone || IsConstRef(r) || m_vInvariant[r - kIrBias] != 0;
    };
    // Header words a trace can never write: array/metatable/next, node array,
    // array size, hash mask. The version word (and anything else) is excluded.
    auto IsStableField = [&](IrRef rAddr) {
        if (rAddr == kIrNone || IsConstRef(rAddr)) return false;
        const IrIns_t& addr = m_vIns[rAddr - kIrBias];
        if (addr.eOp != EIrOp::AddK) return false;
        const std::uint64_t uOfs = ConstOf(addr.rOp2);
        return uOfs == 8 || uOfs == 16 || uOfs == 20 || uOfs == 24 || uOfs == 28;
    };

    for (std::size_t uI = 0; uI < uCount; ++uI) {
        const IrIns_t& ins = m_vIns[uI];
        // Entry-state guards (ChkInt32 narrowing checks) belong to the
        // preamble by construction: they guard trace-ENTRY values, so running
        // them once and exiting through snapshot 0 is exactly their meaning.
        if (m_vForceHoist[uI]) {
            m_vInvariant[uI] = 1;
            continue;
        }
        bool bInv = false;
        switch (ins.eOp) {
            case EIrOp::SLoad:
                bInv = ins.rOp1 < vSlotFixed.size() && vSlotFixed[ins.rOp1] != 0;
                break;
            case EIrOp::Nop: case EIrOp::SStore: case EIrOp::StoreTV:
            case EIrOp::IncU32: case EIrOp::Loop:
            // End is unconditional control flow, not a guard: hoisting it
            // would leave the trace through snapshot 0 before the body ran.
            case EIrOp::End:
                break;
            case EIrOp::LoadU32:
                bInv = IsStableField(ins.rOp1) && Inv(ins.rOp1);
                break;
            case EIrOp::LoadTV:
                // Only the untyped form — that is the node key word, which a
                // raw store never touches. Value loads stay in the loop.
                bInv = ins.eType == EIrType::Int && Inv(ins.rOp1);
                break;
            default:
                bInv = Inv(ins.rOp1) && Inv(ins.rOp2);
                break;
        }
        m_vInvariant[uI] = bInv ? 1 : 0;
    }
}

void C_TraceAsm::ComputeLiveness() {
    const std::size_t uCount = m_vIns.size();
    m_bLooping = !m_vIns.empty() && m_vIns.back().eOp == EIrOp::Loop;
    ComputeInvariance();
    m_vOrder.reserve(uCount);
    for (std::size_t uI = 0; uI < uCount; ++uI)
        if (m_vIns[uI].eOp == EIrOp::SLoad) m_vOrder.push_back(uI);
    for (std::size_t uI = 0; uI < uCount; ++uI)
        if (m_vIns[uI].eOp != EIrOp::SLoad && m_vIns[uI].eOp != EIrOp::Nop &&
            m_vInvariant[uI])
            m_vOrder.push_back(uI);
    m_uHoisted = m_vOrder.size();
    for (std::size_t uI = 0; uI < uCount; ++uI)
        if (m_vIns[uI].eOp != EIrOp::SLoad && m_vIns[uI].eOp != EIrOp::Nop &&
            !m_vInvariant[uI])
            m_vOrder.push_back(uI);

    m_vLastUse.assign(uCount, 0);
    m_vReg.assign(uCount, kNoReg);
    m_vSpilled.assign(uCount, 0);
    m_vNoSpill.assign(uCount, 0);
    if (m_vMemOnly.size() != uCount) m_vMemOnly.assign(uCount, 0);
    const auto uEnd = static_cast<std::uint32_t>(m_vOrder.size());

    auto Use = [&](IrRef r, std::uint32_t uPos) {
        if (r == kIrNone || IsConstRef(r)) return;
        const std::size_t uI = r - kIrBias;
        if (m_vLastUse[uI] < uPos) m_vLastUse[uI] = uPos;
    };
    for (std::uint32_t uP = 0; uP < uEnd; ++uP) {
        const std::size_t uI = m_vOrder[uP];
        const IrIns_t& ins = m_vIns[uI];
        switch (ins.eOp) {
            case EIrOp::SLoad:
                break;                       // rOp1 is a raw slot index
            case EIrOp::SStore:
                Use(ins.rOp2, uP);
                break;
            case EIrOp::AddK:
                Use(ins.rOp1, uP);
                break;
            default:
                Use(ins.rOp1, uP);
                Use(ins.rOp2, uP);
                break;
        }
        const std::uint16_t uSnap = m_vInsSnap[uI];
        if (uSnap != 0xffff) {
            const Snapshot_t& snap = m_vSnap[uSnap];
            for (std::uint32_t uS = 0; uS < snap.uSlotCount; ++uS)
                Use(m_vSnapSlots[snap.uFirstSlot + uS].rValue, uP);
        }
    }
    // Entry loads, every loop-carried value, and everything hoisted into the
    // pre-roll stay live across the back edge: their definitions execute once,
    // so a register freed inside the loop would be read as garbage on the next
    // iteration. Hoisted values may still SPILL — the home slot is written once
    // and reloading from it inside the loop is always correct — but the entry
    // loads may not, because the back-edge copies write to their registers.
    // Linear traces (stems/side traces ending in End) have no back edge and
    // keep natural liveness.
    if (m_bLooping) {
        for (std::size_t uI = 0; uI < uCount; ++uI) {
            if (m_vIns[uI].eOp == EIrOp::SLoad) {
                m_vLastUse[uI] = uEnd;
                m_vNoSpill[uI] = 1;
            } else if (m_vInvariant[uI]) {
                m_vLastUse[uI] = uEnd;
            }
        }
        for (std::size_t uS = 0; uS < m_vSlotValue.size(); ++uS) {
            const IrRef rVal = m_vSlotValue[uS];
            if (rVal == kIrNone || m_vSlotEntry[uS] == kIrNone) continue;
            Use(rVal, uEnd);
        }
    }
}

void C_TraceAsm::Pin(std::uint8_t uReg, bool bFloat) {
    if (uReg == kNoReg) return;
    (bFloat ? m_uXmmPinned : m_uGprPinned) |= 1u << uReg;
}
void C_TraceAsm::UnpinAll() { m_uGprPinned = 0; m_uXmmPinned = 0; }

void C_TraceAsm::FreeDead(std::uint32_t uPos) {
    for (std::uint32_t uR = 0; uR < 16; ++uR) {
        if (m_vGprOwner[uR] && m_vLastUse[m_vGprOwner[uR] - 1] < uPos) m_vGprOwner[uR] = 0;
        if (m_vXmmOwner[uR] && m_vLastUse[m_vXmmOwner[uR] - 1] < uPos) m_vXmmOwner[uR] = 0;
    }
}

bool C_TraceAsm::Allocate(std::size_t uIdx, bool bFloat, std::uint8_t& uOut) {
    std::uint32_t* pOwner = bFloat ? m_vXmmOwner : m_vGprOwner;
    const std::uint8_t* pPool = bFloat ? kAllocXmm : kAllocGpr;
    const std::uint32_t uPoolSize = bFloat ? kNumXmm : kNumGpr;
    for (std::uint32_t uI = 0; uI < uPoolSize; ++uI) {
        if (pOwner[pPool[uI]] == 0) {
            pOwner[pPool[uI]] = static_cast<std::uint32_t>(uIdx) + 1;
            m_vReg[uIdx] = pPool[uI];
            uOut = pPool[uI];
            return true;
        }
    }
    // Spill the live value whose next use is furthest away.
    const std::uint32_t uPinned = bFloat ? m_uXmmPinned : m_uGprPinned;
    std::uint8_t uVictim = kNoReg;
    std::uint32_t uBest = 0;
    for (std::uint32_t uI = 0; uI < uPoolSize; ++uI) {
        const std::uint8_t uR = pPool[uI];
        if (uPinned & (1u << uR)) continue;
        const std::size_t uOwn = pOwner[uR] - 1;
        if (m_vNoSpill[uOwn]) continue;
        if (uVictim == kNoReg || m_vLastUse[uOwn] > uBest) {
            uVictim = uR;
            uBest = m_vLastUse[uOwn];
        }
    }
    if (uVictim == kNoReg) { m_bFailed = true; uOut = kNoReg; return false; }
    const std::size_t uOwn = pOwner[uVictim] - 1;
    // Evicting a value DEFINED IN THE PRE-ROLL is not something this pass can
    // express. Its uses earlier in the linear order run AFTER this point on
    // every iteration but the first, and they would still read the register we
    // are handing away. Record it and let Run() retry with that value pinned to
    // memory for the whole loop.
    if (m_vInvariant[uOwn] && m_vIns[uOwn].eOp != EIrOp::SLoad && !m_vMemOnly[uOwn]) {
        m_vMemOnly[uOwn] = 1;
        m_bNewMemOnly = true;
    }
    // A value whose home slot is already written needs no store here — and
    // MUST not get one: the eviction point is inside the loop body, so on the
    // next iteration the register holds something else and the store would
    // overwrite the home slot with garbage. Pre-roll values are written to
    // their home slot at definition for exactly this reason.
    if (!m_vSpilled[uOwn]) {
        if (bFloat)
            m_Emit.MovsdStore(uVictim, kRsp, SpillDisp(uOwn));
        else
            m_Emit.MovStoreR64(uVictim, kRsp, SpillDisp(uOwn));
        m_vSpilled[uOwn] = 1;
    }
    m_vReg[uOwn] = kNoReg;
    pOwner[uVictim] = static_cast<std::uint32_t>(uIdx) + 1;
    m_vReg[uIdx] = uVictim;
    uOut = uVictim;
    return true;
}

void C_TraceAsm::MaterializeConstGpr(std::uint8_t uReg, std::uint64_t u) {
    m_Emit.MovR64Imm64(uReg, u);
}
void C_TraceAsm::MaterializeConstXmm(std::uint8_t uXmm, std::uint64_t u) {
    m_Emit.MovR64Imm64(kGprScratch[1], u);
    m_Emit.MovqXmmR64(uXmm, kGprScratch[1]);
}

std::uint8_t C_TraceAsm::OperandGpr(IrRef r, int nScratch) {
    if (IsConstRef(r)) {
        MaterializeConstGpr(kGprScratch[nScratch], ConstOf(r));
        return kGprScratch[nScratch];
    }
    const std::size_t uI = r - kIrBias;
    if (m_vMemOnly[uI]) {
        m_Emit.MovLoadR64(kGprScratch[nScratch], kRsp, SpillDisp(uI));
        return kGprScratch[nScratch];
    }
    if (m_vReg[uI] != kNoReg) { Pin(m_vReg[uI], false); return m_vReg[uI]; }
    std::uint8_t uReg = kNoReg;
    if (!Allocate(uI, false, uReg)) return kGprScratch[nScratch];
    m_Emit.MovLoadR64(uReg, kRsp, SpillDisp(uI));
    Pin(uReg, false);
    return uReg;
}

std::uint8_t C_TraceAsm::OperandXmm(IrRef r, int nScratch) {
    if (IsConstRef(r)) {
        MaterializeConstXmm(kXmmScratch[nScratch], ConstOf(r));
        return kXmmScratch[nScratch];
    }
    const std::size_t uI = r - kIrBias;
    if (m_vMemOnly[uI]) {
        m_Emit.MovsdLoad(kXmmScratch[nScratch], kRsp, SpillDisp(uI));
        return kXmmScratch[nScratch];
    }
    if (m_vReg[uI] != kNoReg) { Pin(m_vReg[uI], true); return m_vReg[uI]; }
    std::uint8_t uReg = kNoReg;
    if (!Allocate(uI, true, uReg)) return kXmmScratch[nScratch];
    m_Emit.MovsdLoad(uReg, kRsp, SpillDisp(uI));
    Pin(uReg, true);
    return uReg;
}

// Captures the machine state a snapshot describes at the CURRENT program
// point; the stub emitted later writes it back to the Lua stack.
std::uint32_t C_TraceAsm::BeginExit(std::uint16_t uSnap) {
    PendingExit_t exit;
    // A hoisted guard runs before the loop body has changed anything, so its
    // exit is the entry exit: resume at the trace head, restore nothing.
    if (m_uPos < m_uHoisted) uSnap = 0;
    const Snapshot_t& snap = m_vSnap[uSnap];
    exit.pResumePc = static_cast<const vm::BcIns_t*>(snap.pResumePc);
    exit.nBaseOffset = snap.nBaseOffset;
    for (std::uint32_t uS = 0; uS < snap.uSlotCount; ++uS) {
        const SnapSlot_t& ss = m_vSnapSlots[snap.uFirstSlot + uS];
        ExitLoc_t loc{};
        loc.nSlot = ss.nSlot;
        loc.bFloat = IsFloatType(TypeOf(ss.rValue));
        if (IsConstRef(ss.rValue)) {
            loc.bConst = true;
            loc.uConst = ConstOf(ss.rValue);
            loc.uReg = kNoReg;
        } else {
            const std::size_t uI = ss.rValue - kIrBias;
            loc.uReg = m_vReg[uI];
            loc.bSpilled = loc.uReg == kNoReg;
            loc.uSpillIdx = static_cast<std::uint32_t>(uI);
        }
        exit.vLocs.push_back(loc);
    }
    exit.uIrIdx = m_uCurIdx;
    m_vExits.push_back(std::move(exit));
    return static_cast<std::uint32_t>(m_vExits.size() - 1);
}

void C_TraceAsm::AddPatch(std::uint32_t uExit, std::size_t uPatch) {
    m_vExits[uExit].vPatches.push_back(uPatch);
}

bool C_TraceAsm::EmitOne(std::size_t uIdx) {
    m_uCurIdx = uIdx;
    const IrIns_t& ins = m_vIns[uIdx];
    const std::uint16_t uSnap = m_vInsSnap[uIdx];
    const bool bFloat = IsFloatType(ins.eType);
    std::uint8_t uDst = kNoReg;

    switch (ins.eOp) {
        case EIrOp::Nop:
            return true;

        case EIrOp::SLoad: {
            const std::int32_t nDisp = SlotDisp(ins.rOp1);
            if (ins.eType == EIrType::Num) {
                // Every boxed non-double sorts at or above 0xFFF9''0000''0000''0000,
                // so ONE compare against the high dword classifies "double" —
                // no scratch registers, and the value loads straight into xmm.
                m_Emit.CmpMem32Imm(kRegBase, nDisp + 4, 0xFFF90000u);
                const std::uint32_t uExit = BeginExit(uSnap);
                AddPatch(uExit, m_Emit.Jcc(kCcAe));
                if (!Allocate(uIdx, true, uDst)) return false;
                m_Emit.MovsdLoad(uDst, kRegBase, nDisp);
                return true;
            }
            // GC/primitive tags: the tag is the top 17 bits, so shift the
            // payload bleed out of the high dword and compare once.
            m_Emit.MovLoadR32Mem(kGprScratch[0], kRegBase, nDisp + 4);
            m_Emit.ShrR32(kGprScratch[0], 15);
            m_Emit.CmpR32Imm(kGprScratch[0], TagFor(ins.eType) & 0x1FFFFu);
            const std::uint32_t uExit = BeginExit(uSnap);
            AddPatch(uExit, m_Emit.Jcc(kCcNe));
            if (!Allocate(uIdx, false, uDst)) return false;
            m_Emit.MovLoadR64(uDst, kRegBase, nDisp);
            return true;
        }

        case EIrOp::KLoad: {
            if (!Allocate(uIdx, bFloat, uDst)) return false;
            if (bFloat)
                MaterializeConstXmm(uDst, ConstOf(ins.rOp1));
            else
                MaterializeConstGpr(uDst, ConstOf(ins.rOp1));
            return true;
        }

        case EIrOp::SStore: {
            const std::int32_t nDisp = SlotDisp(ins.rOp1);
            if (IsFloatType(TypeOf(ins.rOp2)) && !IsConstRef(ins.rOp2)) {
                m_Emit.MovsdStore(OperandXmm(ins.rOp2, 0), kRegBase, nDisp);
            } else if (IsConstRef(ins.rOp2)) {
                MaterializeConstGpr(kGprScratch[0], ConstOf(ins.rOp2));
                m_Emit.MovStoreR64(kGprScratch[0], kRegBase, nDisp);
            } else {
                m_Emit.MovStoreR64(OperandGpr(ins.rOp2, 0), kRegBase, nDisp);
            }
            return true;
        }

        case EIrOp::StoreTV: {
            const std::uint8_t uPtr = OperandGpr(ins.rOp1, 0);
            if (IsConstRef(ins.rOp2)) {
                MaterializeConstGpr(kGprScratch[1], ConstOf(ins.rOp2));
                m_Emit.MovStoreR64(kGprScratch[1], uPtr, 0);
            } else if (IsFloatType(TypeOf(ins.rOp2))) {
                m_Emit.MovsdStore(OperandXmm(ins.rOp2, 1), uPtr, 0);
            } else {
                m_Emit.MovStoreR64(OperandGpr(ins.rOp2, 1), uPtr, 0);
            }
            return true;
        }

        case EIrOp::IncU32: {
            const std::uint8_t uPtr = OperandGpr(ins.rOp1, 0);
            m_Emit.AddMem32Imm8(uPtr, 0, 1);
            return true;
        }

        case EIrOp::Add: case EIrOp::Sub: case EIrOp::Mul: case EIrOp::Div: {
            const std::uint8_t uOp = ins.eOp == EIrOp::Add   ? kSseAdd
                                     : ins.eOp == EIrOp::Sub ? kSseSub
                                     : ins.eOp == EIrOp::Mul ? kSseMul
                                                             : kSseDiv;
            const std::uint8_t uA = OperandXmm(ins.rOp1, 0);
            const std::uint8_t uB = OperandXmm(ins.rOp2, 1);
            if (!Allocate(uIdx, true, uDst)) return false;
            if (uDst == uB && uDst != uA) {
                m_Emit.MovsdRegReg(kXmmTemp, uA);
                m_Emit.SseReg(uOp, kXmmTemp, uB);
                m_Emit.MovsdRegReg(uDst, kXmmTemp);
            } else {
                m_Emit.MovsdRegReg(uDst, uA);
                m_Emit.SseReg(uOp, uDst, uB);
            }
            return true;
        }

        case EIrOp::Mod: {
            // a - floor(a/b)*b, exactly as the interpreter computes it.
            const std::uint8_t uA = OperandXmm(ins.rOp1, 0);
            const std::uint8_t uB = OperandXmm(ins.rOp2, 1);
            m_Emit.MovsdRegReg(kXmmTemp, uA);
            m_Emit.SseReg(kSseDiv, kXmmTemp, uB);
            m_Emit.Roundsd(kXmmTemp, kXmmTemp, 0x09);   // round toward -inf
            m_Emit.SseReg(kSseMul, kXmmTemp, uB);
            if (!Allocate(uIdx, true, uDst)) return false;
            m_Emit.MovsdRegReg(uDst, uA);
            m_Emit.SseReg(kSseSub, uDst, kXmmTemp);
            return true;
        }

        case EIrOp::Round: {
            const std::uint8_t uA = OperandXmm(ins.rOp1, 0);
            if (!Allocate(uIdx, true, uDst)) return false;
            m_Emit.Roundsd(uDst, uA, static_cast<std::uint8_t>(ConstOf(ins.rOp2)));
            return true;
        }

        case EIrOp::Sqrt: {
            const std::uint8_t uA = OperandXmm(ins.rOp1, 0);
            if (!Allocate(uIdx, true, uDst)) return false;
            m_Emit.Sqrtsd(uDst, uA);
            return true;
        }

        case EIrOp::Abs: {
            const std::uint8_t uA = OperandXmm(ins.rOp1, 0);
            MaterializeConstXmm(kXmmTemp, ~(std::uint64_t{1} << 63));
            if (!Allocate(uIdx, true, uDst)) return false;
            m_Emit.MovsdRegReg(uDst, uA);
            m_Emit.Andpd(uDst, kXmmTemp);
            return true;
        }

        case EIrOp::Neg: {
            const std::uint8_t uA = OperandXmm(ins.rOp1, 0);
            MaterializeConstXmm(kXmmTemp, std::uint64_t{1} << 63);
            if (!Allocate(uIdx, true, uDst)) return false;
            m_Emit.MovsdRegReg(uDst, uA);
            m_Emit.Xorpd(uDst, kXmmTemp);
            return true;
        }

        case EIrOp::ToInt: {
            const std::uint8_t uA = OperandXmm(ins.rOp1, 0);
            if (!Allocate(uIdx, false, uDst)) return false;
            m_Emit.Cvttsd2si(uDst, uA, true);
            if (uSnap != 0xffff) {
                // Exactness unproven: guard the round-trip. A snapshot-less
                // ToInt means the recorder proved integrality by induction
                // (ChkInt32 in the preamble), so the conversion is exact.
                m_Emit.Cvtsi2sd(kXmmTemp, uDst, true);
                m_Emit.Ucomisd(kXmmTemp, uA);
                const std::uint32_t uExit = BeginExit(uSnap);
                AddPatch(uExit, m_Emit.Jcc(kCcNe));
                AddPatch(uExit, m_Emit.Jcc(kCcP));
            }
            return true;
        }

        case EIrOp::ChkInt32: {
            // 32-bit truncate + widen back: any value that is not an exact
            // int32 — fractional, out of range, or NaN — fails the compare.
            const std::uint8_t uA = OperandXmm(ins.rOp1, 0);
            m_Emit.Cvttsd2si(kGprScratch[0], uA, false);
            m_Emit.Cvtsi2sd(kXmmTemp, kGprScratch[0], false);
            m_Emit.Ucomisd(kXmmTemp, uA);
            const std::uint32_t uExit = BeginExit(uSnap);
            AddPatch(uExit, m_Emit.Jcc(kCcNe));
            AddPatch(uExit, m_Emit.Jcc(kCcP));
            return true;
        }

        case EIrOp::ToNum: {
            const std::uint8_t uA = OperandGpr(ins.rOp1, 0);
            if (!Allocate(uIdx, true, uDst)) return false;
            m_Emit.Cvtsi2sd(uDst, uA, true);
            return true;
        }

        case EIrOp::GuardLt: case EIrOp::GuardGe:
        case EIrOp::GuardLe: case EIrOp::GuardGt: {
            const std::uint8_t uA = OperandXmm(ins.rOp1, 0);
            const std::uint8_t uB = OperandXmm(ins.rOp2, 1);
            m_Emit.Ucomisd(uA, uB);
            const std::uint32_t uExit = BeginExit(uSnap);
            AddPatch(uExit, m_Emit.Jcc(kCcP));     // unordered: leave the trace
            std::uint8_t uCc;
            switch (ins.eOp) {
                case EIrOp::GuardLt: uCc = kCcAe; break;   // exit unless a <  b
                case EIrOp::GuardGe: uCc = kCcB; break;    // exit unless a >= b
                case EIrOp::GuardLe: uCc = kCcA; break;    // exit unless a <= b
                default:             uCc = kCcBe; break;   // exit unless a >  b
            }
            AddPatch(uExit, m_Emit.Jcc(uCc));
            return true;
        }

        case EIrOp::GuardFEq: {
            // Exit unless the doubles compare EQUAL: jne catches inequality,
            // jp catches unordered (NaN is never equal).
            const std::uint8_t uA = OperandXmm(ins.rOp1, 0);
            const std::uint8_t uB = OperandXmm(ins.rOp2, 1);
            m_Emit.Ucomisd(uA, uB);
            const std::uint32_t uExit = BeginExit(uSnap);
            AddPatch(uExit, m_Emit.Jcc(kCcNe));
            AddPatch(uExit, m_Emit.Jcc(kCcP));
            return true;
        }

        case EIrOp::GuardFNe: {
            // Exit only when equal AND ordered — ucomisd raises ZF on
            // unordered too, so parity must skip the equality exit.
            const std::uint8_t uA = OperandXmm(ins.rOp1, 0);
            const std::uint8_t uB = OperandXmm(ins.rOp2, 1);
            m_Emit.Ucomisd(uA, uB);
            const std::uint32_t uExit = BeginExit(uSnap);
            const std::size_t uSkip = m_Emit.Jcc(kCcP);
            AddPatch(uExit, m_Emit.Jcc(kCcE));
            m_Emit.PatchToHere(uSkip);
            return true;
        }

        case EIrOp::GuardEq: case EIrOp::GuardNe: {
            const std::uint8_t uA = OperandGpr(ins.rOp1, 0);
            const std::uint8_t uB = OperandGpr(ins.rOp2, 1);
            m_Emit.CmpR64(uA, uB);
            const std::uint32_t uExit = BeginExit(uSnap);
            AddPatch(uExit, m_Emit.Jcc(ins.eOp == EIrOp::GuardEq ? kCcNe : kCcE));
            return true;
        }

        case EIrOp::GuardEqI: {
            const std::uint8_t uA = OperandGpr(ins.rOp1, 0);
            m_Emit.CmpR32Imm(uA, static_cast<std::uint32_t>(ConstOf(ins.rOp2)));
            const std::uint32_t uExit = BeginExit(uSnap);
            AddPatch(uExit, m_Emit.Jcc(kCcNe));
            return true;
        }

        case EIrOp::GuardBelow: {
            const std::uint8_t uA = OperandGpr(ins.rOp1, 0);
            const std::uint8_t uB = OperandGpr(ins.rOp2, 1);
            m_Emit.CmpR64(uA, uB);
            const std::uint32_t uExit = BeginExit(uSnap);
            AddPatch(uExit, m_Emit.Jcc(kCcAe));
            return true;
        }

        case EIrOp::TabPtr: {
            const std::uint8_t uA = OperandGpr(ins.rOp1, 0);
            if (!Allocate(uIdx, false, uDst)) return false;
            m_Emit.MovR64R64(uDst, uA);
            m_Emit.ShiftR64(uDst, 17, true);
            m_Emit.ShiftR64(uDst, 17, false);
            return true;
        }

        case EIrOp::AddK: {
            const std::uint8_t uA = OperandGpr(ins.rOp1, 0);
            if (!Allocate(uIdx, false, uDst)) return false;
            m_Emit.LeaDisp(uDst, uA, static_cast<std::int32_t>(ConstOf(ins.rOp2)));
            return true;
        }

        case EIrOp::IdxPtr: {
            const std::uint8_t uA = OperandGpr(ins.rOp1, 0);
            const std::uint8_t uB = OperandGpr(ins.rOp2, 1);
            if (!Allocate(uIdx, false, uDst)) return false;
            m_Emit.LeaIndexed8(uDst, uA, uB);
            return true;
        }

        case EIrOp::AndInt: {
            const std::uint8_t uA = OperandGpr(ins.rOp1, 0);
            const std::uint8_t uB = OperandGpr(ins.rOp2, 1);
            if (!Allocate(uIdx, false, uDst)) return false;
            m_Emit.MovR64R64(uDst, uA);
            m_Emit.AndR64(uDst, uB);
            return true;
        }

        case EIrOp::MulK: {
            const std::uint8_t uA = OperandGpr(ins.rOp1, 0);
            if (!Allocate(uIdx, false, uDst)) return false;
            m_Emit.ImulR64Imm(uDst, uA, static_cast<std::uint32_t>(ConstOf(ins.rOp2)));
            return true;
        }

        case EIrOp::RefPtr: {
            const std::uint8_t uA = OperandGpr(ins.rOp1, 0);
            if (!Allocate(uIdx, false, uDst)) return false;
            m_Emit.LeaIndexed8(uDst, kRegArena, uA);
            return true;
        }

        case EIrOp::LoadU32: {
            const std::uint8_t uA = OperandGpr(ins.rOp1, 0);
            if (!Allocate(uIdx, false, uDst)) return false;
            m_Emit.MovLoadR32Mem(uDst, uA, 0);
            return true;
        }

        case EIrOp::LoadTV: {
            const std::uint8_t uA = OperandGpr(ins.rOp1, 0);
            if (ins.eType == EIrType::Num) {
                m_Emit.CmpMem32Imm(uA, 4, 0xFFF90000u);
                const std::uint32_t uExit = BeginExit(uSnap);
                AddPatch(uExit, m_Emit.Jcc(kCcAe));
                if (!Allocate(uIdx, true, uDst)) return false;
                m_Emit.MovsdLoad(uDst, uA, 0);
                return true;
            }
            if (ins.eType != EIrType::Int) {
                m_Emit.MovLoadR32Mem(kGprScratch[1], uA, 4);
                m_Emit.ShrR32(kGprScratch[1], 15);
                m_Emit.CmpR32Imm(kGprScratch[1], TagFor(ins.eType) & 0x1FFFFu);
                const std::uint32_t uExit = BeginExit(uSnap);
                AddPatch(uExit, m_Emit.Jcc(kCcNe));
            }
            if (!Allocate(uIdx, false, uDst)) return false;
            m_Emit.MovLoadR64(uDst, uA, 0);
            return true;
        }

        case EIrOp::Loop:
            EmitBackEdge();
            return true;

        case EIrOp::End: {
            // Terminal transfer: write the snapshot back and leave. The chain
            // loop enters this exit's pChild without touching the interpreter.
            const std::uint32_t uExit = BeginExit(uSnap);
            AddPatch(uExit, m_Emit.Jmp());
            return true;
        }

        default:
            return false;
    }
}

// Loop-carried values back into the registers their entry SLoads own. The
// copies are ordered so a destination is never written before every reader of
// that register has been served; a genuine cycle is broken through a scratch.
void C_TraceAsm::EmitBackEdge() {
    struct Copy_t { std::uint8_t uDst; IrRef rSrc; bool bFloat; bool bDone; };
    std::vector<Copy_t> vCopies;
    for (std::size_t uS = 0; uS < m_vSlotValue.size(); ++uS) {
        const IrRef rEntry = m_vSlotEntry[uS];
        const IrRef rVal = m_vSlotValue[uS];
        if (rEntry == kIrNone || rVal == kIrNone || rVal == rEntry) continue;
        const std::size_t uEntryIdx = rEntry - kIrBias;
        if (m_vReg[uEntryIdx] == kNoReg) { m_bFailed = true; return; }
        vCopies.push_back(Copy_t{m_vReg[uEntryIdx], rVal,
                                 IsFloatType(m_vIns[uEntryIdx].eType), false});
    }
    auto SrcReg = [&](IrRef r) -> std::uint8_t {
        if (IsConstRef(r)) return kNoReg;
        const std::size_t uI = r - kIrBias;
        return m_vReg[uI];
    };
    std::size_t uRemaining = vCopies.size();
    std::uint32_t uGuard = 0;
    while (uRemaining && uGuard++ < 64) {
        bool bProgress = false;
        for (Copy_t& c : vCopies) {
            if (c.bDone) continue;
            bool bBlocked = false;
            for (const Copy_t& o : vCopies)
                if (!o.bDone && &o != &c && SrcReg(o.rSrc) == c.uDst &&
                    IsFloatType(TypeOf(o.rSrc)) == c.bFloat)
                    bBlocked = true;
            if (bBlocked) continue;
            if (c.bFloat)
                m_Emit.MovsdRegReg(c.uDst, OperandXmm(c.rSrc, 0));
            else
                m_Emit.MovR64R64(c.uDst, OperandGpr(c.rSrc, 0));
            c.bDone = true;
            --uRemaining;
            bProgress = true;
        }
        if (!bProgress) { m_bFailed = true; return; }
    }
    if (uRemaining) m_bFailed = true;
}

bool C_TraceAsm::RunOnce() {
    if (m_vIns.empty()) return false;
    ComputeLiveness();
    const std::size_t uFrame = ((m_vIns.size() * 8 + 15) & ~std::size_t{15}) + 8;

    // --- prologue -----------------------------------------------------------
    m_Emit.PushR64(3); m_Emit.PushR64(5); m_Emit.PushR64(12);
    m_Emit.PushR64(13); m_Emit.PushR64(14); m_Emit.PushR64(15);
    m_Emit.SubRspImm32(static_cast<std::uint32_t>(uFrame));
    m_Emit.MovR64R64(kRegBase, kRdi);
    m_Emit.MovR64R64(kRegArena, kRsi);

    // --- preamble + body ----------------------------------------------------
    bool bAtLoopTop = false;
    for (std::uint32_t uP = 0; uP < m_vOrder.size(); ++uP) {
        const std::size_t uI = m_vOrder[uP];
        if (!bAtLoopTop && uP >= m_uHoisted) {
            bAtLoopTop = true;
            if (m_bLooping)
                while (m_Emit.Here() & 15) m_Emit.U8(0x90);   // align the loop top
            m_uLoopTop = m_Emit.Here();
        }
        m_uPos = uP;
        UnpinAll();
        FreeDead(uP);
        if (!EmitOne(uI) || m_bFailed) return false;
        // Everything defined in the pre-roll is written to its home slot right
        // away: the definition runs once, so this is the only point at which a
        // spill store is guaranteed to see the right value.
        if (uP < m_uHoisted && m_vIns[uI].eOp != EIrOp::SLoad && m_vReg[uI] != kNoReg &&
            !m_vSpilled[uI]) {
            if (IsFloatType(m_vIns[uI].eType))
                m_Emit.MovsdStore(m_vReg[uI], kRsp, SpillDisp(uI));
            else
                m_Emit.MovStoreR64(m_vReg[uI], kRsp, SpillDisp(uI));
            m_vSpilled[uI] = 1;
            // Memory-resident for the loop's duration: hand the register back
            // now instead of letting the body evict it and break iteration 2.
            if (m_vMemOnly[uI]) {
                (IsFloatType(m_vIns[uI].eType) ? m_vXmmOwner : m_vGprOwner)[m_vReg[uI]] = 0;
                m_vReg[uI] = kNoReg;
            }
        }
    }
    if (!bAtLoopTop) m_uLoopTop = m_Emit.Here();
    if (m_bLooping) {
        const std::size_t uBackEdge = m_Emit.Jmp();
        m_Emit.PatchTo(uBackEdge, m_uLoopTop);
    }

    // --- exit stubs ---------------------------------------------------------
    std::vector<std::size_t> vToEpilogue;
    for (std::uint32_t uE = 0; uE < m_vExits.size(); ++uE) {
        PendingExit_t& exit = m_vExits[uE];
        const std::size_t uHere = m_Emit.Here();
        for (std::size_t uPatch : exit.vPatches) m_Emit.PatchTo(uPatch, uHere);
        for (const ExitLoc_t& loc : exit.vLocs) {
            const std::int32_t nDisp = loc.nSlot * 8;
            if (loc.bConst) {
                m_Emit.MovR64Imm64(kGprScratch[0], loc.uConst);
                m_Emit.MovStoreR64(kGprScratch[0], kRegBase, nDisp);
            } else if (loc.bSpilled) {
                m_Emit.MovLoadR64(kGprScratch[0], kRsp, SpillDisp(loc.uSpillIdx));
                m_Emit.MovStoreR64(kGprScratch[0], kRegBase, nDisp);
            } else if (loc.bFloat) {
                m_Emit.MovsdStore(loc.uReg, kRegBase, nDisp);
            } else {
                m_Emit.MovStoreR64(loc.uReg, kRegBase, nDisp);
            }
        }
        m_Emit.MovR32Imm(kRax, uE);
        vToEpilogue.push_back(m_Emit.Jmp());
    }

    // --- epilogue -----------------------------------------------------------
    const std::size_t uEpilogue = m_Emit.Here();
    for (std::size_t uPatch : vToEpilogue) m_Emit.PatchTo(uPatch, uEpilogue);
    m_Emit.AddRspImm32(static_cast<std::uint32_t>(uFrame));
    m_Emit.PopR64(15); m_Emit.PopR64(14); m_Emit.PopR64(13);
    m_Emit.PopR64(12); m_Emit.PopR64(5); m_Emit.PopR64(3);
    m_Emit.Ret();
    return true;
}

// Allocation is a fixed point, not a single pass: a pre-roll value the body
// evicts has to be demoted to memory and everything re-emitted, because that
// eviction is not expressible in a linear scan over a loop. Each retry can only
// demote more values, so this terminates.
bool C_TraceAsm::Run() {
    for (std::uint32_t uAttempt = 0; uAttempt < 8; ++uAttempt) {
        m_Emit = X64Emitter{};
        m_vExits.clear();
        std::memset(m_vGprOwner, 0, sizeof m_vGprOwner);
        std::memset(m_vXmmOwner, 0, sizeof m_vXmmOwner);
        m_uGprPinned = m_uXmmPinned = 0;
        m_uLoopTop = m_uHoisted = 0;
        m_vOrder.clear();
        m_bFailed = false;
        m_bNewMemOnly = false;
        const bool bOk = RunOnce();
        if (!m_bNewMemOnly) return bOk;
    }
    return false;
}

}  // namespace

std::uint8_t* C_TraceJit::AllocCode(std::size_t uBytes) {
    if (!m_pCodeArena) {
        m_pCodeArena = static_cast<std::uint8_t*>(
            mmap(nullptr, kCodeArenaSize, PROT_READ | PROT_WRITE | PROT_EXEC,
                 MAP_PRIVATE | MAP_ANONYMOUS, -1, 0));
        if (m_pCodeArena == MAP_FAILED) {
            m_pCodeArena = nullptr;
            return nullptr;
        }
    }
    uBytes = (uBytes + 15) & ~std::size_t{15};
    if (m_uCodeUsed + uBytes > kCodeArenaSize) return nullptr;
    std::uint8_t* pCode = m_pCodeArena + m_uCodeUsed;
    m_uCodeUsed += uBytes;
    return pCode;
}

Trace_t* C_TraceJit::Assemble() {
    C_TraceAsm asmb(m_vIns, m_vInsSnap, m_vConst, m_vSnapshots, m_vSnapSlots, m_vSlotValue,
                    m_vSlotEntry, m_vForceHoist, kSlotBias);
    if (!asmb.Run()) {
        if (TraceDebug()) std::fprintf(stderr, "[trace] backend refused the trace\n");
        return nullptr;
    }
    std::uint8_t* pCode = AllocCode(asmb.m_Emit.Size());
    if (!pCode) return nullptr;
    std::memcpy(pCode, asmb.m_Emit.Data(), asmb.m_Emit.Size());
    __builtin___clear_cache(reinterpret_cast<char*>(pCode),
                            reinterpret_cast<char*>(pCode + asmb.m_Emit.Size()));
    auto* pTrace = new Trace_t;
    pTrace->pCode = pCode;
    for (std::size_t uE = 0; uE < asmb.m_vExits.size(); ++uE) {
        const PendingExit_t& exit = asmb.m_vExits[uE];
        pTrace->vExits.push_back(TraceExit_t{exit.pResumePc, exit.nBaseOffset});
        if (m_vIns[exit.uIrIdx].eOp == EIrOp::End)
            pTrace->nLinkExit = static_cast<std::int32_t>(uE);
    }
    if (TraceDebug())
        for (std::size_t uE = 0; uE < asmb.m_vExits.size(); ++uE)
            std::fprintf(stderr, "[trace]   exit %zu <- ir %zu %s\n", uE,
                         asmb.m_vExits[uE].uIrIdx,
                         IrOpName(m_vIns[asmb.m_vExits[uE].uIrIdx].eOp));
    if (const char* sDump = std::getenv("LJX_TRACEDUMP"); sDump && *sDump) {
        std::FILE* pFile = std::fopen(sDump, "wb");
        if (pFile) {
            std::fwrite(asmb.m_Emit.Data(), 1, asmb.m_Emit.Size(), pFile);
            std::fclose(pFile);
        }
    }
    return pTrace;
}

}  // namespace ljx::jit
