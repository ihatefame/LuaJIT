// LJX — counted-loop JIT: x86-64 native codegen for hot numeric for-loops.
#include "ljx/jit/LoopJit.hpp"

#include <bit>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <sys/mman.h>
#include <vector>

#include "ljx/vm/Interpreter.hpp"

namespace ljx::jit {

using vm::BcIns_t;
using vm::EBcOp;

namespace {

// ---------------------------------------------------------------------------
// Tiny x86-64 emitter (SysV). rdi = pBase, rsi = pKNum. Slots addressed as
// [rdi + slot*8]; number constants as [rsi + idx*8]. xmm15/14/13 hold the
// loop's idx/stop/step; xmm12 is a zero; xmm0/1 are body scratch.
// ---------------------------------------------------------------------------

class X64Emitter {
public:
    [[nodiscard]] std::size_t Size() const noexcept { return m_vCode.size(); }
    [[nodiscard]] const std::uint8_t* Data() const noexcept { return m_vCode.data(); }

    void U8(std::uint8_t by) { m_vCode.push_back(by); }
    void U32(std::uint32_t u) {
        for (int nI = 0; nI < 4; ++nI) U8(static_cast<std::uint8_t>(u >> (nI * 8)));
    }
    void U64(std::uint64_t u) {
        for (int nI = 0; nI < 8; ++nI) U8(static_cast<std::uint8_t>(u >> (nI * 8)));
    }

    // disp32 [base+disp]; base is a GP reg number (rdi=7, rsi=6).
    void ModRmDisp(std::uint8_t uXmm, std::uint8_t uBaseReg, std::int32_t nDisp) {
        // mod=10 (disp32), reg=xmm low 3 bits, rm=base low 3 bits.
        U8(static_cast<std::uint8_t>(0x80 | ((uXmm & 7) << 3) | (uBaseReg & 7)));
        if ((uBaseReg & 7) == 4) U8(0x24);  // SIB for rsp/r12 family (unused here)
        U32(static_cast<std::uint32_t>(nDisp));
    }

    // movsd xmmDst, [base+disp]   F2 [REX] 0F 10 /r
    void MovsdLoad(std::uint8_t uXmm, std::uint8_t uBaseReg, std::int32_t nDisp) {
        U8(0xF2);
        RexForXmmMem(uXmm, uBaseReg);
        U8(0x0F); U8(0x10);
        ModRmDisp(uXmm, uBaseReg, nDisp);
    }
    // movsd [base+disp], xmmSrc   F2 [REX] 0F 11 /r
    void MovsdStore(std::uint8_t uXmm, std::uint8_t uBaseReg, std::int32_t nDisp) {
        U8(0xF2);
        RexForXmmMem(uXmm, uBaseReg);
        U8(0x0F); U8(0x11);
        ModRmDisp(uXmm, uBaseReg, nDisp);
    }
    // SSE binary op xmmDst, [base+disp]. uOp = 0x58 add/0x5C sub/0x59 mul/0x5E div.
    void SseMem(std::uint8_t uOp, std::uint8_t uXmm, std::uint8_t uBaseReg, std::int32_t nDisp) {
        U8(0xF2);
        RexForXmmMem(uXmm, uBaseReg);
        U8(0x0F); U8(uOp);
        ModRmDisp(uXmm, uBaseReg, nDisp);
    }
    // SSE binary op xmmDst, xmmSrc (reg-reg).
    void SseReg(std::uint8_t uOp, std::uint8_t uDst, std::uint8_t uSrc) {
        U8(0xF2);
        RexForXmmXmm(uDst, uSrc);
        U8(0x0F); U8(uOp);
        U8(static_cast<std::uint8_t>(0xC0 | ((uDst & 7) << 3) | (uSrc & 7)));
    }
    // ucomisd xmmA, xmmB   66 [REX] 0F 2E /r
    void Ucomisd(std::uint8_t uA, std::uint8_t uB) {
        U8(0x66);
        RexForXmmXmm(uA, uB);
        U8(0x0F); U8(0x2E);
        U8(static_cast<std::uint8_t>(0xC0 | ((uA & 7) << 3) | (uB & 7)));
    }
    // xorpd xmm, xmm  66 [REX] 0F 57 /r  (used to zero xmm12)
    void Xorpd(std::uint8_t uDst, std::uint8_t uSrc) {
        U8(0x66);
        RexForXmmXmm(uDst, uSrc);
        U8(0x0F); U8(0x57);
        U8(static_cast<std::uint8_t>(0xC0 | ((uDst & 7) << 3) | (uSrc & 7)));
    }

    // Register-to-register copy: movaps xmmDst, xmmSrc  ([REX] 0F 28 /r).
    // NOT movsd reg,reg — that form MERGES the upper 64 bits, creating a false
    // dependency on the destination's previous value and serializing the loop
    // through a register that should be dependency-free. movaps copies the
    // whole register and is recognized by move elimination on modern cores.
    void MovsdRegReg(std::uint8_t uDst, std::uint8_t uSrc) {
        if (uDst == uSrc) return;
        RexForXmmXmm(uDst, uSrc);
        U8(0x0F); U8(0x28);
        U8(static_cast<std::uint8_t>(0xC0 | ((uDst & 7) << 3) | (uSrc & 7)));
    }
    // movq xmm, rax   66 REX.W 0F 6E /r
    void MovqXmmRax(std::uint8_t uXmm) {
        U8(0x66);
        U8(static_cast<std::uint8_t>(0x48 | ((uXmm >= 8) ? 4 : 0)));
        U8(0x0F); U8(0x6E);
        U8(static_cast<std::uint8_t>(0xC0 | ((uXmm & 7) << 3)));  // rm = rax
    }

    // mov rax, imm64  (48 B8 imm64)
    void MovRaxImm64(std::uint64_t u) { U8(0x48); U8(0xB8); U64(u); }
    // mov [rdi+disp], rax  (48 89 87 disp32)
    void MovStoreRax(std::int32_t nDisp) { U8(0x48); U8(0x89); U8(0x87); U32(static_cast<std::uint32_t>(nDisp)); }
    // mov rax, [rdi+disp]  (48 8B 87 disp32)
    void MovLoadRax(std::int32_t nDisp) { U8(0x48); U8(0x8B); U8(0x87); U32(static_cast<std::uint32_t>(nDisp)); }
    // sar rax, 47  (48 C1 F8 2F)
    void SarRax(std::uint8_t nBits) { U8(0x48); U8(0xC1); U8(0xF8); U8(nBits); }
    // cmp eax, imm32  (3D imm32)
    void CmpEaxImm(std::uint32_t u) { U8(0x3D); U32(u); }
    // xor eax, eax
    void XorEax() { U8(0x31); U8(0xC0); }
    // mov eax, imm32
    void MovEaxImm(std::uint32_t u) { U8(0xB8); U32(u); }
    void Ret() { U8(0xC3); }

    // Conditional jump with rel32; returns the offset of the rel32 field to
    // patch later. uCc: 0x83 jae, 0x82 jb, 0x84 je, 0x85 jne, 0x86 jbe.
    std::size_t Jcc(std::uint8_t uCc) {
        U8(0x0F); U8(uCc);
        const std::size_t uPatch = m_vCode.size();
        U32(0);
        return uPatch;
    }
    std::size_t Jmp() {
        U8(0xE9);
        const std::size_t uPatch = m_vCode.size();
        U32(0);
        return uPatch;
    }
    void PatchToHere(std::size_t uPatch) {
        const std::int32_t nRel =
            static_cast<std::int32_t>(m_vCode.size()) - static_cast<std::int32_t>(uPatch + 4);
        std::memcpy(&m_vCode[uPatch], &nRel, 4);
    }
    void PatchTo(std::size_t uPatch, std::size_t uTarget) {
        const std::int32_t nRel =
            static_cast<std::int32_t>(uTarget) - static_cast<std::int32_t>(uPatch + 4);
        std::memcpy(&m_vCode[uPatch], &nRel, 4);
    }
    [[nodiscard]] std::size_t Here() const noexcept { return m_vCode.size(); }

private:
    // REX for xmm-reg + GP-mem: R bit from xmm>=8, B bit from base>=8.
    void RexForXmmMem(std::uint8_t uXmm, std::uint8_t uBaseReg) {
        const std::uint8_t uR = (uXmm >= 8) ? 4 : 0;
        const std::uint8_t uB = (uBaseReg >= 8) ? 1 : 0;
        if (uR | uB) U8(static_cast<std::uint8_t>(0x40 | uR | uB));
    }
    void RexForXmmXmm(std::uint8_t uDst, std::uint8_t uSrc) {
        const std::uint8_t uR = (uDst >= 8) ? 4 : 0;
        const std::uint8_t uB = (uSrc >= 8) ? 1 : 0;
        if (uR | uB) U8(static_cast<std::uint8_t>(0x40 | uR | uB));
    }
    std::vector<std::uint8_t> m_vCode;
};

constexpr std::uint8_t kRdi = 7;   // pBase
constexpr std::uint8_t kRsi = 6;   // pKNum
constexpr std::uint8_t kIdx = 15, kStop = 14, kStep = 13, kZero = 12;
constexpr std::uint8_t kScratch = 11;             // codegen temp
constexpr std::uint8_t kMaxPromoted = 11;         // xmm0..xmm10
constexpr std::uint8_t kNoReg = 0xff;
constexpr std::uint32_t kNotDoubleTag = 0xFFFFFFF2;  // (uint32)EValueTag::NumInt

constexpr std::uint8_t kSseAdd = 0x58, kSseMul = 0x59, kSseSub = 0x5C, kSseDiv = 0x5E;

// A supported body instruction, pre-decoded.
struct BodyOp_t {
    EBcOp eOp;
    std::uint8_t uA, uB, uC;
    std::uint16_t uD;
};

// Slot -> xmm register map (kNoReg for slots not live in this loop).
struct RegMap_t {
    std::uint8_t vSlotReg[256];
};

// Emit one body op with all operands register-resident. Number constants stay
// as memory operands ([rsi+k*8]): they are read-only, L1-hot, and folding them
// into the SSE operand costs nothing.
bool EmitBody(X64Emitter& emit, const BodyOp_t& op, const RegMap_t& regs) {
    const std::uint8_t uA = regs.vSlotReg[op.uA];
    if (uA == kNoReg) return false;
    switch (op.eOp) {
        case EBcOp::KShort: {
            const double flValue = static_cast<double>(static_cast<std::int16_t>(op.uD));
            emit.MovRaxImm64(std::bit_cast<std::uint64_t>(flValue));
            emit.MovqXmmRax(uA);
            return true;
        }
        case EBcOp::KNum:
            emit.MovsdLoad(uA, kRsi, op.uD * 8);
            return true;
        case EBcOp::Mov: {
            const std::uint8_t uSrc = regs.vSlotReg[op.uD];
            if (uSrc == kNoReg) return false;
            emit.MovsdRegReg(uA, uSrc);
            return true;
        }
        case EBcOp::Unm: {
            const std::uint8_t uSrc = regs.vSlotReg[op.uD];
            if (uSrc == kNoReg) return false;
            if (uA != uSrc) {
                emit.MovsdRegReg(uA, kZero);
                emit.SseReg(kSseSub, uA, uSrc);
            } else {
                emit.MovsdRegReg(kScratch, kZero);
                emit.SseReg(kSseSub, kScratch, uSrc);
                emit.MovsdRegReg(uA, kScratch);
            }
            return true;
        }
        case EBcOp::AddVV: case EBcOp::SubVV: case EBcOp::MulVV: case EBcOp::DivVV: {
            const std::uint8_t uB = regs.vSlotReg[op.uB];
            const std::uint8_t uC = regs.vSlotReg[op.uC];
            if (uB == kNoReg || uC == kNoReg) return false;
            const bool bCommutative = op.eOp == EBcOp::AddVV || op.eOp == EBcOp::MulVV;
            const std::uint8_t uSse = op.eOp == EBcOp::AddVV ? kSseAdd
                                     : op.eOp == EBcOp::SubVV ? kSseSub
                                     : op.eOp == EBcOp::MulVV ? kSseMul : kSseDiv;
            if (uA == uB) {                       // dst == lhs: one instruction
                emit.SseReg(uSse, uA, uC);
            } else if (uA == uC && bCommutative) {
                emit.SseReg(uSse, uA, uB);
            } else if (uA == uC) {                // dst aliases rhs, non-commutative
                emit.MovsdRegReg(kScratch, uB);
                emit.SseReg(uSse, kScratch, uC);
                emit.MovsdRegReg(uA, kScratch);
            } else {
                emit.MovsdRegReg(uA, uB);
                emit.SseReg(uSse, uA, uC);
            }
            return true;
        }
        case EBcOp::AddVN: case EBcOp::SubVN: case EBcOp::MulVN: case EBcOp::DivVN:
        case EBcOp::AddNV: case EBcOp::MulNV: {
            // base[B] op K[C]  (NV variants of add/mul are commutative).
            const std::uint8_t uB = regs.vSlotReg[op.uB];
            if (uB == kNoReg) return false;
            std::uint8_t uSse;
            switch (op.eOp) {
                case EBcOp::AddVN: case EBcOp::AddNV: uSse = kSseAdd; break;
                case EBcOp::MulVN: case EBcOp::MulNV: uSse = kSseMul; break;
                case EBcOp::SubVN: uSse = kSseSub; break;
                default: uSse = kSseDiv; break;
            }
            emit.MovsdRegReg(uA, uB);
            emit.SseMem(uSse, uA, kRsi, op.uC * 8);
            return true;
        }
        case EBcOp::SubNV: case EBcOp::DivNV: {
            // K[C] op base[B] — non-commutative, constant on the left.
            const std::uint8_t uB = regs.vSlotReg[op.uB];
            if (uB == kNoReg) return false;
            const std::uint8_t uSse = op.eOp == EBcOp::SubNV ? kSseSub : kSseDiv;
            if (uA != uB) {
                emit.MovsdLoad(uA, kRsi, op.uC * 8);
                emit.SseReg(uSse, uA, uB);
            } else {
                emit.MovsdLoad(kScratch, kRsi, op.uC * 8);
                emit.SseReg(uSse, kScratch, uB);
                emit.MovsdRegReg(uA, kScratch);
            }
            return true;
        }
        default:
            return false;
    }
}

}  // namespace

C_LoopJit::~C_LoopJit() {
    if (m_pCodeArena) munmap(m_pCodeArena, kCodeArenaSize);
}

std::uint8_t* C_LoopJit::AllocCode(std::size_t uBytes) {
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

namespace {
bool JitDebug() {
    static const bool bOn = std::getenv("LJX_JITDEBUG") != nullptr;
    return bOn;
}
}  // namespace

CompiledLoop_f C_LoopJit::LookupOrTick(const BcIns_t* pForIPc, std::uint8_t uBaseSlot) {
    LoopState_t& state = m_mapLoops[pForIPc];
    if (state.fnCompiled) return state.fnCompiled;
    if (state.bRejected) return nullptr;
    if (++state.uHitCount < kHotThreshold) return nullptr;
    CompiledLoop_f fn = Compile(pForIPc, uBaseSlot);
    if (fn) {
        state.fnCompiled = fn;
        ++m_uCompiled;
    } else {
        state.bRejected = true;  // give up permanently on this site
    }
    if (JitDebug())
        std::fprintf(stderr, "[ljx-jit] loop@%p base=%u -> %s\n",
                     static_cast<const void*>(pForIPc), uBaseSlot,
                     fn ? "COMPILED" : "rejected");
    return fn;
}

CompiledLoop_f C_LoopJit::Compile(const BcIns_t* pForIPc, std::uint8_t uBaseSlot) {
    // ---- 1. scan the body (FORI+1 .. matching FORL) ------------------------
    const BcIns_t* pBody = pForIPc + 1;
    const BcIns_t* pForL = nullptr;
    std::vector<BodyOp_t> vBody;
    bool vWritten[256] = {false};
    bool vUsed[256] = {false};          // any slot touched by the body
    std::uint32_t vInputGuards[256];
    std::uint32_t uNumGuards = 0;
    const std::uint8_t uCtrlLo = uBaseSlot;   // idx / stop / step
    const std::uint8_t uExt = uBaseSlot + 3;  // visible loop variable
    bool bExtWritten = false;

    auto NoteRead = [&](std::uint8_t uSlot) {
        vUsed[uSlot] = true;
        // Read-before-write slots must be proven doubles on entry; the loop
        // variable is a double by construction.
        if (uSlot != uExt && !vWritten[uSlot] && uNumGuards < 250)
            vInputGuards[uNumGuards++] = uSlot;
    };

    for (const BcIns_t* pIns = pBody; pIns < pBody + 512; ++pIns) {
        const BcIns_t ins{pIns->uRaw};
        const EBcOp eOp = ins.Op();
        if ((eOp == EBcOp::ForL || eOp == EBcOp::JForL || eOp == EBcOp::IForL) &&
            ins.A() == uBaseSlot) {
            pForL = pIns;
            break;
        }
        BodyOp_t op{eOp, ins.A(), ins.B(), ins.C(), ins.D()};
        // Writes to the control slots would desynchronize our register copies.
        if (op.uA >= uCtrlLo && op.uA < uExt) {
            if (JitDebug()) std::fprintf(stderr, "[ljx-jit]   reject: writes control slot\n");
            return nullptr;
        }
        switch (eOp) {
            case EBcOp::AddVV: case EBcOp::SubVV: case EBcOp::MulVV: case EBcOp::DivVV:
                NoteRead(op.uB); NoteRead(op.uC); break;
            case EBcOp::AddVN: case EBcOp::SubVN: case EBcOp::MulVN: case EBcOp::DivVN:
            case EBcOp::AddNV: case EBcOp::SubNV: case EBcOp::MulNV: case EBcOp::DivNV:
                NoteRead(op.uB); break;
            case EBcOp::Mov: case EBcOp::Unm:
                NoteRead(static_cast<std::uint8_t>(op.uD)); break;
            case EBcOp::KShort: case EBcOp::KNum: break;
            default:
                if (JitDebug())
                    std::fprintf(stderr, "[ljx-jit]   reject: unsupported op %s\n",
                                 vm::OpName(eOp));
                return nullptr;
        }
        vWritten[op.uA] = true;
        vUsed[op.uA] = true;
        if (op.uA == uExt) bExtWritten = true;
        vBody.push_back(op);
        if (vBody.size() > 64) {
            if (JitDebug()) std::fprintf(stderr, "[ljx-jit]   reject: body too large\n");
            return nullptr;
        }
    }
    if (!pForL || vBody.empty()) {
        if (JitDebug())
            std::fprintf(stderr, "[ljx-jit]   reject: %s\n",
                         pForL ? "empty body" : "no matching ForL");
        return nullptr;
    }

    // ---- 2. promote every live body slot into an xmm register --------------
    // All-or-nothing: a body needing more than kMaxPromoted live slots is left
    // to the interpreter rather than emitting a slower memory-operand mix.
    RegMap_t regs;
    std::memset(regs.vSlotReg, kNoReg, sizeof regs.vSlotReg);
    std::vector<std::uint8_t> vPromoted;   // slots that need load/store-back
    std::uint8_t uNextReg = 0;
    for (std::uint32_t uSlot = 0; uSlot < 256; ++uSlot) {
        if (!vUsed[uSlot]) continue;
        if (uSlot == uExt && !bExtWritten) continue;  // aliases the idx register
        if (uNextReg >= kMaxPromoted) {
            if (JitDebug())
                std::fprintf(stderr, "[ljx-jit]   reject: >%u live slots\n", kMaxPromoted);
            return nullptr;
        }
        regs.vSlotReg[uSlot] = uNextReg++;
        vPromoted.push_back(static_cast<std::uint8_t>(uSlot));
    }
    // The loop variable reads straight out of the induction register when the
    // body never assigns it (the common case) — saves a move per iteration.
    if (!bExtWritten) regs.vSlotReg[uExt] = kIdx;

    // ---- 3. emit -----------------------------------------------------------
    X64Emitter emit;
    const std::int32_t nIdx = uCtrlLo * 8;
    const std::int32_t nStop = (uCtrlLo + 1) * 8;
    const std::int32_t nStep = (uCtrlLo + 2) * 8;

    std::vector<std::size_t> vBailPatches;
    auto GuardDouble = [&](std::int32_t nDisp) {
        emit.MovLoadRax(nDisp);
        emit.SarRax(47);
        emit.CmpEaxImm(kNotDoubleTag);
        vBailPatches.push_back(emit.Jcc(0x83));  // jae -> not a double -> bail
    };
    // Entry guards run before ANY state is touched, so bailing is always safe.
    GuardDouble(nIdx);
    GuardDouble(nStop);
    GuardDouble(nStep);
    for (std::uint32_t uI = 0; uI < uNumGuards; ++uI)
        GuardDouble(static_cast<std::int32_t>(vInputGuards[uI]) * 8);

    emit.MovsdLoad(kIdx, kRdi, nIdx);
    emit.MovsdLoad(kStop, kRdi, nStop);
    emit.MovsdLoad(kStep, kRdi, nStep);
    emit.Xorpd(kZero, kZero);
    // Load promoted slots once, before the loop.
    for (std::uint8_t uSlot : vPromoted)
        emit.MovsdLoad(regs.vSlotReg[uSlot], kRdi, uSlot * 8);

    bool bOk = true;
    auto EmitLoop = [&](bool bAscending) {
        std::size_t uSkip;
        if (bAscending) {            // enter only if idx <= stop
            emit.Ucomisd(kStop, kIdx);
            uSkip = emit.Jcc(0x82);  // jb -> stop < idx -> skip
        } else {                     // enter only if idx >= stop
            emit.Ucomisd(kIdx, kStop);
            uSkip = emit.Jcc(0x82);
        }
        const std::size_t uTop = emit.Here();
        if (bExtWritten) emit.MovsdRegReg(regs.vSlotReg[uExt], kIdx);
        for (const BodyOp_t& op : vBody)
            if (!EmitBody(emit, op, regs)) { bOk = false; return; }
        emit.SseReg(kSseAdd, kIdx, kStep);          // idx += step
        if (bAscending) {
            emit.Ucomisd(kStop, kIdx);
            emit.PatchTo(emit.Jcc(0x83), uTop);     // jae -> stop >= idx -> loop
        } else {
            emit.Ucomisd(kIdx, kStop);
            emit.PatchTo(emit.Jcc(0x83), uTop);     // jae -> idx >= stop -> loop
        }
        emit.PatchToHere(uSkip);
    };

    emit.Ucomisd(kStep, kZero);
    const std::size_t uToDesc = emit.Jcc(0x82);     // step < 0 -> descending
    EmitLoop(true);
    const std::size_t uToDone = emit.Jmp();
    emit.PatchToHere(uToDesc);
    EmitLoop(false);
    emit.PatchToHere(uToDone);
    if (!bOk) return nullptr;

    // Write promoted slots (and the final induction value) back to the stack
    // so the interpreter and the GC observe exactly what they would have.
    for (std::uint8_t uSlot : vPromoted)
        emit.MovsdStore(regs.vSlotReg[uSlot], kRdi, uSlot * 8);
    emit.MovsdStore(kIdx, kRdi, nIdx);
    emit.XorEax();   // 0 = loop completed
    emit.Ret();

    for (std::size_t uPatch : vBailPatches) emit.PatchToHere(uPatch);
    emit.MovEaxImm(1);   // 1 = guard failed, interpret this loop
    emit.Ret();

    std::uint8_t* pCode = AllocCode(emit.Size());
    if (!pCode) {
        if (JitDebug()) std::fprintf(stderr, "[ljx-jit]   reject: no executable memory\n");
        return nullptr;
    }
    std::memcpy(pCode, emit.Data(), emit.Size());
    __builtin___clear_cache(reinterpret_cast<char*>(pCode),
                            reinterpret_cast<char*>(pCode + emit.Size()));
    if (JitDebug())
        std::fprintf(stderr, "[ljx-jit]   emitted %zu bytes, %zu ops, %zu regs\n",
                     emit.Size(), vBody.size(), vPromoted.size());
    return reinterpret_cast<CompiledLoop_f>(pCode);
}

}  // namespace ljx::jit
