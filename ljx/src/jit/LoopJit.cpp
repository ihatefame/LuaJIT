// LJX — counted-loop JIT: x86-64 native codegen for hot numeric for-loops.
#include "ljx/jit/LoopJit.hpp"

#include <bit>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <sys/mman.h>
#include <vector>

#include "ljx/core/Memory.hpp"
#include "ljx/vm/Interpreter.hpp"

namespace ljx::jit {

using vm::BcIns_t;
using vm::EBcOp;

// Runtime entry called from compiled code, once at loop entry, to guarantee
// the array part can hold every index the loop will store to. Runs while the
// whole VM state is still in memory (no registers are live yet), so it is a
// safe point: growing may allocate, and that is fine here.
extern "C" void LjxJitEnsureArray(vm::C_GcTable* pTab, std::uint32_t uNeeded,
                                  vm::C_Universe* pUni) noexcept {
    if (uNeeded < pTab->m_uArraySize || uNeeded >= (1u << 26)) return;
    pTab->Resize(*pUni, uNeeded + 1, 0);
}

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

    // ---- integer / addressing forms used by the array-access path --------
    // cvttsd2si r32/r64, xmm    F2 [REX(.W)] 0F 2C /r
    // NOTE: the 32-bit form zero-extends, which silently turns a negative
    // loop step into a huge positive one — the induction path must use the
    // 64-bit form.
    void Cvttsd2si(std::uint8_t uGpr, std::uint8_t uXmm, bool bWide = false) {
        U8(0xF2);
        RexRegRm(uGpr, uXmm, bWide);
        U8(0x0F); U8(0x2C);
        U8(static_cast<std::uint8_t>(0xC0 | ((uGpr & 7) << 3) | (uXmm & 7)));
    }
    // cvtsi2sd xmm, r32/r64     F2 [REX(.W)] 0F 2A /r
    void Cvtsi2sd(std::uint8_t uXmm, std::uint8_t uGpr, bool bWide = false) {
        U8(0xF2);
        RexRegRm(uXmm, uGpr, bWide);
        U8(0x0F); U8(0x2A);
        U8(static_cast<std::uint8_t>(0xC0 | ((uXmm & 7) << 3) | (uGpr & 7)));
    }
    // cmp r64A, r64B    REX.W 39 /r
    void CmpR64(std::uint8_t uA, std::uint8_t uB) {
        RexRegRm(uB, uA, true);
        U8(0x39);
        U8(static_cast<std::uint8_t>(0xC0 | ((uB & 7) << 3) | (uA & 7)));
    }
    // cmp r32A, r32B         [REX] 39 /r   (rm = A, reg = B)
    void CmpR32(std::uint8_t uA, std::uint8_t uB) {
        RexRegRm(uB, uA, false);
        U8(0x39);
        U8(static_cast<std::uint8_t>(0xC0 | ((uB & 7) << 3) | (uA & 7)));
    }
    // add r64Dst, r64Src     REX.W 01 /r   (rm = dst, reg = src)
    void AddR64(std::uint8_t uDst, std::uint8_t uSrc) {
        RexRegRm(uSrc, uDst, true);
        U8(0x01);
        U8(static_cast<std::uint8_t>(0xC0 | ((uSrc & 7) << 3) | (uDst & 7)));
    }
    // mov r64, imm64         REX.W B8+rd
    void MovR64Imm64(std::uint8_t uGpr, std::uint64_t u) {
        U8(static_cast<std::uint8_t>(0x48 | ((uGpr >= 8) ? 1 : 0)));
        U8(static_cast<std::uint8_t>(0xB8 + (uGpr & 7)));
        U64(u);
    }
    // mov r32Dst, [r64Base+disp8]   8B /r   (zero-extends to 64)
    void MovLoadR32(std::uint8_t uDst, std::uint8_t uBase, std::int8_t nDisp) {
        RexRegRm(uDst, uBase, false);
        U8(0x8B);
        U8(static_cast<std::uint8_t>(0x40 | ((uDst & 7) << 3) | (uBase & 7)));
        U8(static_cast<std::uint8_t>(nDisp));
    }
    // mov r64Dst, [rdi+disp32]      REX.W 8B /r
    void MovLoadR64Rdi(std::uint8_t uDst, std::int32_t nDisp) {
        RexRegRm(uDst, kRdiNum, true);
        U8(0x8B);
        U8(static_cast<std::uint8_t>(0x80 | ((uDst & 7) << 3) | kRdiNum));
        U32(static_cast<std::uint32_t>(nDisp));
    }
    // shl/shr r64, imm8      REX.W C1 /4 (shl) or /5 (shr)
    void ShiftR64(std::uint8_t uGpr, std::uint8_t uBits, bool bLeft) {
        U8(static_cast<std::uint8_t>(0x48 | ((uGpr >= 8) ? 1 : 0)));
        U8(0xC1);
        U8(static_cast<std::uint8_t>(0xC0 | ((bLeft ? 4 : 5) << 3) | (uGpr & 7)));
        U8(uBits);
    }
    // movq xmm, r64          66 REX.W 0F 6E /r
    void MovqXmmR64(std::uint8_t uXmm, std::uint8_t uGpr) {
        U8(0x66);
        U8(static_cast<std::uint8_t>(0x48 | ((uXmm >= 8) ? 4 : 0) | ((uGpr >= 8) ? 1 : 0)));
        U8(0x0F); U8(0x6E);
        U8(static_cast<std::uint8_t>(0xC0 | ((uXmm & 7) << 3) | (uGpr & 7)));
    }
    // movsd xmm, [base + index*8]  /  movsd [base + index*8], xmm
    void MovsdIndexed(std::uint8_t uXmm, std::uint8_t uBase, std::uint8_t uIndex,
                      bool bStore) {
        U8(0xF2);
        const std::uint8_t uR = (uXmm >= 8) ? 4 : 0;
        const std::uint8_t uX = (uIndex >= 8) ? 2 : 0;
        const std::uint8_t uB = (uBase >= 8) ? 1 : 0;
        if (uR | uX | uB) U8(static_cast<std::uint8_t>(0x40 | uR | uX | uB));
        U8(0x0F); U8(bStore ? 0x11 : 0x10);
        U8(static_cast<std::uint8_t>(0x04 | ((uXmm & 7) << 3)));            // mod=00, rm=SIB
        U8(static_cast<std::uint8_t>(0xC0 | ((uIndex & 7) << 3) | (uBase & 7)));  // scale=8
    }
    // mov r64Dst, [base + index*8]   REX.W 8B /r + SIB
    void MovLoadR64Indexed(std::uint8_t uDst, std::uint8_t uBase, std::uint8_t uIndex) {
        U8(static_cast<std::uint8_t>(0x48 | ((uDst >= 8) ? 4 : 0) |
                                     ((uIndex >= 8) ? 2 : 0) | ((uBase >= 8) ? 1 : 0)));
        U8(0x8B);
        U8(static_cast<std::uint8_t>(0x04 | ((uDst & 7) << 3)));
        U8(static_cast<std::uint8_t>(0xC0 | ((uIndex & 7) << 3) | (uBase & 7)));
    }
    // sar r64, imm8
    void SarR64(std::uint8_t uGpr, std::uint8_t uBits) {
        U8(static_cast<std::uint8_t>(0x48 | ((uGpr >= 8) ? 1 : 0)));
        U8(0xC1);
        U8(static_cast<std::uint8_t>(0xF8 | (uGpr & 7)));
        U8(uBits);
    }
    // cmp r32, imm32
    void CmpR32Imm(std::uint8_t uGpr, std::uint32_t u) {
        if (uGpr >= 8) U8(0x41);
        if ((uGpr & 7) == 0) { U8(0x3D); } else { U8(0x81); U8(static_cast<std::uint8_t>(0xF8 | (uGpr & 7))); }
        U32(u);
    }

    // cvttsd2si r32, [rdi+disp32]   F2 [REX] 0F 2C /r (mod=10)
    void Cvttsd2siMem(std::uint8_t uGpr, std::uint8_t uBase, std::int32_t nDisp) {
        U8(0xF2);
        RexRegRm(uGpr, uBase, false);
        U8(0x0F); U8(0x2C);
        U8(static_cast<std::uint8_t>(0x80 | ((uGpr & 7) << 3) | (uBase & 7)));
        U32(static_cast<std::uint32_t>(nDisp));
    }
    // cmovl r32Dst, r32Src   0F 4C /r
    void CmovlR32(std::uint8_t uDst, std::uint8_t uSrc) {
        RexRegRm(uDst, uSrc, false);
        U8(0x0F); U8(0x4C);
        U8(static_cast<std::uint8_t>(0xC0 | ((uDst & 7) << 3) | (uSrc & 7)));
    }
    // mov r64Dst, r64Src   REX.W 89 /r  (rm = dst, reg = src)
    void MovR64R64(std::uint8_t uDst, std::uint8_t uSrc) {
        RexRegRm(uSrc, uDst, true);
        U8(0x89);
        U8(static_cast<std::uint8_t>(0xC0 | ((uSrc & 7) << 3) | (uDst & 7)));
    }
    // mov r32Dst, r32Src   89 /r
    void MovR32R32(std::uint8_t uDst, std::uint8_t uSrc) {
        RexRegRm(uSrc, uDst, false);
        U8(0x89);
        U8(static_cast<std::uint8_t>(0xC0 | ((uSrc & 7) << 3) | (uDst & 7)));
    }
    void PushR64(std::uint8_t uGpr) {
        if (uGpr >= 8) U8(0x41);
        U8(static_cast<std::uint8_t>(0x50 + (uGpr & 7)));
    }
    void PopR64(std::uint8_t uGpr) {
        if (uGpr >= 8) U8(0x41);
        U8(static_cast<std::uint8_t>(0x58 + (uGpr & 7)));
    }
    void SubRspImm8(std::int8_t n) { U8(0x48); U8(0x83); U8(0xEC); U8(static_cast<std::uint8_t>(n)); }
    void AddRspImm8(std::int8_t n) { U8(0x48); U8(0x83); U8(0xC4); U8(static_cast<std::uint8_t>(n)); }
    void CallRax() { U8(0xFF); U8(0xD0); }

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

    static constexpr std::uint8_t kRdiNum = 7;

private:
    // Generic REX for a (reg, rm) pair.
    void RexRegRm(std::uint8_t uReg, std::uint8_t uRm, bool bWide) {
        const std::uint8_t uW = bWide ? 8 : 0;
        const std::uint8_t uR = (uReg >= 8) ? 4 : 0;
        const std::uint8_t uB = (uRm >= 8) ? 1 : 0;
        if (uW | uR | uB) U8(static_cast<std::uint8_t>(0x40 | uW | uR | uB));
    }
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
// GP roles (all caller-saved in SysV, so no prologue/epilogue is needed):
constexpr std::uint8_t kGprRax = 0, kGprIdxInt = 1 /*rcx*/, kGprTag = 2 /*rdx*/;
constexpr std::uint8_t kGprArray = 8, kGprSize = 9, kGprTab = 10, kGprStepInt = 11;
constexpr std::uint32_t kTableTag = 0xFFFFFFF4;   // (uint32)EValueTag::Table
// C_GcTable field offsets the compiled code reads directly.
constexpr std::int8_t kOfsTabArray = 8;           // MRef m_rArray
constexpr std::int8_t kOfsTabArraySize = 24;      // uint32 m_uArraySize
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
bool EmitBody(X64Emitter& emit, const BodyOp_t& op, const RegMap_t& regs,
              std::vector<std::size_t>& vInLoopBails) {
    // Array store reads its A operand; every other op defines it.
    const std::uint8_t uA = regs.vSlotReg[op.uA];
    if (uA == kNoReg) return false;
    switch (op.eOp) {
        case EBcOp::TGetV: {
            // The index bound was checked at the top of the iteration; here we
            // only have to prove the loaded element is a number.
            emit.MovsdIndexed(uA, kGprArray, kGprIdxInt, /*bStore=*/false);
            emit.MovLoadR64Indexed(kGprTag, kGprArray, kGprIdxInt);
            emit.SarR64(kGprTag, 47);
            emit.CmpR32Imm(kGprTag, kNotDoubleTag);
            vInLoopBails.push_back(emit.Jcc(0x83));   // jae -> not a number
            return true;
        }
        case EBcOp::TSetV:
            // In-range array store of a double: no reallocation, no GC barrier.
            emit.MovsdIndexed(uA, kGprArray, kGprIdxInt, /*bStore=*/true);
            return true;
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
    // LJX_NOJIT=1 forces everything through the interpreter — the reference
    // side of the differential test in tests/run_lua_tests.sh.
    static const bool bDisabled = std::getenv("LJX_NOJIT") != nullptr;
    if (bDisabled) return nullptr;
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
    std::uint32_t uTabSlot = 0xffffffffu;   // the single table this loop indexes
    bool bUsesArray = false;
    bool bStoresArray = false;
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
            case EBcOp::TGetV: case EBcOp::TSetV: {
                // Array access indexed by the loop variable: t[i]. Only one
                // table per loop, and it must not be reassigned inside it.
                if (op.uC != uExt) {
                    if (JitDebug())
                        std::fprintf(stderr, "[ljx-jit]   reject: table index is not the "
                                             "loop variable\n");
                    return nullptr;
                }
                if (uTabSlot == 0xffffffffu) uTabSlot = op.uB;
                else if (uTabSlot != op.uB) {
                    if (JitDebug())
                        std::fprintf(stderr, "[ljx-jit]   reject: more than one table\n");
                    return nullptr;
                }
                if (vWritten[op.uB]) {
                    if (JitDebug())
                        std::fprintf(stderr, "[ljx-jit]   reject: table slot reassigned\n");
                    return nullptr;
                }
                if (eOp == EBcOp::TSetV) {
                    NoteRead(op.uA);        // the stored value
                    bStoresArray = true;
                }
                bUsesArray = true;
                break;
            }
            default:
                if (JitDebug())
                    std::fprintf(stderr, "[ljx-jit]   reject: unsupported op %s\n",
                                 vm::OpName(eOp));
                return nullptr;
        }
        if (eOp != EBcOp::TSetV) {   // TSetV reads its A operand, never defines it
            vWritten[op.uA] = true;
            vUsed[op.uA] = true;
            if (op.uA == uExt) bExtWritten = true;
        }
        vBody.push_back(op);
        if (vBody.size() > 64) {
            if (JitDebug()) std::fprintf(stderr, "[ljx-jit]   reject: body too large\n");
            return nullptr;
        }
    }
    // The array index register tracks the induction variable, so a body that
    // reassigns the loop variable cannot use the fast index path.
    if (bUsesArray && bExtWritten) {
        if (JitDebug())
            std::fprintf(stderr, "[ljx-jit]   reject: loop variable reassigned with array use\n");
        return nullptr;
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
        if (!vUsed[uSlot] || uSlot == uTabSlot) continue;
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

    std::vector<std::size_t> vBailPatches;      // entry guards: nothing to undo
    std::vector<std::size_t> vInLoopBails;      // mid-loop: state must be flushed
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

    if (bUsesArray) {
        // The table-type guard must precede anything that dereferences the
        // table — in particular the pre-grow call below.
        emit.MovLoadRax(static_cast<std::int32_t>(uTabSlot) * 8);
        emit.SarRax(47);
        emit.CmpEaxImm(kTableTag);
        vBailPatches.push_back(emit.Jcc(0x85));         // jne -> not a table
    }
    if (bStoresArray) {
        // Pre-grow the array to cover the whole index range, so a storing loop
        // runs entirely in native code instead of bailing on its first append.
        // Emitted BEFORE any value is cached in a register: the VM state is
        // wholly in memory here, so the call is a safe point.
        const std::int32_t nTab = static_cast<std::int32_t>(uTabSlot) * 8;
        emit.Cvttsd2siMem(kGprRax, kRdi, nStop);        // eax = (int)stop
        emit.Cvttsd2siMem(kGprIdxInt, kRdi, nIdx);      // ecx = (int)start
        emit.CmpR32(kGprRax, kGprIdxInt);
        emit.CmovlR32(kGprRax, kGprIdxInt);             // eax = max(stop, start)
        emit.PushR64(7);                                // save pBase
        emit.PushR64(6);                                // save pKNum
        emit.MovR32R32(6, kGprRax);                     // esi = needed
        emit.MovLoadR64Rdi(7, nTab);                    // rdi = tagged table value
        emit.ShiftR64(7, 17, true);
        emit.ShiftR64(7, 17, false);                    // rdi = table pointer
        emit.MovR64Imm64(kGprTag, reinterpret_cast<std::uint64_t>(m_pUniverse));
        emit.SubRspImm8(8);                             // 16-byte alignment at the call
        emit.MovRaxImm64(reinterpret_cast<std::uint64_t>(&LjxJitEnsureArray));
        emit.CallRax();
        emit.AddRspImm8(8);
        emit.PopR64(6);
        emit.PopR64(7);
    }

    emit.MovsdLoad(kIdx, kRdi, nIdx);
    emit.MovsdLoad(kStop, kRdi, nStop);
    emit.MovsdLoad(kStep, kRdi, nStep);
    emit.Xorpd(kZero, kZero);
    // Load promoted slots once, before the loop.
    for (std::uint8_t uSlot : vPromoted)
        emit.MovsdLoad(regs.vSlotReg[uSlot], kRdi, uSlot * 8);

    if (bUsesArray) {
        const std::int32_t nTab = static_cast<std::int32_t>(uTabSlot) * 8;
        // Hoist the array pointer and length (type already guarded above). Nothing inside the compiled loop
        // can reallocate the array: an out-of-range index bails to the
        // interpreter instead of growing it, so both stay valid throughout.
        emit.MovLoadR64Rdi(kGprTab, nTab);
        emit.ShiftR64(kGprTab, 17, true);               // strip the NaN-box tag
        emit.ShiftR64(kGprTab, 17, false);
        emit.MovLoadR32(kGprArray, kGprTab, kOfsTabArray);   // compressed MRef
        emit.ShiftR64(kGprArray, core::kCompressedRefShift, true);
        emit.MovR64Imm64(kGprRax, m_pUniverse->ArenaBase());
        emit.AddR64(kGprArray, kGprRax);
        emit.MovLoadR32(kGprSize, kGprTab, kOfsTabArraySize);
        // Parallel integer induction variable, so an array index needs no
        // per-iteration float->int conversion. Non-integral bounds bail.
        emit.Cvttsd2si(kGprIdxInt, kIdx, /*bWide=*/true);
        emit.Cvtsi2sd(kScratch, kGprIdxInt, /*bWide=*/true);
        emit.Ucomisd(kScratch, kIdx);
        vBailPatches.push_back(emit.Jcc(0x85));         // jne -> non-integral start
        emit.Cvttsd2si(kGprStepInt, kStep, /*bWide=*/true);
        emit.Cvtsi2sd(kScratch, kGprStepInt, /*bWide=*/true);
        emit.Ucomisd(kScratch, kStep);
        vBailPatches.push_back(emit.Jcc(0x85));         // jne -> non-integral step
    }

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
        if (bUsesArray) {                       // hoisted bounds guard
            emit.CmpR64(kGprIdxInt, kGprSize);  // unsigned 64-bit: catches negatives
            vInLoopBails.push_back(emit.Jcc(0x83));   // jae -> out of array range
        }
        for (const BodyOp_t& op : vBody)
            if (!EmitBody(emit, op, regs, vInLoopBails)) { bOk = false; return; }
        emit.SseReg(kSseAdd, kIdx, kStep);          // idx += step
        if (bUsesArray) emit.AddR64(kGprIdxInt, kGprStepInt);
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

    // Mid-loop deoptimization: flush every register-resident value and the
    // induction variable back to the stack, then hand the REMAINING iterations
    // to the interpreter. The guard sits at the top of the iteration, before
    // any side effect, so the flushed state is exactly the iteration boundary.
    if (!vInLoopBails.empty()) {
        for (std::size_t uPatch : vInLoopBails) emit.PatchToHere(uPatch);
        for (std::uint8_t uSlot : vPromoted)
            emit.MovsdStore(regs.vSlotReg[uSlot], kRdi, uSlot * 8);
        emit.MovsdStore(kIdx, kRdi, nIdx);
        emit.MovEaxImm(1);
        emit.Ret();
    }
    // Entry guards fire before any state is touched: nothing to flush.
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
    // LJX_JITDUMP=<path> writes the raw machine code for disassembly:
    //   objdump -D -b binary -m i386:x86-64 -M intel <path>
    if (const char* sDumpPath = std::getenv("LJX_JITDUMP")) {
        if (std::FILE* pFile = std::fopen(sDumpPath, "ab")) {
            std::fwrite(emit.Data(), 1, emit.Size(), pFile);
            std::fclose(pFile);
        }
    }
    return reinterpret_cast<CompiledLoop_f>(pCode);
}

}  // namespace ljx::jit
