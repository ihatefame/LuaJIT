// LJX — minimal x86-64 (SysV) machine-code emitter shared by the JIT tiers.
//
// Deliberately tiny and direct: instructions are appended to a byte vector and
// forward branches are patched by offset. Every encoding here has been checked
// against `objdump -D -b binary -m i386:x86-64`.
#pragma once

#include <cstdint>
#include <cstring>
#include <vector>

namespace ljx::jit {

// Register numbers (GP and xmm share the numbering used by the encodings).
constexpr std::uint8_t kRdi = 7;   // first integer argument
constexpr std::uint8_t kRsi = 6;   // second integer argument
constexpr std::uint8_t kRax = 0;
constexpr std::uint8_t kRsp = 4;
constexpr std::uint8_t kRbx = 3;

// SSE scalar-double opcodes.
constexpr std::uint8_t kSseAdd = 0x58, kSseMul = 0x59, kSseSub = 0x5C, kSseDiv = 0x5E;

// Condition codes for Jcc (0x80 | cc).
constexpr std::uint8_t kCcB = 0x82, kCcAe = 0x83, kCcE = 0x84, kCcNe = 0x85;
constexpr std::uint8_t kCcBe = 0x86, kCcA = 0x87, kCcP = 0x8A, kCcNp = 0x8B;

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
    // cmp r32, imm32   81 /7 id
    // NOTE: the 0x3D short form is NOT usable here — it hardwires eax and
    // ignores REX.B, so `cmp r8d, imm` would silently compare eax instead.
    void CmpR32Imm(std::uint8_t uGpr, std::uint32_t u) {
        if (uGpr >= 8) U8(0x41);
        U8(0x81);
        U8(static_cast<std::uint8_t>(0xF8 | (uGpr & 7)));
        U32(u);
    }

    // cmp dword [base+disp32], imm32   81 /7 id
    void CmpMem32Imm(std::uint8_t uBase, std::int32_t nDisp, std::uint32_t u) {
        RexRegRm(7, uBase, false);
        U8(0x81);
        ModRmDisp(7, uBase, nDisp);
        U32(u);
    }
    // shr r32, imm8   C1 /5 ib
    void ShrR32(std::uint8_t uGpr, std::uint8_t uBits) {
        if (uGpr >= 8) U8(0x41);
        U8(0xC1);
        U8(static_cast<std::uint8_t>(0xE8 | (uGpr & 7)));
        U8(uBits);
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

    // ---- generic memory forms used by the trace backend ------------------
    // mov r64Dst, [base+disp32]      REX.W 8B /r
    void MovLoadR64(std::uint8_t uDst, std::uint8_t uBase, std::int32_t nDisp) {
        RexRegRm(uDst, uBase, true);
        U8(0x8B);
        ModRmDisp(uDst, uBase, nDisp);
    }
    // mov [base+disp32], r64Src      REX.W 89 /r
    void MovStoreR64(std::uint8_t uSrc, std::uint8_t uBase, std::int32_t nDisp) {
        RexRegRm(uSrc, uBase, true);
        U8(0x89);
        ModRmDisp(uSrc, uBase, nDisp);
    }
    // mov r32Dst, [base+disp32]      8B /r  (zero-extends to 64)
    void MovLoadR32Mem(std::uint8_t uDst, std::uint8_t uBase, std::int32_t nDisp) {
        RexRegRm(uDst, uBase, false);
        U8(0x8B);
        ModRmDisp(uDst, uBase, nDisp);
    }
    // lea r64Dst, [base+disp32]      REX.W 8D /r
    void LeaDisp(std::uint8_t uDst, std::uint8_t uBase, std::int32_t nDisp) {
        RexRegRm(uDst, uBase, true);
        U8(0x8D);
        ModRmDisp(uDst, uBase, nDisp);
    }
    // lea r64Dst, [base + index*8]   REX.W 8D /r + SIB (disp8 = 0 form, which
    // is also the encoding rbp/r13 require as a base).
    void LeaIndexed8(std::uint8_t uDst, std::uint8_t uBase, std::uint8_t uIndex) {
        U8(static_cast<std::uint8_t>(0x48 | ((uDst >= 8) ? 4 : 0) |
                                     ((uIndex >= 8) ? 2 : 0) | ((uBase >= 8) ? 1 : 0)));
        U8(0x8D);
        U8(static_cast<std::uint8_t>(0x44 | ((uDst & 7) << 3)));            // mod=01, rm=SIB
        U8(static_cast<std::uint8_t>(0xC0 | ((uIndex & 7) << 3) | (uBase & 7)));  // scale=8
        U8(0);
    }
    // add dword [base+disp32], imm8  83 /0 ib
    void AddMem32Imm8(std::uint8_t uBase, std::int32_t nDisp, std::int8_t nImm) {
        RexRegRm(0, uBase, false);
        U8(0x83);
        ModRmDisp(0, uBase, nDisp);
        U8(static_cast<std::uint8_t>(nImm));
    }
    // roundsd xmmDst, xmmSrc, imm8   66 [REX] 0F 3A 0B /r ib
    void Roundsd(std::uint8_t uDst, std::uint8_t uSrc, std::uint8_t uMode) {
        U8(0x66);
        RexForXmmXmm(uDst, uSrc);
        U8(0x0F); U8(0x3A); U8(0x0B);
        U8(static_cast<std::uint8_t>(0xC0 | ((uDst & 7) << 3) | (uSrc & 7)));
        U8(uMode);
    }
    // and r64Dst, r64Src     REX.W 21 /r   (rm = dst, reg = src)
    void AndR64(std::uint8_t uDst, std::uint8_t uSrc) {
        RexRegRm(uSrc, uDst, true);
        U8(0x21);
        U8(static_cast<std::uint8_t>(0xC0 | ((uSrc & 7) << 3) | (uDst & 7)));
    }
    // imul r64Dst, r64Src, imm32   REX.W 69 /r id
    void ImulR64Imm(std::uint8_t uDst, std::uint8_t uSrc, std::uint32_t u) {
        RexRegRm(uDst, uSrc, true);
        U8(0x69);
        U8(static_cast<std::uint8_t>(0xC0 | ((uDst & 7) << 3) | (uSrc & 7)));
        U32(u);
    }
    // sqrtsd xmmDst, xmmSrc   F2 [REX] 0F 51 /r
    void Sqrtsd(std::uint8_t uDst, std::uint8_t uSrc) {
        U8(0xF2);
        RexForXmmXmm(uDst, uSrc);
        U8(0x0F); U8(0x51);
        U8(static_cast<std::uint8_t>(0xC0 | ((uDst & 7) << 3) | (uSrc & 7)));
    }
    // andpd xmmDst, xmmSrc    66 [REX] 0F 54 /r
    void Andpd(std::uint8_t uDst, std::uint8_t uSrc) {
        U8(0x66);
        RexForXmmXmm(uDst, uSrc);
        U8(0x0F); U8(0x54);
        U8(static_cast<std::uint8_t>(0xC0 | ((uDst & 7) << 3) | (uSrc & 7)));
    }
    // cmp r64, [base+disp32]   REX.W 3B /r
    void CmpR64Mem(std::uint8_t uReg, std::uint8_t uBase, std::int32_t nDisp) {
        RexRegRm(uReg, uBase, true);
        U8(0x3B);
        ModRmDisp(uReg, uBase, nDisp);
    }
    // sub/add rsp, imm32
    void SubRspImm32(std::uint32_t u) { U8(0x48); U8(0x81); U8(0xEC); U32(u); }
    void AddRspImm32(std::uint32_t u) { U8(0x48); U8(0x81); U8(0xC4); U32(u); }
    // mov r32Dst, imm32
    void MovR32Imm(std::uint8_t uDst, std::uint32_t u) {
        if (uDst >= 8) U8(0x41);
        U8(static_cast<std::uint8_t>(0xB8 + (uDst & 7)));
        U32(u);
    }

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

// movsd [rsp+disp8], xmm  /  movsd xmm, [rsp+disp8]   (SIB form, base = rsp)
inline void MovsdSpill(X64Emitter& emit, std::uint8_t uXmm, std::int8_t nDisp, bool bStore) {
    emit.U8(0xF2);
    if (uXmm >= 8) emit.U8(0x44);
    emit.U8(0x0F);
    emit.U8(bStore ? 0x11 : 0x10);
    emit.U8(static_cast<std::uint8_t>(0x44 | ((uXmm & 7) << 3)));
    emit.U8(0x24);
    emit.U8(static_cast<std::uint8_t>(nDisp));
}

}  // namespace ljx::jit
