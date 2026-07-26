// LJX — counted-loop JIT: x86-64 native codegen for hot numeric for-loops.
#include "ljx/jit/LoopJit.hpp"

#include <bit>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <sys/mman.h>
#include <vector>

#include "X64Emitter.hpp"

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

// rdi = pBase, rsi = pKNum (see the header for the shared register numbering).
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
