// LJX — whole-function JIT for numeric-closed functions (see the header).
#include "ljx/jit/FuncJit.hpp"

#include <bit>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <sys/mman.h>
#include <sys/resource.h>

#include "X64Emitter.hpp"

#include "ljx/vm/Interpreter.hpp"
#include "ljx/vm/Object.hpp"

namespace ljx::jit {

using vm::BcIns_t;
using vm::EBcOp;

namespace {

bool JitDebugFn() {
    static const bool bOn = std::getenv("LJX_JITDEBUG") != nullptr;
    return bOn;
}

// Slots live in xmm8..xmm15; xmm7 is the codegen scratch (free once the
// entry has moved the incoming arguments out of xmm0..xmm7). rbx holds the
// prototype's number-constant base across the whole function.
constexpr std::uint8_t kSlotRegBase = 8;
constexpr std::uint8_t kScratchReg = 7;

struct CallSite_t {
    std::size_t uPatchOfs;   // offset of the imm64 holding the call target
};

struct JumpFixup_t {
    std::size_t uPatchOfs;   // offset of the rel32 field
    std::uint32_t uTargetPc; // bytecode index to land on
};

}  // namespace

// Raised from compiled code when the native stack is nearly exhausted. It
// longjmps to the enclosing protected call; abandoning the native frames is
// safe because compiled functions hold no destructors and no VM state.
extern "C" [[noreturn]] void LjxFuncJitStackOverflow(vm::C_Universe* pUni) {
    vm::RaiseError(*pUni, "stack overflow");
}

C_FuncJit::C_FuncJit(vm::C_Universe& uni) noexcept : m_pUniverse(&uni) {
    struct rlimit limit {};
    std::size_t uStackBytes = 8u << 20;
    if (getrlimit(RLIMIT_STACK, &limit) == 0 && limit.rlim_cur != RLIM_INFINITY &&
        limit.rlim_cur > (2u << 20))
        uStackBytes = static_cast<std::size_t>(limit.rlim_cur);
    const auto uHere = reinterpret_cast<std::uintptr_t>(&limit);
    m_uStackLimit = uHere - (uStackBytes - (1u << 20));   // keep 1 MB of margin
}

C_FuncJit::~C_FuncJit() {
    for (CompiledFunc_t* pFn : m_vCompiled) delete pFn;
    if (m_pCodeArena) munmap(m_pCodeArena, kCodeArenaSize);
}

std::uint8_t* C_FuncJit::AllocCode(std::size_t uBytes) {
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

const CompiledFunc_t* C_FuncJit::LookupOrTick(vm::C_GcProto* pProto,
                                              const vm::C_GcFunction* pFn) {
    static const bool bDisabled = std::getenv("LJX_NOJIT") != nullptr ||
                                 std::getenv("LJX_NOFUNCJIT") != nullptr;
    if (bDisabled) return nullptr;
    if (pProto->m_pNative) {
        auto* pCompiled = static_cast<CompiledFunc_t*>(pProto->m_pNative);
        if (pCompiled == reinterpret_cast<CompiledFunc_t*>(1)) return nullptr;  // rejected
        // Compiled code bakes in this closure's self-reference, so it is only
        // valid for the closure it was compiled against — AND only while that
        // upvalue still holds it. Nothing inside compiled code can reassign an
        // upvalue, so re-checking once per outermost entry is sufficient.
        if (pCompiled->pClosure != pFn) return nullptr;
        if (pCompiled->uSelfUpvalue != 0xff) {
            auto* pUpval = m_pUniverse->Deref<vm::C_GcUpvalue>(
                const_cast<vm::C_GcFunction*>(pFn)->UpvalRefs()[pCompiled->uSelfUpvalue]);
            const auto* pSlot = static_cast<const vm::TValue_t*>(
                core::RefToPtr(m_pUniverse->ArenaBase(), pUpval->m_rValue));
            if (pSlot->AsGcPointer() != pFn) return nullptr;   // reassigned
        }
        return pCompiled;
    }
    if (++pProto->m_uJitCount < kHotCallCount) return nullptr;
    CompiledFunc_t* pCompiled = Compile(pProto, pFn);
    pProto->m_pNative = pCompiled ? pCompiled : reinterpret_cast<void*>(1);
    if (pCompiled) ++m_uCompiled;
    if (JitDebugFn())
        std::fprintf(stderr, "[ljx-fnjit] proto@%p -> %s\n", static_cast<void*>(pProto),
                     pCompiled ? "COMPILED" : "rejected");
    return pCompiled;
}

CompiledFunc_t* C_FuncJit::Compile(vm::C_GcProto* pProto, const vm::C_GcFunction* pFn) {
    const std::uint32_t uArity = pProto->ParamCount();
    const std::uint32_t uFrame = pProto->FrameSize();
    auto Reject = [&](const char* sWhy) -> CompiledFunc_t* {
        if (JitDebugFn()) std::fprintf(stderr, "[ljx-fnjit]   reject: %s\n", sWhy);
        return nullptr;
    };
    if (uArity > kMaxArity) return Reject("arity too large");
    if (uFrame > kMaxSlots) return Reject("too many slots");
    if (pProto->m_uFlags & static_cast<std::uint8_t>(vm::EProtoFlag::IsVararg))
        return Reject("vararg");

    const auto* pBc = reinterpret_cast<const BcIns_t*>(pProto->Bytecode());
    const std::uint32_t uCount = pProto->m_uBcCount;

    // ---- pass 1: verify the function is numeric-closed ---------------------
    // A self-call is `UGet slot, uv` followed (later) by `Call slot, …`; the
    // upvalue must currently hold this very closure.
    std::uint8_t uSelfUv = 0xff;
    bool vIsSelfRef[256] = {false};      // slot currently holds the self-reference
    const vm::TValue_t tvSelf =
        vm::TValue_t::GcObject(vm::EValueTag::Function, pFn);

    for (std::uint32_t uPc = 1; uPc < uCount; ++uPc) {   // [0] is the header
        const BcIns_t ins{pBc[uPc].uRaw};
        switch (ins.Op()) {
            case EBcOp::KShort: case EBcOp::KNum: case EBcOp::Mov: case EBcOp::Unm:
            case EBcOp::AddVV: case EBcOp::SubVV: case EBcOp::MulVV: case EBcOp::DivVV:
            case EBcOp::AddVN: case EBcOp::SubVN: case EBcOp::MulVN: case EBcOp::DivVN:
            case EBcOp::AddNV: case EBcOp::SubNV: case EBcOp::MulNV: case EBcOp::DivNV:
                vIsSelfRef[ins.A()] = false;
                break;
            case EBcOp::IsLt: case EBcOp::IsGe: case EBcOp::IsLe: case EBcOp::IsGt:
            case EBcOp::IsEqN: case EBcOp::IsNeN: case EBcOp::IsEqV: case EBcOp::IsNeV:
                if (uPc + 1 >= uCount || pBc[uPc + 1].Op() != EBcOp::Jmp)
                    return Reject("comparison not followed by Jmp");
                ++uPc;   // the paired Jmp is consumed by the comparison
                break;
            case EBcOp::Jmp:
                break;
            case EBcOp::UGet: {
                // Only tolerated when it materializes the self-reference.
                const auto* pDescs = static_cast<const std::uint16_t*>(
                    core::RefToPtr(m_pUniverse->ArenaBase(), pProto->m_rUpvalDescs));
                (void)pDescs;
                const std::uint32_t uUv = ins.D();
                if (uUv >= pFn->UpvalCount()) return Reject("bad upvalue index");
                auto* pUpval = m_pUniverse->Deref<vm::C_GcUpvalue>(
                    const_cast<vm::C_GcFunction*>(pFn)->UpvalRefs()[uUv]);
                const auto* pSlot = static_cast<const vm::TValue_t*>(
                    core::RefToPtr(m_pUniverse->ArenaBase(), pUpval->m_rValue));
                if (pSlot->uRaw != tvSelf.uRaw) return Reject("upvalue is not the self-reference");
                if (uSelfUv != 0xff && uSelfUv != uUv) return Reject("multiple self upvalues");
                uSelfUv = static_cast<std::uint8_t>(uUv);
                vIsSelfRef[ins.A()] = true;
                break;
            }
            case EBcOp::Call: {
                if (!vIsSelfRef[ins.A()]) return Reject("call target is not the self-reference");
                if (ins.B() != 2) return Reject("call must want exactly one result");
                const std::uint32_t uArgs = ins.C() - 1;
                if (uArgs != uArity) return Reject("recursive call arity mismatch");
                vIsSelfRef[ins.A()] = false;   // the slot now holds the result
                break;
            }
            case EBcOp::Ret1:
                break;
            case EBcOp::Ret0:
                // Only the trailing implicit return, which must be unreachable.
                if (uPc + 1 != uCount) return Reject("Ret0 in the middle of the function");
                if (uPc == 0 || (pBc[uPc - 1].Op() != EBcOp::Ret1 &&
                                 pBc[uPc - 1].Op() != EBcOp::Jmp))
                    return Reject("function can return no value");
                break;
            default:
                return Reject(vm::OpName(ins.Op()));
        }
    }
    if (uSelfUv == 0xff && uCount > 2) {
        // No recursion: the loop JIT already covers straight-line numeric work
        // inside loops, and a leaf call still pays the interpreter's frame
        // setup, so compiling here would rarely pay for itself.
    }

    // ---- pass 2: emit ------------------------------------------------------
    X64Emitter emit;
    std::vector<std::size_t> vPcOffset(uCount + 1, 0);
    std::vector<JumpFixup_t> vFixups;
    std::vector<CallSite_t> vCallSites;

    const auto* pKNum = static_cast<const vm::TValue_t*>(
        core::RefToPtr(m_pUniverse->ArenaBase(), pProto->m_rConstants));
    const std::int32_t nSpillBytes = 64;   // 8 slots, keeps rsp 16-byte aligned

    auto SlotReg = [](std::uint32_t uSlot) {
        return static_cast<std::uint8_t>(kSlotRegBase + uSlot);
    };

    // Prologue. First the native-stack guard: recursion here is real machine
    // recursion, so without it an unbounded Lua recursion would run off the C
    // stack instead of raising.
    emit.MovRaxImm64(m_uStackLimit);
    emit.CmpR64(kRsp, kRax);                    // rsp - limit
    const std::size_t uStackOk = emit.Jcc(kCcAe);
    emit.SubRspImm8(8);                         // 16-byte alignment at the call
    emit.MovR64Imm64(kRdi, reinterpret_cast<std::uint64_t>(m_pUniverse));
    emit.MovRaxImm64(reinterpret_cast<std::uint64_t>(&LjxFuncJitStackOverflow));
    emit.CallRax();                             // noreturn: longjmps out
    emit.PatchToHere(uStackOk);

    // Then save rbx, reserve the spill area, pin the constant base, and move
    // the incoming arguments into their slot registers (argument regs
    // xmm0..7 never overlap slot regs xmm8..15, so the order is free).
    emit.PushR64(kRbx);
    emit.SubRspImm8(static_cast<std::int8_t>(nSpillBytes));
    emit.MovR64Imm64(kRbx, reinterpret_cast<std::uint64_t>(pKNum));
    for (std::uint32_t uI = 0; uI < uArity; ++uI)
        emit.MovsdRegReg(SlotReg(uI), static_cast<std::uint8_t>(uI));

    auto EmitEpilogueRet = [&]() {
        emit.AddRspImm8(static_cast<std::int8_t>(nSpillBytes));
        emit.PopR64(kRbx);
        emit.Ret();
    };

    for (std::uint32_t uPc = 1; uPc < uCount; ++uPc) {
        vPcOffset[uPc] = emit.Here();
        const BcIns_t ins{pBc[uPc].uRaw};
        const std::uint8_t uA = SlotReg(ins.A());
        switch (ins.Op()) {
            case EBcOp::KShort: {
                const double flValue = static_cast<double>(static_cast<std::int16_t>(ins.D()));
                emit.MovRaxImm64(std::bit_cast<std::uint64_t>(flValue));
                emit.MovqXmmRax(uA);
                break;
            }
            case EBcOp::KNum:
                emit.MovsdLoad(uA, kRbx, ins.D() * 8);
                break;
            case EBcOp::Mov:
                emit.MovsdRegReg(uA, SlotReg(ins.D()));
                break;
            case EBcOp::Unm:
                emit.Xorpd(kScratchReg, kScratchReg);
                emit.SseReg(kSseSub, kScratchReg, SlotReg(ins.D()));
                emit.MovsdRegReg(uA, kScratchReg);
                break;
            case EBcOp::AddVV: case EBcOp::SubVV: case EBcOp::MulVV: case EBcOp::DivVV: {
                const std::uint8_t uB = SlotReg(ins.B()), uC = SlotReg(ins.C());
                const std::uint8_t uSse = ins.Op() == EBcOp::AddVV ? kSseAdd
                                        : ins.Op() == EBcOp::SubVV ? kSseSub
                                        : ins.Op() == EBcOp::MulVV ? kSseMul : kSseDiv;
                if (uA == uB) {
                    emit.SseReg(uSse, uA, uC);
                } else if (uA == uC) {
                    emit.MovsdRegReg(kScratchReg, uB);
                    emit.SseReg(uSse, kScratchReg, uC);
                    emit.MovsdRegReg(uA, kScratchReg);
                } else {
                    emit.MovsdRegReg(uA, uB);
                    emit.SseReg(uSse, uA, uC);
                }
                break;
            }
            case EBcOp::AddVN: case EBcOp::SubVN: case EBcOp::MulVN: case EBcOp::DivVN:
            case EBcOp::AddNV: case EBcOp::MulNV: {
                std::uint8_t uSse;
                switch (ins.Op()) {
                    case EBcOp::AddVN: case EBcOp::AddNV: uSse = kSseAdd; break;
                    case EBcOp::MulVN: case EBcOp::MulNV: uSse = kSseMul; break;
                    case EBcOp::SubVN: uSse = kSseSub; break;
                    default: uSse = kSseDiv; break;
                }
                emit.MovsdRegReg(uA, SlotReg(ins.B()));
                emit.SseMem(uSse, uA, kRbx, ins.C() * 8);
                break;
            }
            case EBcOp::SubNV: case EBcOp::DivNV: {
                const std::uint8_t uSse = ins.Op() == EBcOp::SubNV ? kSseSub : kSseDiv;
                emit.MovsdLoad(kScratchReg, kRbx, ins.C() * 8);
                emit.SseReg(uSse, kScratchReg, SlotReg(ins.B()));
                emit.MovsdRegReg(uA, kScratchReg);
                break;
            }
            case EBcOp::IsLt: case EBcOp::IsGe: case EBcOp::IsLe: case EBcOp::IsGt:
            case EBcOp::IsEqN: case EBcOp::IsNeN: case EBcOp::IsEqV: case EBcOp::IsNeV: {
                const BcIns_t insJmp{pBc[uPc + 1].uRaw};
                const std::uint32_t uTarget =
                    static_cast<std::uint32_t>(static_cast<std::int32_t>(uPc + 2) +
                                               insJmp.JumpTarget());
                const std::uint8_t uLhs = SlotReg(ins.A());
                switch (ins.Op()) {
                    // Operands are swapped so the unordered (NaN) case lands on
                    // the same side the interpreter puts it on.
                    case EBcOp::IsLt:                     // taken when a < b
                        emit.Ucomisd(SlotReg(ins.D()), uLhs);
                        vFixups.push_back({emit.Jcc(kCcA), uTarget});
                        break;
                    case EBcOp::IsLe:                     // taken when a <= b
                        emit.Ucomisd(SlotReg(ins.D()), uLhs);
                        vFixups.push_back({emit.Jcc(kCcAe), uTarget});
                        break;
                    case EBcOp::IsGe:                     // taken when !(a < b)
                        emit.Ucomisd(SlotReg(ins.D()), uLhs);
                        vFixups.push_back({emit.Jcc(kCcBe), uTarget});
                        break;
                    case EBcOp::IsGt:                     // taken when !(a <= b)
                        emit.Ucomisd(SlotReg(ins.D()), uLhs);
                        vFixups.push_back({emit.Jcc(kCcB), uTarget});
                        break;
                    case EBcOp::IsEqV: case EBcOp::IsNeV:
                    case EBcOp::IsEqN: case EBcOp::IsNeN: {
                        const bool bConst =
                            ins.Op() == EBcOp::IsEqN || ins.Op() == EBcOp::IsNeN;
                        const bool bEqual =
                            ins.Op() == EBcOp::IsEqV || ins.Op() == EBcOp::IsEqN;
                        if (bConst) emit.MovsdLoad(kScratchReg, kRbx, ins.D() * 8);
                        emit.Ucomisd(uLhs, bConst ? kScratchReg : SlotReg(ins.D()));
                        if (bEqual) {
                            // NaN compares unordered: skip the taken branch.
                            const std::size_t uSkip = emit.Jcc(kCcP);
                            vFixups.push_back({emit.Jcc(kCcE), uTarget});
                            emit.PatchToHere(uSkip);
                        } else {
                            vFixups.push_back({emit.Jcc(kCcP), uTarget});
                            vFixups.push_back({emit.Jcc(kCcNe), uTarget});
                        }
                        break;
                    }
                    default: break;
                }
                ++uPc;    // consume the paired Jmp
                break;
            }
            case EBcOp::Jmp: {
                const std::uint32_t uTarget =
                    static_cast<std::uint32_t>(static_cast<std::int32_t>(uPc + 1) +
                                               ins.JumpTarget());
                vFixups.push_back({emit.Jmp(), uTarget});
                break;
            }
            case EBcOp::UGet:
                break;   // the self-reference needs no code
            case EBcOp::Call: {
                // Spill every slot (xmm is entirely caller-saved), pass the
                // arguments, recurse natively, restore, take the result.
                for (std::uint32_t uI = 0; uI < uFrame; ++uI)
                    MovsdSpill(emit, SlotReg(uI), static_cast<std::int8_t>(uI * 8), true);
                for (std::uint32_t uI = 0; uI < uArity; ++uI)
                    emit.MovsdRegReg(static_cast<std::uint8_t>(uI),
                                     SlotReg(ins.A() + 2 + uI));
                emit.MovRaxImm64(0);                       // patched to the entry point
                vCallSites.push_back({emit.Here() - 8});
                emit.CallRax();
                emit.MovsdRegReg(kScratchReg, 0);          // stash the result
                for (std::uint32_t uI = 0; uI < uFrame; ++uI)
                    MovsdSpill(emit, SlotReg(uI), static_cast<std::int8_t>(uI * 8), false);
                emit.MovsdRegReg(uA, kScratchReg);
                break;
            }
            case EBcOp::Ret1:
                emit.MovsdRegReg(0, uA);
                EmitEpilogueRet();
                break;
            case EBcOp::Ret0:
                emit.Xorpd(0, 0);
                EmitEpilogueRet();
                break;
            default:
                return Reject("internal: unhandled op in emit");
        }
    }
    vPcOffset[uCount] = emit.Here();

    std::uint8_t* pCode = AllocCode(emit.Size());
    if (!pCode) return Reject("no executable memory");
    std::memcpy(pCode, emit.Data(), emit.Size());

    // Resolve forward/backward branches and the self-call targets.
    for (const JumpFixup_t& fix : vFixups) {
        const std::int32_t nRel =
            static_cast<std::int32_t>(vPcOffset[fix.uTargetPc]) -
            static_cast<std::int32_t>(fix.uPatchOfs + 4);
        std::memcpy(pCode + fix.uPatchOfs, &nRel, 4);
    }
    const auto uEntry = reinterpret_cast<std::uint64_t>(pCode);
    for (const CallSite_t& call : vCallSites)
        std::memcpy(pCode + call.uPatchOfs, &uEntry, 8);

    __builtin___clear_cache(reinterpret_cast<char*>(pCode),
                            reinterpret_cast<char*>(pCode + emit.Size()));

    auto* pCompiled = new CompiledFunc_t{pCode, pFn, static_cast<std::uint8_t>(uArity),
                                         uSelfUv};
    m_vCompiled.push_back(pCompiled);
    if (JitDebugFn())
        std::fprintf(stderr, "[ljx-fnjit]   emitted %zu bytes, arity %u, %zu call sites\n",
                     emit.Size(), uArity, vCallSites.size());
    return pCompiled;
}

}  // namespace ljx::jit
