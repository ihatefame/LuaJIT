// LJX — pragmatic counted-loop JIT (roadmap phase 3, first increment).
//
// This is NOT the full trace compiler described in jit/Trace.hpp. It is a
// focused, self-contained native-code generator for the single most valuable
// case: hot numeric `for` loops whose body is straight-line number arithmetic.
// It proves the machine-code path end to end — hot detection, an x86-64
// emitter, type guards, and a clean bail-back to the interpreter — and it
// beats the interpreter on exactly the loops it accepts. The general trace
// JIT (recorder → IR → fold → LSRA → backend) supersedes it later; the two
// share the deopt philosophy (guard, else fall back), not code.
//
// Contract of a compiled loop (SysV x86-64):
//   int fn(TValue_t* pBase, const double* pKNum)
//   returns 0 -> the loop ran to completion (resume after the loop)
//           1 -> an entry type guard failed (run the loop in the interpreter)
#pragma once

#include <cstdint>
#include <unordered_map>

#include "ljx/vm/Bytecode.hpp"
#include "ljx/vm/Value.hpp"

namespace ljx::vm {
class C_Universe;
}

namespace ljx::jit {

using CompiledLoop_f = int (*)(vm::TValue_t* pBase, const double* pKNum);

class C_LoopJit {
public:
    // Compile on first entry: a numeric for-loop entered even once may iterate
    // millions of times internally, so entry-counting would miss exactly the
    // loops most worth compiling. The emitter is cheap (direct, unoptimized
    // emission) and rejection is cached, so one-shot loops pay only microseconds.
    static constexpr std::uint32_t kHotThreshold = 1;
    static constexpr std::size_t kCodeArenaSize = 1u << 20;  // 1 MB RX page

    explicit C_LoopJit(vm::C_Universe& uni) noexcept : m_pUniverse(&uni) {}
    ~C_LoopJit();

    // Called from the FORI handler. Returns a compiled loop for this site, or
    // nullptr (still cold, or permanently rejected as uncompilable).
    [[nodiscard]] CompiledLoop_f LookupOrTick(const vm::BcIns_t* pForIPc,
                                              std::uint8_t uBaseSlot);

    [[nodiscard]] std::uint64_t CompiledCount() const noexcept { return m_uCompiled; }

private:
    struct LoopState_t {
        CompiledLoop_f fnCompiled = nullptr;
        std::uint32_t uHitCount = 0;
        bool bRejected = false;
    };

    [[nodiscard]] CompiledLoop_f Compile(const vm::BcIns_t* pForIPc, std::uint8_t uBaseSlot);
    [[nodiscard]] std::uint8_t* AllocCode(std::size_t uBytes);

    vm::C_Universe* m_pUniverse;
    std::unordered_map<const vm::BcIns_t*, LoopState_t> m_mapLoops;
    std::uint8_t* m_pCodeArena = nullptr;
    std::size_t m_uCodeUsed = 0;
    std::uint64_t m_uCompiled = 0;
};

}  // namespace ljx::jit
