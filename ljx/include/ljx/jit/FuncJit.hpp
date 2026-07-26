// LJX — whole-function JIT for numeric-closed Lua functions.
//
// A function is *numeric-closed* when every value it computes is a number
// given number arguments: arithmetic, numeric comparisons, branches, and
// self-recursion, with no tables, strings, upvalue writes, or varargs. That
// property buys two things the loop JIT has to work for:
//
//   * one guard, at the boundary. Numbers in ⇒ numbers throughout, so no
//     guard is needed anywhere inside and there is no mid-function deopt;
//   * no allocation ⇒ the collector cannot run while compiled code executes,
//     so compiled frames need no Lua stack presence at all.
//
// Compiled functions therefore use a plain native ABI —
// `double f(double, …)` in xmm0.. — and recursion becomes a native `call`.
// This is what closes the gap on call-heavy numeric code (fib, ackermann,
// numeric kernels), which the loop JIT cannot touch because there is no loop.
#pragma once

#include <cstdint>
#include <vector>

#include "ljx/vm/Bytecode.hpp"

namespace ljx::vm {
class C_Universe;
class C_GcProto;
class C_GcFunction;
}

namespace ljx::jit {

// Native entry points by arity (SysV passes doubles in xmm0..xmm7).
using NativeFn1_f = double (*)(double);
using NativeFn2_f = double (*)(double, double);
using NativeFn3_f = double (*)(double, double, double);
using NativeFn4_f = double (*)(double, double, double, double);

struct CompiledFunc_t {
    void* pCode = nullptr;
    const vm::C_GcFunction* pClosure = nullptr;  // compiled against this closure
    std::uint8_t uArity = 0;
    std::uint8_t uSelfUpvalue = 0xff;            // upvalue holding the self-reference
};

class C_FuncJit {
public:
    static constexpr std::uint32_t kHotCallCount = 8;   // compile after N calls
    static constexpr std::uint8_t kMaxArity = 4;
    static constexpr std::uint8_t kMaxSlots = 8;        // xmm8..xmm15
    static constexpr std::size_t kCodeArenaSize = 1u << 20;

    explicit C_FuncJit(vm::C_Universe& uni) noexcept;
    ~C_FuncJit();

    // Called from the function-header handler. Returns the compiled entry for
    // this exact closure, or nullptr while cold / permanently rejected.
    [[nodiscard]] const CompiledFunc_t* LookupOrTick(vm::C_GcProto* pProto,
                                                     const vm::C_GcFunction* pFn);

    [[nodiscard]] std::uint64_t CompiledCount() const noexcept { return m_uCompiled; }

private:
    [[nodiscard]] CompiledFunc_t* Compile(vm::C_GcProto* pProto,
                                          const vm::C_GcFunction* pFn);
    [[nodiscard]] std::uint8_t* AllocCode(std::size_t uBytes);

    vm::C_Universe* m_pUniverse;
    std::vector<CompiledFunc_t*> m_vCompiled;   // owns the descriptors
    std::uint8_t* m_pCodeArena = nullptr;
    std::size_t m_uCodeUsed = 0;
    std::uint64_t m_uCompiled = 0;
    // Compiled code recurses on the NATIVE stack, which has no Lua-side depth
    // check, so every compiled entry tests rsp against this limit and raises a
    // Lua error instead of running off the end of the C stack.
    std::uintptr_t m_uStackLimit = 0;
};

}  // namespace ljx::jit
