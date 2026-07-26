// LJX — builtin fast functions and library registration.
// Layer 3.
//
// Every stdlib builtin that matters is dispatched AT BYTECODE COST: its
// closure's dispatch PC points at a synthetic per-builtin opcode (allocated
// above EBcOp::Count_ in the dispatch table), whose handler is the hot fast
// path. The C fallback runs only on type/argument misses and follows the
// retry protocol below. This is LuaJIT's LJLIB/ffid machinery with the
// offline buildvm scan replaced by a constexpr registry: ids, dispatch-table
// entries, fallback linkage, and JIT record-handler ids are all derived from
// ONE consteval-processed descriptor list — the compiler enforces agreement
// that four generated headers used to provide by construction.
#pragma once

#include <cstddef>
#include <cstdint>

#include "ljx/vm/Interpreter.hpp"
#include "ljx/vm/Object.hpp"

namespace ljx::vm {

// Illustrative id registry: full set mirrors LuaJIT's ~57 asm fast functions.
// X(name) — order is frozen once shipped (ids are baked into dispatch).
#define LJX_FASTFUNC_REGISTRY(X)                                    \
    X(Assert) X(Type) X(Next) X(Pairs) X(IPairs) X(GetMetatable)    \
    X(ToNumber) X(ToString) X(RawGet) X(RawEqual) X(PCall) X(XPCall)\
    X(Select) X(Unpack)                                             \
    X(MathAbs) X(MathFloor) X(MathCeil) X(MathSqrt) X(MathMin)      \
    X(MathMax) X(MathHuge)                                          \
    X(StringLen) X(StringSub) X(StringByte) X(StringChar)           \
    X(TableInsert) X(TableRemove) X(TableConcat)                    \
    X(BitAnd) X(BitOr) X(BitXor) X(BitNot) X(BitShl) X(BitShr)

enum class EFastFunc : std::uint16_t {
    None = 0,   // ffid 0 = Lua closure
    C = 1,      // ffid 1 = plain C closure
#define LJX_FF_ENUM(name) name,
    LJX_FASTFUNC_REGISTRY(LJX_FF_ENUM)
#undef LJX_FF_ENUM
    Count_,
};

static_assert(static_cast<std::uint32_t>(EFastFunc::Count_) - 2 <=
                  C_DispatchTable::kBuiltinOps,
              "every fast function needs a synthetic opcode slot");

// ---------------------------------------------------------------------------
// C-fallback protocol: the fast path bails to the registered C fallback,
// which may FIX the condition (coerce, grow the stack) and request exactly
// one retry of the fast path — or produce results / a tailcall itself.
// ---------------------------------------------------------------------------

enum class EFfhResult : std::int32_t {
    Retry = 0,       // condition fixed: re-enter the fast path (once)
    Tailcall = -1,   // callable placed below base: dispatch it
    // Positive values: N results already on the stack (encode via FfhRes).
};

[[nodiscard]] constexpr EFfhResult FfhRes(std::int32_t nResults) noexcept {
    return static_cast<EFfhResult>(nResults + 1);
}

using FastFuncFallback_f = EFfhResult (*)(C_LuaThread* pThread);

// ---------------------------------------------------------------------------
// Library registration: one constexpr descriptor per exported symbol; a
// consteval pass derives the per-module registration stream C_LibraryRegistry
// replays at universe creation (creating closures with the right ffid and
// dispatch PC, interning names, wiring upvalues).
// ---------------------------------------------------------------------------

struct LibFuncDef_t {
    const char* sName;
    EFastFunc eFastFunc;             // None → plain C function
    CFunction_f fnCFunction;         // implementation or fast-path fallback
    std::uint16_t uRecordHandlerId;  // JIT record handler (0 = not recordable)
};

class C_LibraryRegistry {
public:
    // Registers one module's descriptor span into the global environment.
    void RegisterModule(C_Universe& uni, const char* sModuleName,
                        const LibFuncDef_t* pDefs, std::uint32_t uCount);

    // The synthetic dispatch entry bytecode for a fast function (the cell a
    // builtin closure's dispatch PC points at — GG bcff[] equivalent).
    [[nodiscard]] const BcIns_t* DispatchCell(EFastFunc eFunc) const noexcept;

    [[nodiscard]] FastFuncFallback_f Fallback(EFastFunc eFunc) const noexcept;

private:
    BcIns_t m_vDispatchCells[static_cast<std::size_t>(EFastFunc::Count_)]{};
    FastFuncFallback_f m_vFallbacks[static_cast<std::size_t>(EFastFunc::Count_)]{};
};

}  // namespace ljx::vm
