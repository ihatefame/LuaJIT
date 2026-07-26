// LJX — trace IR: linear, typed, SSA.
//
// One 64-bit instruction per operation, 16-bit references, constants interned
// below a bias so `ref < kIrBias` identifies a constant with a single compare
// (ARCHITECTURE.md §7.1). Instructions of the same opcode are chained through
// `rPrev`, which is what makes CSE, load forwarding and guard elimination cost
// a short bounded walk instead of a hash lookup.
//
// The IR is deliberately *below* Lua semantics: by the time a bytecode has
// been recorded, all its type dispatch has become guards and all its address
// arithmetic has become explicit pointer ops. The backend therefore never
// needs to know what a table is.
#pragma once

#include <cstdint>

#include "ljx/vm/Value.hpp"

namespace ljx::jit {

using IrRef = std::uint16_t;

inline constexpr IrRef kIrBias = 0x8000;   // constants below, instructions above
inline constexpr IrRef kIrNone = 0;

[[nodiscard]] constexpr bool IsConstRef(IrRef r) noexcept { return r < kIrBias; }

// Types a trace specializes on. The first seven mirror Lua types; `Int` and
// `Ptr` are untagged machine values that only exist inside a trace.
enum class EIrType : std::uint8_t {
    Nil, False, True, Num, Str, Tab, Func, Int, Ptr, Nothing,
};

// Values of type Num live in an xmm register; everything else in a GPR.
[[nodiscard]] constexpr bool IsFloatType(EIrType eType) noexcept {
    return eType == EIrType::Num;
}

#define LJX_IR_OPS(X)                                                          \
    X(Nop)                                                                     \
    /* stack slots. SLoad is hoisted into the trace preamble and carries the */\
    /* entry type guard; SStore is the write-through for slots the trace     */\
    /* assigns but never reads (see TRACE_DESIGN.md §6).                     */\
    X(SLoad) X(SStore)                                                         \
    X(KLoad)          /* constant -> register, so the loop body never          */ \
                      /* rematerializes it                                    */ \
    /* arithmetic on guarded numbers */                                        \
    X(Add) X(Sub) X(Mul) X(Div) X(Mod) X(Neg)                                  \
    X(ToInt)          /* double -> int64, guarded exact                     */ \
    X(ToNum)          /* int64  -> double                                   */ \
    /* guards: every one of these carries a snapshot (C_TraceJit::m_vInsSnap)*/ \
    X(GuardLt) X(GuardGe) X(GuardLe) X(GuardGt)   /* ordered, non-NaN       */ \
    X(GuardEq) X(GuardNe)        /* 64-bit raw word                         */ \
    X(GuardEqI) X(GuardBelow)    /* 32-bit equal / unsigned below           */ \
    /* address arithmetic */                                                   \
    X(TabPtr)         /* tagged value  -> untagged 47-bit pointer           */ \
    X(AddK)           /* ptr + constant byte offset                         */ \
    X(IdxPtr)         /* ptr + index*8                                      */ \
    X(RefPtr)         /* arena base + granule index*8 (GcRef_t/MRef_t)      */ \
    X(AndInt)         /* 64-bit bitwise and (hash & mask)                   */ \
    X(MulK)           /* int * constant (node index -> granule index)       */ \
    /* memory */                                                               \
    X(LoadTV)         /* tagged word at [ptr]; carries the type guard       */ \
    X(StoreTV)                                                                 \
    X(LoadU32)                                                                 \
    X(IncU32)         /* ++*(uint32*)ptr — the table version bump           */ \
    /* control */                                                              \
    X(Loop)           /* the back edge                                      */

enum class EIrOp : std::uint8_t {
#define LJX_IR_ENUM(name) name,
    LJX_IR_OPS(LJX_IR_ENUM)
#undef LJX_IR_ENUM
    Count_,
};

[[nodiscard]] const char* IrOpName(EIrOp eOp) noexcept;

// True for operations that may be eliminated by CSE. Loads are included, and
// their chains are truncated at every store, which is what makes load
// forwarding safe without an alias analysis.
[[nodiscard]] constexpr bool IsPureOp(EIrOp eOp) noexcept {
    switch (eOp) {
        case EIrOp::Nop:
        case EIrOp::SLoad:
        case EIrOp::SStore:
        case EIrOp::StoreTV:
        case EIrOp::Loop:
            return false;
        default:
            return true;
    }
}

[[nodiscard]] constexpr bool IsGuardOp(EIrOp eOp) noexcept {
    switch (eOp) {
        case EIrOp::GuardLt: case EIrOp::GuardGe: case EIrOp::GuardLe:
        case EIrOp::GuardGt: case EIrOp::GuardEq: case EIrOp::GuardNe:
        case EIrOp::GuardEqI: case EIrOp::GuardBelow:
        case EIrOp::SLoad: case EIrOp::LoadTV: case EIrOp::ToInt:
            return true;
        default:
            return false;
    }
}

struct IrIns_t {
    IrRef rOp1 = 0;
    IrRef rOp2 = 0;
    EIrOp eOp{};
    EIrType eType{};
    IrRef rPrev = 0;      // previous instruction with the same opcode (CSE chain)
};
static_assert(sizeof(IrIns_t) == 8, "IR instructions stay one 64-bit word");

// A constant referenced by the trace: a raw 64-bit payload plus its type.
// Num holds the double's bits, Str/Tab/Func the tagged word, Int/Ptr the raw
// machine value.
struct IrConst_t {
    std::uint64_t uValue = 0;
    EIrType eType{};
};

// One entry of a snapshot: which stack slot, and which IR value goes in it.
struct SnapSlot_t {
    std::int32_t nSlot;    // relative to the trace's entry base (may be negative)
    IrRef rValue;
};

// State the interpreter needs to resume after a guard fails.
struct Snapshot_t {
    std::uint32_t uFirstSlot = 0;   // index into the shared slot array
    std::uint32_t uSlotCount = 0;
    std::uint32_t uResumeOfs = 0;   // bytecode index within the resume frame
    std::int32_t nBaseOffset = 0;   // frame base, relative to the entry base
    const void* pResumePc = nullptr;   // absolute resume PC
};

}  // namespace ljx::jit
