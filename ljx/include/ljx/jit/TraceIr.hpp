// LJX — trace IR: linear, typed, SSA.
//
// One 64-bit instruction per operation, 16-bit references, constants interned
// below a bias so `ref < kIrBias` identifies a constant with a single compare
// (ARCHITECTURE.md §7.1). Instructions of the same opcode are chained through
// `rPrev`, which is what makes CSE, load forwarding and guard elimination cost
// a short bounded walk instead of a hash lookup.
#pragma once

#include <cstdint>

#include "ljx/vm/Value.hpp"

namespace ljx::jit {

using IrRef = std::uint16_t;

inline constexpr IrRef kIrBias = 0x8000;   // constants below, instructions above
inline constexpr IrRef kIrNone = 0;

[[nodiscard]] constexpr bool IsConstRef(IrRef r) noexcept { return r < kIrBias; }

// Types a trace specializes on. Anything not listed aborts recording.
enum class EIrType : std::uint8_t {
    Nil, False, True, Num, Str, Tab, Func, Ptr, Nothing,
};

#define LJX_IR_OPS(X)                                                        \
    /* constants and slot access */                                          \
    X(KNum) X(KGc) X(KPri) X(KInt)                                           \
    X(SLoad)          /* read a stack slot; carries the type guard        */ \
    /* arithmetic (numbers, already guarded) */                              \
    X(Add) X(Sub) X(Mul) X(Div) X(Mod) X(Neg)                                \
    /* comparisons: guards that the recorded direction is taken */           \
    X(Lt) X(Ge) X(Le) X(Gt) X(EqV) X(NeV)                                    \
    /* object access */                                                      \
    X(ARef)           /* &tab.array[i], after a bounds guard              */ \
    X(HRefK)          /* &node for a constant key at its main position    */ \
    X(ALoad) X(HLoad) X(AStore) X(HStore)                                    \
    X(FLoadTabAsize) X(FLoadTabHmask) X(FLoadTabMeta) X(FLoadTabVersion)     \
    /* guards */                                                             \
    X(GuardType)      /* value has the recorded type                      */ \
    X(GuardEq)        /* value equals a recorded constant                 */ \
    X(GuardBound)     /* unsigned index < limit                           */ \
    /* control */                                                            \
    X(Phi) X(Loop) X(Nop)

enum class EIrOp : std::uint8_t {
#define LJX_IR_ENUM(name) name,
    LJX_IR_OPS(LJX_IR_ENUM)
#undef LJX_IR_ENUM
    Count_,
};

[[nodiscard]] const char* IrOpName(EIrOp eOp) noexcept;

struct IrIns_t {
    IrRef rOp1 = 0;
    IrRef rOp2 = 0;
    EIrOp eOp{};
    EIrType eType{};
    IrRef rPrev = 0;      // previous instruction with the same opcode (CSE chain)

    [[nodiscard]] constexpr std::uint32_t Operands() const noexcept {
        return static_cast<std::uint32_t>(rOp1) | (static_cast<std::uint32_t>(rOp2) << 16);
    }
};
static_assert(sizeof(IrIns_t) == 8, "IR instructions stay one 64-bit word");

// A constant referenced by the trace: a raw 64-bit payload plus its type.
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
    std::uint32_t uResumePc = 0;    // bytecode index within the resume frame
    std::int32_t nBaseOffset = 0;   // frame base, relative to the entry base
    std::uint32_t uExitLabel = 0;   // patched with the stub's code offset
};

}  // namespace ljx::jit
