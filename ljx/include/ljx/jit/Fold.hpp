// LJX — the fold engine: on-the-fly algebraic simplification, CSE, and
// memory-forwarding dispatch, applied to EVERY emitted IR instruction.
// Layer 5.
//
// The runtime machinery is LuaJIT's, kept verbatim because it is measured,
// minimal, and branch-predictable: a 24-bit (op, leftOp, rightOp/literal)
// key, a branchless wildcard cascade from most- to least-specific, and a
// 1–2 probe semi-perfect hash into a rule table. What changes is WHERE the
// table comes from: buildvm's C-comment scraping + offline search becomes a
// consteval builder over constexpr rule descriptors — same table, no build
// drift, and the rule-set invariants become compile-time diagnostics where
// expressible.
//
// Rule-set invariants (normative):
//  R1  a rule must preserve the destination type;
//  R2  a rule must never create new instructions referencing operands across
//      a PHI barrier;
//  R3  the rule system must be monotonic (strength-reducing) — guarantees
//      fixed-point termination;
//  R4  EVERY load, store, and allocation opcode must have an any/any rule,
//      or it falls through to generic CSE and merges illegally.
#pragma once

#include <array>
#include <cstddef>
#include <cstdint>

#include "ljx/jit/Ir.hpp"

namespace ljx::jit {

class C_TraceRecorder;

// Fold outcome protocol (sentinel refs below the real-ref space).
inline constexpr IrRef_t kFoldRetry = 1;   // rule mutated input; restart match
inline constexpr IrRef_t kFoldNext = 2;    // keep matching less-specific rules
inline constexpr IrRef_t kFoldKInt = 3;    // result interned as int constant
inline constexpr IrRef_t kFoldFail = 4;    // guard provably fails → abort trace
inline constexpr IrRef_t kFoldDropGuard = 5;  // guard provably true → drop
inline constexpr IrRef_t kFoldMaxSentinel = 0x20;

// A fold rule's match pattern: instruction opcode + operand-defining opcodes
// (or wildcard / literal-value match on op2).
//
// Key layout is 24 bits: (uInsOp << 17) | (uLeftOp << 10) | uRightOp — the
// LEFT field is 7 bits wide, the RIGHT field 10 bits (it must also hold
// literal operand values, e.g. conversion modes). The wildcards are therefore
// ASYMMETRIC, exactly as in buildvm: a shared 10-bit "any" in the left field
// would overflow into the opcode bits and collide distinct rules.
struct FoldRule_t {
    std::uint16_t uInsOp;    // EIrOp of the emitted instruction
    std::uint16_t uLeftOp;   // EIrOp of op1's def, or kFoldAnyLeft
    std::uint16_t uRightOp;  // EIrOp of op2's def, literal value, or kFoldAnyRight
    IrRef_t (*fnApply)(C_TraceRecorder& rec);  // reads rec.FoldState()
};

inline constexpr std::uint16_t kFoldAnyLeft = 0x7f;    // 7-bit left field
inline constexpr std::uint16_t kFoldAnyRight = 0x3ff;  // 10-bit right field

static_assert(static_cast<std::uint16_t>(EIrOp::Count_) < kFoldAnyLeft,
              "opcodes must fit the 7-bit left key field with the wildcard on top");
static_assert(kFoldAnyRight == (1u << 10) - 1 && kFoldAnyLeft == (1u << 7) - 1,
              "key fields: ins[23:17] | left[16:10] | right[9:0] — no overlap");

// Working state handed to rule functions: the instruction being emitted plus
// COPIES of its operand instructions (two slots each — captures the 64-bit
// constant payload slot), so rules dereference operands without re-fetching.
struct FoldState_t {
    IrIns_t insNew;
    IrIns_t vLeft[2];
    IrIns_t vRight[2];
};

// ---------------------------------------------------------------------------
// consteval table construction. The builder runs the same smallest-table
// search buildvm_fold.c performs (multiply-shift and rotate-subtract hash
// families; every key must land in its primary or secondary slot) — at
// compile time, over the constexpr rule registry in jit/FoldRules.hpp.
//
// Two-pass by-value idiom (C++20-legal: no pointers escape constant
// evaluation): pass 1 sizes the table + selects hash parameters; pass 2
// materializes it as a std::array that initializes a constinit global.
// ---------------------------------------------------------------------------

struct FoldHashParams_t {
    std::uint32_t uTableSize;       // odd
    std::uint8_t uShiftA, uShiftB;  // selected hash parameters
    bool bRotateFamily;
};

[[nodiscard]] consteval FoldHashParams_t SearchFoldHashParams();

// 32-bit entries: 8-bit rule index | 24-bit key. TSize comes from
// SearchFoldHashParams().uTableSize (+1 secondary-probe slot).
template <std::size_t TSize>
[[nodiscard]] consteval std::array<std::uint32_t, TSize> BuildFoldTable(FoldHashParams_t params);

// Runtime view over the constinit array (what C_FoldEngine dispatches on).
struct FoldTable_t {
    const std::uint32_t* pHashTable;
    FoldHashParams_t params;
};

// ---------------------------------------------------------------------------
// C_FoldEngine — the per-emission dispatch pipeline:
//     fold rules → alias-analysis/forwarding → CSE → emit.
// ---------------------------------------------------------------------------

class C_FoldEngine {
public:
    explicit C_FoldEngine(C_IrBuffer& irBuf) noexcept : m_pIr(&irBuf) {}

    // The single entry: every IR emission in the recorder flows through here.
    [[nodiscard]] TRef_t Fold(C_TraceRecorder& rec);

    // Generic CSE over the per-opcode chain (search bounded by max operand).
    [[nodiscard]] IrRef_t Cse(EIrOp eOp, std::uint8_t uTypeBits,
                              IrRef_t rOp1, IrRef_t rOp2);

    // GC-step barrier: folds/CSE across the loop head are suppressed while
    // any allocation chain is non-empty (the GC may run at LOOP).
    [[nodiscard]] bool GcStepBarrier() const noexcept;

private:
    // 24-bit key: ins[23:17] | left[16:10] | right[9:0]. The wildcard cascade
    // walks most→least specific — (ins,left,right), (ins,any,right),
    // (ins,left,any), (ins,any,any) — branchlessly via the LuaJIT mask dance
    // (any = (any | (any >> 10)) ^ 0xffc00; probe key | (any & 0x1ffff)).
    [[nodiscard]] static constexpr std::uint32_t MakeKey(std::uint16_t uInsOp,
                                                         std::uint16_t uLeftOp,
                                                         std::uint16_t uRightOp) noexcept {
        return (static_cast<std::uint32_t>(uInsOp) << 17) |
               (static_cast<std::uint32_t>(uLeftOp) << 10) | uRightOp;
    }

    C_IrBuffer* m_pIr;
};

}  // namespace ljx::jit
