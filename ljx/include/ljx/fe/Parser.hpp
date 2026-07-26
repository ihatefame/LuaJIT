// LJX — single-pass parser: source → registerized bytecode. NO AST, ever.
// Layer 4.
//
// Expressions live in ExpDesc_t, a POD tagged struct mutated in place by the
// discharge state machine (deliberately NOT std::variant/visit — the
// transitions are the codegen algorithm). Constants fold at parse time on
// real tagged values. Register allocation is pure stack discipline; scopes
// live on the native stack; jump lists are threaded through emitted Jmp
// instructions' D fields. The per-function compilation state (FuncState_t)
// is internal to Parser.cpp — the public seam is source → C_GcProto*.
#pragma once

#include <cstdint>

#include "ljx/fe/Lexer.hpp"
#include "ljx/vm/Bytecode.hpp"

namespace ljx::fe {

// Expression states. ORDER IS LOAD-BEARING: the three primitive constants
// come first (their enum value maps directly to the KPri operand encoding)
// and KNil..KNumber form the "is foldable constant" range tested by one
// compare. KCData sits DELIBERATELY outside that range (mirroring LuaJIT's
// VKLAST = VKNUM): cdata literals fold in place at creation but must never
// enter constant folding / operand commutation.
enum class EExpKind : std::uint8_t {
    KNil, KFalse, KTrue,          // ← value == primitive operand encoding
    KString, KNumber,             // ← KNil..KNumber: IsConstant() range
    KCData,                       // ← carries a value, NOT a constant expr
    Local, Upvalue, Global, Indexed,
    Jump,          // condition materialized as a pending jump
    Relocatable,   // instruction emitted, destination register unassigned
    NonReloc,      // value in a fixed register
    Call,
    Void,
};

static_assert(static_cast<int>(EExpKind::KNil) == 0 &&
                  static_cast<int>(EExpKind::KFalse) == 1 &&
                  static_cast<int>(EExpKind::KTrue) == 2,
              "primitive kinds ARE the KPri operand encoding");
static_assert(EExpKind::KNumber < EExpKind::KCData && EExpKind::KCData < EExpKind::Local,
              "KCData is outside the constant range but before value kinds");

struct SlotPair_t {
    std::uint32_t uInfo;  // register / pc / upvalue index (by kind)
    std::uint32_t uAux;   // key encoding (Indexed), call base (Call)
};

inline constexpr std::uint32_t kNoJumpPos = ~std::uint32_t{0};

struct ExpDesc_t {
    union {
        SlotPair_t slotPair;
        vm::TValue_t tvValue;   // KString / KNumber payloads (full TValue)
    } payload{};
    EExpKind eKind = EExpKind::Void;
    std::uint32_t posTrueList = kNoJumpPos;
    std::uint32_t posFalseList = kNoJumpPos;

    [[nodiscard]] constexpr bool IsConstant() const noexcept {
        return eKind <= EExpKind::KNumber;
    }
    [[nodiscard]] constexpr bool HasJumps() const noexcept {
        return posTrueList != posFalseList;
    }
};

class C_Parser {
public:
    C_Parser(C_Lexer& lexer, vm::C_Universe& uni) noexcept
        : m_pLexer(&lexer), m_pUniverse(&uni) {}

    // Parses a whole chunk into a zero-parameter prototype.
    [[nodiscard]] vm::C_GcProto* ParseChunk();

private:
    C_Lexer* m_pLexer;
    vm::C_Universe* m_pUniverse;
};

}  // namespace ljx::fe
