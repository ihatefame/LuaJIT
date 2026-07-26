// LJX — single-pass parser: source → registerized bytecode. NO AST, ever.
// Layer 4.
//
// Expressions live in ExpDesc_t, a POD tagged struct mutated in place by the
// discharge state machine (deliberately NOT std::variant/visit — the
// transitions are the codegen algorithm). Constants fold at parse time
// through the VM's own arithmetic kernel so parse-time and run-time results
// can never disagree (never folding to NaN/-0). Register allocation is pure
// stack discipline; scopes are objects on the native stack; jump lists are
// threaded through emitted Jmp instructions. See ARCHITECTURE.md §9 for the
// carried-over peepholes and this fork's syntax extensions.
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
// enter constant folding / operand commutation — their arithmetic goes
// through metamethods.
enum class EExpKind : std::uint8_t {
    KNil, KFalse, KTrue,          // ← value == primitive operand encoding
    KString, KNumber,             // ← KNil..KNumber: IsConstant() range
    KCData,                       // ← carries a value, NOT a constant expr
    Local, Upvalue, Global, Indexed,
    Jump,          // condition materialized as a pending jump
    Relocatable,   // instruction emitted, destination register unassigned
    NonReloc,      // value in a fixed register
    Call, CallSafeNav,            // fork: ?. produces guarded calls
    Void,
};

static_assert(static_cast<int>(EExpKind::KNil) == 0 &&
                  static_cast<int>(EExpKind::KFalse) == 1 &&
                  static_cast<int>(EExpKind::KTrue) == 2,
              "primitive kinds ARE the KPri operand encoding");
static_assert(EExpKind::KNumber < EExpKind::KCData &&
                  EExpKind::KCData < EExpKind::Local,
              "KCData is outside the constant range but before value kinds");

struct SlotPair_t {
    std::uint32_t uInfo;  // register / pc / upvalue index (by kind)
    std::uint32_t uAux;   // vstack index / key encoding (by kind)
};

struct ExpDesc_t {
    // Payload: constants keep a FULL TValue_t (dual-number integers must not
    // collapse to double before interning — parse-time folding uses the VM's
    // arithmetic kernel on real tagged values; KCData's payload also lives
    // here; strings are interned-string TValues).
    union {
        SlotPair_t slotPair;
        vm::TValue_t tvValue;
    } payload{};
    EExpKind eKind = EExpKind::Void;
    vm::C_BytecodeEmitter::BcPos_t posTrueList = vm::C_BytecodeEmitter::kNoJump;
    vm::C_BytecodeEmitter::BcPos_t posFalseList = vm::C_BytecodeEmitter::kNoJump;

    [[nodiscard]] constexpr bool IsConstant() const noexcept {
        return eKind <= EExpKind::KNumber;
    }
    [[nodiscard]] constexpr bool HasJumps() const noexcept {
        return posTrueList != posFalseList;
    }
};

// Scope block — lives on the NATIVE stack of the recursive-descent functions.
struct ScopeState_t {
    ScopeState_t* pPrev = nullptr;
    std::uint32_t uVarStackStart = 0;
    std::uint8_t uActiveVars = 0;
    std::uint8_t uFlags = 0;   // loop / break / goto-label / upval / continue
};

// Per-function compilation state. Constants intern through a REAL Lua table
// (value → slot index); the variable stack and bytecode stack are arenas
// shared across nested functions.
class C_FuncState {
public:
    static constexpr std::uint8_t kMaxSlots = vm::kMaxFrameSlots;

    // Stack-discipline register allocation.
    [[nodiscard]] std::uint8_t ReserveSlots(std::uint8_t uCount);
    void FreeSlot(std::uint8_t uSlot) noexcept;      // LIFO-asserted
    [[nodiscard]] std::uint8_t FreeSlotTop() const noexcept { return m_uFreeSlot; }

    // Constant interning (dedup via the constant table). Numbers intern as
    // full TValues: dual-number int constants stay distinct from doubles.
    [[nodiscard]] std::uint16_t ConstNumber(const vm::TValue_t& tvValue);
    [[nodiscard]] std::uint16_t ConstString(vm::C_GcString* pString);

    // Expression discharge state machine (the codegen core).
    void Discharge(ExpDesc_t& expr);                  // kill indirections
    void ToRegister(ExpDesc_t& expr, std::uint8_t uSlot);
    [[nodiscard]] std::uint8_t ToAnyRegister(ExpDesc_t& expr);
    void ToNextRegister(ExpDesc_t& expr);

    [[nodiscard]] vm::C_BytecodeEmitter& Emitter() noexcept { return m_Emitter; }
    [[nodiscard]] vm::C_GcProto* Finish();            // seal into a proto

private:
    vm::C_BytecodeEmitter m_Emitter;
    vm::C_GcTable* m_pConstTable = nullptr;  // rooted on the Lua stack
    C_FuncState* m_pParent = nullptr;
    ScopeState_t* m_pScope = nullptr;
    std::uint32_t m_uVarStackBase = 0;
    std::uint16_t m_uNumConstCount = 0;
    std::uint16_t m_uGcConstCount = 0;
    std::uint8_t m_uFreeSlot = 0;
    std::uint8_t m_uActiveVars = 0;
    std::uint8_t m_uFrameSize = 0;   // high-water mark
    std::uint8_t m_uUpvalCount = 0;
};

class C_Parser {
public:
    C_Parser(C_Lexer& lexer, vm::C_Universe& uni) noexcept;

    // Parse a chunk to a prototype (the only public entry).
    [[nodiscard]] vm::C_GcProto* ParseChunk();

private:
    // Recursive descent; one function per grammar production. Expression
    // parsing is precedence-climbing over a constexpr binding-power table.
    void ParseStatement();
    void ParseExpression(ExpDesc_t& expr, std::uint8_t uLimit);

    // O(1) local-variable lookup: direct-mapped hash on the interned name's
    // StrId (fork feature), LIFO push/pop discipline.
    static constexpr std::uint32_t kVarHashSize = 32;

    C_Lexer* m_pLexer;
    C_FuncState* m_pActiveFunc = nullptr;
    std::uint32_t m_vVarHash[kVarHashSize]{};
};

}  // namespace ljx::fe
