// LJX — bytecode ISA: instruction format, opcode registry, emitter interface.
// Layer 1 (frozen format).
//
// 32-bit fixed-width instructions, opcode in the LOW byte (single-byte load to
// dispatch), two layouts:
//        ┌────────┬────────┬────────┬────────┐
//    ABC │   B    │   C    │   A    │   OP   │
//        ├────────┴────────┼────────┼────────┤
//    AD  │        D        │   A    │   OP   │
//        └─────────────────┴────────┴────────┘
// Conventions carried from LuaJIT (all load-bearing, see ARCHITECTURE.md §5):
//  * jumps bias D by kJumpBias (branchless application);
//  * comparison ops are ALWAYS followed by a Jmp the handler consumes inline;
//  * GC-constant operands are stored complemented (`not` yields the negative
//    index into the proto's split constant array);
//  * KShort's D is a sign-extended int16;
//  * opcode ALGEBRA (pairs/triples computed by arithmetic) is pinned below.
//
// The registry is a single X-macro generating: EBcOp, operand-mode metadata,
// dispatch-table population, disassembler names, and the algebra asserts —
// one source of truth replacing buildvm's four generated headers.
#pragma once

#include <cstddef>
#include <cstdint>

#include "ljx/core/Types.hpp"

namespace ljx::vm {

class C_GcProto;

// Truncated illustrative registry: the full set mirrors this fork's 104 ops
// (incl. BAND..BSAR bit ops) + synthetic builtin fast-function ops at the end.
// X(name, opAMode, opBMode, opCDMode, associatedMetamethod)
#define LJX_BC_REGISTRY(X)                                     \
    /* comparison ops — each MUST be followed by Jmp */        \
    X(IsLt)   X(IsGe)   X(IsLe)   X(IsGt)                      \
    X(IsEqV)  X(IsNeV)  X(IsEqS)  X(IsNeS)                     \
    X(IsEqN)  X(IsNeN)  X(IsEqP)  X(IsNeP)                     \
    /* unary test/copy */                                      \
    X(IsTC)   X(IsFC)   X(IsT)    X(IsF)                       \
    X(Mov)    X(Not)    X(Unm)    X(Len)                       \
    /* binary ops: VN / NV / VV operand-kind variants */       \
    X(AddVN)  X(SubVN)  X(MulVN)  X(DivVN)  X(ModVN)           \
    X(AddNV)  X(SubNV)  X(MulNV)  X(DivNV)  X(ModNV)           \
    X(AddVV)  X(SubVV)  X(MulVV)  X(DivVV)  X(ModVV)           \
    X(Pow)    X(Cat)                                           \
    /* constants */                                            \
    X(KStr)   X(KCData) X(KShort) X(KNum)   X(KPri)  X(KNil)   \
    /* upvalues / globals / tables */                          \
    X(UGet)   X(USetV)  X(USetS)  X(USetN)  X(USetP)           \
    X(UClo)   X(FNew)                                          \
    X(TNew)   X(TDup)   X(GGet)   X(GSet)                      \
    X(TGetV)  X(TGetS)  X(TGetB)  X(TSetV)  X(TSetS) X(TSetB)  \
    X(TSetM)                                                   \
    /* calls & returns (adjacency algebra pinned below) */     \
    X(CallM)  X(Call)   X(CallMT) X(CallT)                     \
    X(IterC)  X(IterN)  X(VarG)   X(IsNext)                    \
    X(RetM)   X(Ret)    X(Ret0)   X(Ret1)                      \
    /* loops: each has Interpreted / JIT-patched variants at +1/+2 */ \
    X(ForI)   X(JForI)                                         \
    X(ForL)   X(IForL)  X(JForL)                               \
    X(IterL)  X(IIterL) X(JIterL)                              \
    X(Loop)   X(ILoop)  X(JLoop)                               \
    X(Jmp)                                                     \
    /* function headers — called THROUGH the dispatch table */ \
    X(FuncF)  X(IFuncF) X(JFuncF)                              \
    X(FuncV)  X(IFuncV) X(JFuncV)                              \
    X(FuncC)  X(FuncCW)

enum class EBcOp : std::uint8_t {
#define LJX_BC_ENUM(name) name,
    LJX_BC_REGISTRY(LJX_BC_ENUM)
#undef LJX_BC_ENUM
    Count_,
    // Synthetic builtin fast-function opcodes are allocated from Count_ up.
};

inline constexpr std::uint32_t kJumpBias = 0x8000;
inline constexpr std::uint8_t kMaxFrameSlots = 250;   // ISA limit (8-bit A/B/C)
inline constexpr std::uint8_t kNoReg = 0xff;

struct BcIns_t {
    std::uint32_t uRaw = 0;

    [[nodiscard]] constexpr EBcOp Op() const noexcept {
        return static_cast<EBcOp>(uRaw & 0xff);
    }
    [[nodiscard]] constexpr std::uint8_t A() const noexcept {
        return static_cast<std::uint8_t>(uRaw >> 8);
    }
    [[nodiscard]] constexpr std::uint8_t B() const noexcept {
        return static_cast<std::uint8_t>(uRaw >> 24);
    }
    [[nodiscard]] constexpr std::uint8_t C() const noexcept {
        return static_cast<std::uint8_t>(uRaw >> 16);
    }
    [[nodiscard]] constexpr std::uint16_t D() const noexcept {
        return static_cast<std::uint16_t>(uRaw >> 16);
    }
    [[nodiscard]] constexpr std::int32_t JumpTarget() const noexcept {
        return static_cast<std::int32_t>(D()) - static_cast<std::int32_t>(kJumpBias);
    }

    [[nodiscard]] static constexpr BcIns_t MakeABC(EBcOp eOp, std::uint8_t uA,
                                                   std::uint8_t uB, std::uint8_t uC) noexcept {
        return {static_cast<std::uint32_t>(eOp) | (std::uint32_t{uA} << 8) |
                (std::uint32_t{uC} << 16) | (std::uint32_t{uB} << 24)};
    }
    [[nodiscard]] static constexpr BcIns_t MakeAD(EBcOp eOp, std::uint8_t uA,
                                                  std::uint16_t uD) noexcept {
        return {static_cast<std::uint32_t>(eOp) | (std::uint32_t{uA} << 8) |
                (std::uint32_t{uD} << 16)};
    }
};
static_assert(core::IsFrozenLayout<BcIns_t> && sizeof(BcIns_t) == 4);

// ---- opcode algebra pins (recorder/dispatch compute variants by arithmetic) —
namespace detail {
constexpr std::uint8_t U(EBcOp eOp) { return static_cast<std::uint8_t>(eOp); }
}  // namespace detail

static_assert((detail::U(EBcOp::IsLt) ^ 1) == detail::U(EBcOp::IsGe),
              "condition flip is opcode XOR 1");
static_assert((detail::U(EBcOp::IsLe) ^ 1) == detail::U(EBcOp::IsGt));
static_assert((detail::U(EBcOp::IsLt) ^ 3) == detail::U(EBcOp::IsGt),
              "operand-swap comparison is opcode XOR 3");
static_assert((detail::U(EBcOp::IsEqV) ^ 1) == detail::U(EBcOp::IsNeV));
static_assert(detail::U(EBcOp::ForL) + 1 == detail::U(EBcOp::IForL) &&
              detail::U(EBcOp::ForL) + 2 == detail::U(EBcOp::JForL),
              "I/J loop variants at fixed offsets (blacklist & trace patching)");
static_assert(detail::U(EBcOp::Loop) + 1 == detail::U(EBcOp::ILoop) &&
              detail::U(EBcOp::Loop) + 2 == detail::U(EBcOp::JLoop));
static_assert(detail::U(EBcOp::FuncF) + 1 == detail::U(EBcOp::IFuncF) &&
              detail::U(EBcOp::FuncF) + 2 == detail::U(EBcOp::JFuncF));
static_assert(detail::U(EBcOp::CallT) - detail::U(EBcOp::Call) ==
              detail::U(EBcOp::CallMT) - detail::U(EBcOp::CallM),
              "tailcall derivation is a constant offset");
static_assert(detail::U(EBcOp::RetM) + 1 == detail::U(EBcOp::Ret),
              "RetM falls through into Ret");

// ---------------------------------------------------------------------------
// Per-opcode metadata, built at compile time from the registry.
// ---------------------------------------------------------------------------

enum class EBcOperandMode : std::uint8_t {
    None, Slot, Base, Upval, Literal, SignedLit, Prim, NumConst, StrConst,
    TabConst, FuncConst, CDataConst, JumpOfs, CallCount, ResCount,
};

struct BcOpInfo_t {
    EBcOperandMode eModeA;
    EBcOperandMode eModeB;
    EBcOperandMode eModeCD;
    std::uint8_t uMetamethod;  // associated metamethod for the generic path
};

// consteval-built table indexed by EBcOp (definition in vm/Bytecode.cpp).
[[nodiscard]] const BcOpInfo_t& OpInfo(EBcOp eOp) noexcept;
[[nodiscard]] const char* OpName(EBcOp eOp) noexcept;

// ---------------------------------------------------------------------------
// C_BytecodeEmitter — the parser's back end (fe/Parser.hpp drives this).
// Owns the growing instruction/line arrays and the peephole rules that must
// see the instruction stream (KNIL merging, CAT fusion, jump threading).
// ---------------------------------------------------------------------------

class C_BytecodeEmitter {
public:
    using BcPos_t = std::uint32_t;
    static constexpr BcPos_t kNoJump = ~BcPos_t{0};

    [[nodiscard]] BcPos_t Emit(BcIns_t insCode, core::BcLine_t uLine);
    [[nodiscard]] BcPos_t EmitJump(core::BcLine_t uLine);

    // Jump-list handling: lists are threaded through the D field of the Jmp
    // instructions themselves (no side arrays).
    void AppendJump(BcPos_t& posList, BcPos_t posJump) noexcept;
    void PatchJumpsTo(BcPos_t posList, BcPos_t posTarget) noexcept;

    // Peepholes (contract: never merge across a jump target).
    void EmitNilRange(std::uint8_t uFirstSlot, std::uint8_t uCount, core::BcLine_t uLine);
    [[nodiscard]] BcPos_t CurrentPos() const noexcept { return m_uCount; }
    void MarkJumpTarget() noexcept { m_uLastTarget = m_uCount; }

private:
    BcIns_t* m_pCode = nullptr;
    core::BcLine_t* m_pLines = nullptr;
    std::uint32_t m_uCount = 0;
    std::uint32_t m_uCapacity = 0;
    BcPos_t m_uLastTarget = 0;
    BcPos_t m_uPendingJumps = kNoJump;  // jumps to patch to the next instruction
};

// ---------------------------------------------------------------------------
// Bytecode serialization (a frozen on-disk format; new version, LuaJIT-style
// concepts: ULEB128 everywhere, 33-bit int/double tagged numbers, children
// written depth-first before parents, per-proto length framing for cheap
// validation, endian-swap on load). Deterministic output is the DEFAULT.
// ---------------------------------------------------------------------------

inline constexpr std::uint8_t kBcDumpVersion = 0x80;  // private-format range

enum class EBcDumpFlag : std::uint32_t {
    None = 0,
    StripDebug = 1 << 0,
    BigEndian = 1 << 1,
    UsesFfi = 1 << 2,
};

class C_BytecodeSerializer {
public:
    using Writer_f = bool (*)(void* pUserData, const void* pBlock, std::size_t uBytes);

    // Writes a proto tree (lua_dump / -b equivalent).
    [[nodiscard]] bool Dump(const C_GcProto* pRoot, EBcDumpFlag eFlags,
                            Writer_f fnWriter, void* pUserData);

    // Reads a dump into protos; validates header flags (frame-layout and
    // version mismatches are rejected, endianness is converted).
    [[nodiscard]] C_GcProto* Read(class C_Universe& uni, const std::uint8_t* pData,
                                  std::size_t uBytes);
};

}  // namespace ljx::vm
