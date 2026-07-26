// LJX — trace IR: 64-bit instructions, biased 16-bit refs, interned constants.
// Layer 5 (frozen format; the compiler's central data structure).
//
// IrIns_t is EXACTLY 64 bits. The reference scheme is LuaJIT's, kept because
// one magnitude compare implements four unrelated mechanisms:
//   * is-constant (constants grow DOWN from kRefBias, instructions UP),
//   * DCE operand marking (operand >= kRefBias),
//   * loop invariant/variant classification (ref < loop boundary),
//   * regalloc eviction priority (lowest ref = constant = free to remat).
// Constants are interned per opcode → reference equality IS value equality
// (CSE, alias analysis, snapshot replay, and sunk-store replay all rely on
// this). The rPrev field is the per-opcode CSE chain link before register
// allocation and the (register, spill) pair after — expose both phases as
// named accessors, never as separate storage (the 8-byte cell keeps fold's
// working set on one cache line; SoA was measured against and rejected).
#pragma once

#include <bit>
#include <cstddef>
#include <cstdint>

#include "ljx/core/Types.hpp"

namespace ljx::jit {

// 16-bit biased instruction reference.
using IrRef_t = std::uint16_t;
inline constexpr IrRef_t kRefBias = 0x8000;
inline constexpr IrRef_t kRefTrue = kRefBias - 3;
inline constexpr IrRef_t kRefFalse = kRefBias - 2;
inline constexpr IrRef_t kRefNil = kRefBias - 1;
inline constexpr IrRef_t kRefBase = kRefBias;      // the BASE instruction
inline constexpr IrRef_t kRefFirst = kRefBias + 1;
inline constexpr IrRef_t kRefDrop = 0xffff;

[[nodiscard]] constexpr bool IsConstRef(IrRef_t rRef) noexcept { return rRef < kRefBias; }

// Illustrative opcode subset; the full registry mirrors LuaJIT's IR with the
// same ORDER-pinning (comparison XOR algebra, load→store delta, CallN..CArg
// contiguity), enforced by the static_asserts at the bottom.
#define LJX_IR_REGISTRY(X)                                                   \
    /* guarded comparisons (XOR algebra pinned) */                           \
    X(Lt) X(Ge) X(Le) X(Gt) X(ULt) X(UGe) X(ULe) X(UGt) X(Eq) X(Ne)          \
    /* constants (two-slot 64-bit payloads for KNum/KInt64/KGc) */           \
    X(KPri) X(KInt) X(KGc) X(KPtr) X(KNull) X(KNum) X(KInt64) X(KSlot)       \
    /* guards & specialization */                                            \
    X(SLoad) X(ALoad) X(HLoad) X(ULoad) X(FLoad) X(XLoad)                    \
    /* refs */                                                               \
    X(ARef) X(HRefK) X(HRef) X(NewRef) X(URefO) X(URefC) X(StrRef)           \
    /* stores (must sit at constant delta from their loads) */               \
    X(AStore) X(HStore) X(UStore) X(FStore) X(XStore)                        \
    /* arithmetic */                                                         \
    X(Add) X(Sub) X(Mul) X(Div) X(Mod) X(Pow) X(Neg) X(Abs)                  \
    X(AddOv) X(SubOv) X(MulOv)                                               \
    X(BAnd) X(BOr) X(BXor) X(BNot) X(BShl) X(BShr) X(BSar) X(BRol) X(BRor)   \
    X(Conv) X(ToBit) X(ToStr) X(StrTo)                                       \
    /* allocation (every alloc/load/store MUST have an any/any fold rule) */ \
    X(SNew) X(XSNew) X(TNew) X(TDup) X(CNew) X(CNewI)                        \
    /* barriers & misc */                                                    \
    X(TBar) X(OBar) X(GcStep)                                                \
    /* calls (CallN..CArg contiguity pinned) */                              \
    X(CallN) X(CallA) X(CallL) X(CallS) X(CallXS) X(CArg)                    \
    /* control */                                                            \
    X(Base) X(Phi) X(Loop) X(Rename) X(Nop)

enum class EIrOp : std::uint8_t {
#define LJX_IR_ENUM(name) name,
    LJX_IR_REGISTRY(LJX_IR_ENUM)
#undef LJX_IR_ENUM
    Count_,
};

// IR types: low 5 bits type id; high 3 bits flags packed in the same byte.
enum class EIrType : std::uint8_t {
    Nil, False, True, LightUd, Str, P32, Thread, Proto, Func, P64, CData,
    Tab, UData, Float, Num, I8, U8, I16, U16, Int, U32, I64, U64, Soft,
    Count_,
};

inline constexpr std::uint8_t kIrTypeMask = 0x1f;
inline constexpr std::uint8_t kIrFlagMark = 0x20;
inline constexpr std::uint8_t kIrFlagPhi = 0x40;
inline constexpr std::uint8_t kIrFlagGuard = 0x80;

// Operand-mode metadata byte per opcode (drives generic fold/loop/sink/asm
// code): 2+2 bits operand modes, kind bits (normal/alloc/load/store),
// commutative bit, non-weak-guard bit. Built consteval from the registry.
enum class EIrMode : std::uint8_t {};
[[nodiscard]] EIrMode ModeOf(EIrOp eOp) noexcept;

struct IrIns_t {
    // View 1: operands + opcode/type + chain (emission & CSE phase).
    IrRef_t rOp1 = 0;
    IrRef_t rOp2 = 0;
    std::uint16_t uOpAndType = 0;  // low byte: EIrOp; high byte: type+flags
    IrRef_t rPrev = 0;             // CSE chain before RA; (reg,spill) after

    // ---- accessors -------------------------------------------------------
    [[nodiscard]] constexpr EIrOp Op() const noexcept {
        return static_cast<EIrOp>(uOpAndType & 0xff);
    }
    [[nodiscard]] constexpr std::uint8_t TypeBits() const noexcept {
        return static_cast<std::uint8_t>(uOpAndType >> 8);
    }
    [[nodiscard]] constexpr EIrType Type() const noexcept {
        return static_cast<EIrType>(TypeBits() & kIrTypeMask);
    }
    [[nodiscard]] constexpr bool IsGuard() const noexcept {
        return (TypeBits() & kIrFlagGuard) != 0;
    }
    [[nodiscard]] constexpr bool IsPhi() const noexcept {
        return (TypeBits() & kIrFlagPhi) != 0;
    }

    // Both operands as ONE 32-bit word — the single-compare CSE hit test.
    [[nodiscard]] constexpr std::uint32_t Op12() const noexcept {
        return static_cast<std::uint32_t>(rOp1) |
               (static_cast<std::uint32_t>(rOp2) << 16);
    }

    // Pre-regalloc view of rPrev: the per-opcode CSE chain link.
    [[nodiscard]] constexpr IrRef_t ChainPrev() const noexcept { return rPrev; }

    // Post-regalloc views of rPrev.
    [[nodiscard]] constexpr std::uint8_t AllocatedReg() const noexcept {
        return static_cast<std::uint8_t>(rPrev & 0xff);
    }
    [[nodiscard]] constexpr std::uint8_t SpillSlot() const noexcept {
        return static_cast<std::uint8_t>(rPrev >> 8);
    }
};
static_assert(core::IsBitCastableTo64<IrIns_t>, "IrIns_t must stay exactly 64 bits");

// 64-bit constant payloads occupy the FOLLOWING slot as a raw value.
union IrConstPayload_t {
    double flNumber;
    std::int64_t nInt64;
    std::uint64_t uRaw;
};

// TRef_t — the recorder's tagged reference: IR type in the top byte + flags +
// 16-bit ref. Flag values ARE the snapshot-entry flag values so snapshot
// entries are built by masking a TRef (asserted in jit/Snapshot.hpp).
inline constexpr std::uint32_t kTRefFlagFrame = 0x010000;
inline constexpr std::uint32_t kTRefFlagCont = 0x020000;
inline constexpr std::uint32_t kTRefFlagKeyIndex = 0x100000;

struct TRef_t {
    std::uint32_t uRaw = 0;

    [[nodiscard]] constexpr IrRef_t Ref() const noexcept {
        return static_cast<IrRef_t>(uRaw);
    }
    [[nodiscard]] constexpr EIrType Type() const noexcept {
        return static_cast<EIrType>((uRaw >> 24) & kIrTypeMask);
    }
    [[nodiscard]] constexpr bool IsFrame() const noexcept {
        return (uRaw & kTRefFlagFrame) != 0;
    }
    [[nodiscard]] constexpr bool IsContinuation() const noexcept {
        return (uRaw & kTRefFlagCont) != 0;
    }
    [[nodiscard]] constexpr bool IsKeyIndex() const noexcept {
        return (uRaw & kTRefFlagKeyIndex) != 0;
    }
};
static_assert(core::IsFrozenLayout<TRef_t> && sizeof(TRef_t) == 4);

// ---------------------------------------------------------------------------
// C_IrBuffer — the bidirectional instruction buffer.
//
// Biased base pointer: buffer[ref] addresses instructions (ref >= kRefBias)
// and constants (ref < kRefBias) uniformly. Growth doubles the top or shifts/
// reallocates the bottom, preserving the bias. Also owns the per-opcode CSE
// chain heads (search bounded by max(op1, op2)) and constant interning.
// ---------------------------------------------------------------------------

class C_IrBuffer {
public:
    [[nodiscard]] IrIns_t& At(IrRef_t rRef) noexcept { return m_pBiased[rRef]; }
    [[nodiscard]] const IrIns_t& At(IrRef_t rRef) const noexcept { return m_pBiased[rRef]; }

    [[nodiscard]] IrRef_t InstructionCount() const noexcept { return m_rTop; }
    [[nodiscard]] IrRef_t ConstantFloor() const noexcept { return m_rBottom; }

    // Emission (fold engine calls this after rule dispatch fails to fold).
    [[nodiscard]] IrRef_t Emit(EIrOp eOp, std::uint8_t uTypeBits, IrRef_t rOp1, IrRef_t rOp2);

    // Interned constants (identical value ⇒ identical ref, load-bearing).
    [[nodiscard]] IrRef_t ConstInt(std::int32_t nValue);
    [[nodiscard]] IrRef_t ConstNum(double flValue);
    [[nodiscard]] IrRef_t ConstInt64(std::int64_t nValue);
    [[nodiscard]] IrRef_t ConstGc(const void* pObject, EIrType eType);

    // Per-opcode CSE chains.
    [[nodiscard]] IrRef_t ChainHead(EIrOp eOp) const noexcept {
        return m_vChain[static_cast<std::size_t>(eOp)];
    }
    void SetChainHead(EIrOp eOp, IrRef_t rRef) noexcept {
        m_vChain[static_cast<std::size_t>(eOp)] = rRef;
    }

    void Rollback(IrRef_t rTop) noexcept;  // abort/undo support (loop retry)

private:
    void GrowTop();
    void GrowBottom();

    IrIns_t* m_pBiased = nullptr;   // base - bias: index directly with refs
    IrRef_t m_rTop = kRefFirst;     // next instruction slot
    IrRef_t m_rBottom = kRefBias;   // lowest interned constant
    IrRef_t m_vChain[static_cast<std::size_t>(EIrOp::Count_)]{};
};

// ---- ORDER pins ------------------------------------------------------------
namespace detail {
constexpr std::uint8_t U(EIrOp eOp) { return static_cast<std::uint8_t>(eOp); }
}  // namespace detail

static_assert((detail::U(EIrOp::Lt) ^ 1) == detail::U(EIrOp::Ge),
              "guard inversion is opcode XOR 1");
static_assert((detail::U(EIrOp::Lt) ^ 3) == detail::U(EIrOp::Gt),
              "operand swap is opcode XOR 3");
static_assert((detail::U(EIrOp::Lt) ^ 4) == detail::U(EIrOp::ULt),
              "signed↔unsigned flip is opcode XOR 4");
static_assert(detail::U(EIrOp::Eq) + 1 == detail::U(EIrOp::Ne));
static_assert(detail::U(EIrOp::AStore) - detail::U(EIrOp::ALoad) ==
                  detail::U(EIrOp::HStore) - detail::U(EIrOp::HLoad),
              "load→store at constant delta (forwarding indexes store chains)");
static_assert(detail::U(EIrOp::CArg) - detail::U(EIrOp::CallN) == 5,
              "CallN..CArg contiguous (loop PHI-mark scanning)");

}  // namespace ljx::jit
