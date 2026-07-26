// LJX — call frames ON the value stack (no CallInfo array).
// Layer 1 (frozen format).
//
// FR2 two-slot frames, unconditional:
//     [base-2] = tagged function
//     [base-1] = 64-bit frame link
//     [base+0…] locals
// Continuation frames prepend [base-4]=continuation, [base-3]=saved PC.
//
// The link's low 3 bits encode the frame type; bytecode PCs are 4-byte
// aligned, so low bits 00 mean "the link IS the return PC" (a Lua frame).
// The PREVIOUS Lua frame is found by decoding the A operand of the call
// instruction before the saved PC — frame sizes live in the bytecode itself.
// This is the cheapest known pcall/frame scheme and is kept verbatim.
#pragma once

#include <cstdint>

#include "ljx/vm/Bytecode.hpp"
#include "ljx/vm/Value.hpp"

namespace ljx::vm {

enum class EFrameType : std::uint8_t {
    Lua      = 0,  // link == return PC (4-byte aligned)
    C        = 1,
    Cont     = 2,  // continuation frame (metamethod return path)
    Vararg   = 3,
    // bit patterns with the pcall group bit:
    CPCall   = 5,
    PCall    = 6,
    PCallHook = 7,
};

inline constexpr std::uint64_t kFrameTypeMask = 7;
inline constexpr std::uint64_t kFramePCallMask = 6;  // (link & 6) == PCall group

struct FrameLink_t {
    std::uint64_t uRaw = 0;

    [[nodiscard]] static FrameLink_t FromReturnPc(const BcIns_t* pPc) noexcept {
        return {reinterpret_cast<std::uintptr_t>(pPc)};
    }
    [[nodiscard]] static constexpr FrameLink_t FromDelta(std::uint64_t uByteDelta,
                                                         EFrameType eType) noexcept {
        return {(uByteDelta << 3) | static_cast<std::uint64_t>(eType)};
    }

    [[nodiscard]] constexpr EFrameType Type() const noexcept {
        return (uRaw & 3) == 0 ? EFrameType::Lua
                               : static_cast<EFrameType>(uRaw & kFrameTypeMask);
    }
    [[nodiscard]] constexpr bool IsLua() const noexcept { return (uRaw & 3) == 0; }
    [[nodiscard]] constexpr bool IsPCall() const noexcept {
        return (uRaw & kFramePCallMask) == static_cast<std::uint64_t>(EFrameType::PCall);
    }
    [[nodiscard]] const BcIns_t* ReturnPc() const noexcept {
        return reinterpret_cast<const BcIns_t*>(uRaw);
    }
    [[nodiscard]] constexpr std::uint64_t Delta() const noexcept { return uRaw >> 3; }
};
static_assert(core::IsBitCastableTo64<FrameLink_t>);

// ---------------------------------------------------------------------------
// Frame navigation (used by error unwinding, debug library, GC stack walks).
// ---------------------------------------------------------------------------

class C_FrameWalker {
public:
    // pFrameBase points at base (slots); the frame's link is pFrameBase[-1].
    [[nodiscard]] static FrameLink_t LinkOf(const TValue_t* pFrameBase) noexcept {
        return {pFrameBase[-1].uRaw};
    }
    [[nodiscard]] static const TValue_t* FunctionSlot(const TValue_t* pFrameBase) noexcept {
        return pFrameBase - 2;
    }

    // Previous frame base. Lua frames: decode bc A-operand of the call site
    // (pcRet[-1]); other frames: subtract the stored byte delta.
    [[nodiscard]] static const TValue_t* Previous(const TValue_t* pFrameBase) noexcept;
};

// Always-available headroom above L->top for metamethod frame setup, written
// WITHOUT a stack check (hard contract with the stack allocator).
inline constexpr std::uint32_t kStackExtraSlots = 8;

// Must stay < 64K so 16-bit slot deltas cover any frame distance.
inline constexpr std::uint32_t kMaxStackSlots = 65500;

}  // namespace ljx::vm
