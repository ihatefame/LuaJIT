// LJX — fundamental scalar types, base concepts, and bit utilities.
// Layer 0.
//
// Naming (normative, see ARCHITECTURE.md §2):
//   C_Name   class          Name_t  struct/POD      EName  enum class
//   IsName   concept        Name_f  function type   kName  constexpr constant
//   m_/s_/g_ member/static/global prefixes; Hungarian tags on all variables:
//   n int, u unsigned, fl float/double, b bool, s string, p pointer, e enum,
//   fn callable, it iterator, v container, by byte, h handle, tv TValue,
//   r ref-id (IrRef_t / GcRef_t).
#pragma once

#include <bit>
#include <concepts>
#include <cstdint>
#include <type_traits>

#include "ljx/core/Config.hpp"

namespace ljx::core {

// Bytecode/IR-visible scalar aliases (fixed widths are part of on-disk and
// in-trace formats; never use platform-varying types in frozen layouts).
using BcLine_t = std::uint32_t;   // source line number
using StrId_t  = std::uint32_t;   // dense string-interning id (table hashing key)
using StrHash_t = std::uint32_t;  // content hash (interning chains only)
using GcSize_t = std::uint64_t;   // heap accounting

// ---------------------------------------------------------------------------
// Concepts gating frozen layouts.
//
// Every structure the interpreter, JIT, or GC addresses by raw offset must be
// trivially copyable and standard-layout: no vtables, no RTTI, bit_cast-able.
// ---------------------------------------------------------------------------

template <typename TType>
concept IsFrozenLayout =
    std::is_trivially_copyable_v<TType> && std::is_standard_layout_v<TType>;

template <typename TType>
concept IsBitCastableTo64 = IsFrozenLayout<TType> && sizeof(TType) == 8;

// ---------------------------------------------------------------------------
// Bit utilities (replacing lj_ffs/lj_fls/lj_rol hand-rolled variants).
// ---------------------------------------------------------------------------

[[nodiscard]] constexpr unsigned FindFirstSet(std::uint64_t uMask) noexcept {
    return static_cast<unsigned>(std::countr_zero(uMask));
}

[[nodiscard]] constexpr unsigned FindLastSet(std::uint64_t uMask) noexcept {
    return 63u - static_cast<unsigned>(std::countl_zero(uMask));
}

[[nodiscard]] constexpr std::uint32_t RotL32(std::uint32_t uValue, int nShift) noexcept {
    return std::rotl(uValue, nShift);
}

// ---------------------------------------------------------------------------
// Floating-point ↔ integer conversion contract.
//
// The VM's semantics on out-of-range / NaN inputs are observable (e.g. the
// array-bounds check exploits the cvttsd2si INDEFINITE 0x80000000 result) and
// such conversions are UB in C++. Interpreter, runtime, and JIT-generated code
// MUST all route through these named kernels — implemented with intrinsics
// (SSE2 cvttsd2si / AArch64 fcvtzs), never a plain cast.
// ---------------------------------------------------------------------------

// Truncating conversion; returns 0x80000000 (INDEFINITE) on NaN/overflow,
// matching hardware. Bit-identical across interpreter/runtime/traces.
[[nodiscard]] std::int32_t NumToInt32Trunc(double flValue) noexcept;

// Checked conversion: returns false unless flValue is exactly representable.
[[nodiscard]] bool NumToInt32Check(double flValue, std::int32_t& nOut) noexcept;

[[nodiscard]] std::int64_t NumToInt64Trunc(double flValue) noexcept;
[[nodiscard]] std::uint64_t NumToUint64Trunc(double flValue) noexcept;

// BitOp semantics: wrap-around double→int32 via the 2^52+2^51 magic-add.
[[nodiscard]] std::int32_t NumToBit(double flValue) noexcept;

// ---------------------------------------------------------------------------
// PRNG (one per universe; seeds string hashing, StrID reseeding, allocator
// placement, and penalty randomization — secure seeding is a hard startup
// requirement, as in LuaJIT).
// ---------------------------------------------------------------------------

struct PrngState_t {
    std::uint64_t vState[4];
};

class C_Prng {
public:
    // Returns false if the OS entropy source fails; the VM refuses to start.
    [[nodiscard]] bool SeedSecure() noexcept;
    [[nodiscard]] std::uint64_t Next() noexcept;
    [[nodiscard]] double NextDouble() noexcept;  // [0,1) with 52-bit mantissa

private:
    PrngState_t m_State{};
};

}  // namespace ljx::core
