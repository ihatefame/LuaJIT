// LJX — the universal 8-byte NaN-boxed value.
// Layer 1 (frozen layout; the single most load-bearing format in the VM).
//
// Encoding (GC64-style, unconditional — ARCHITECTURE.md §4.1):
//
//   |----------------------- 64 bit ------------------------|
//   | 1…1 (13 bits) | tag (4 bits) |     payload (47 bits)  |   boxed values
//   |                IEEE-754 double (verbatim)             |   numbers
//
// Tags are complements of small integers, totally ordered, primitives with
// all-ones payloads:
//   * one unsigned compare classifies {double | int | primitive | GC object}
//     (extract tag with an arithmetic shift by 47; doubles sort below all tags)
//   * nil is the all-ones word: tested/stored with a sign-extended imm32 (-1)
//   * false = ~(1<<47), true = ~(2<<47): synthesizable in two ALU ops
//   * a table key compare is ONE 64-bit compare (type + identity fused)
// The tag ORDER is load-bearing for dozens of range predicates and is pinned
// by static_asserts below. Reordering EValueTag is a silent miscompile; don't.
#pragma once

#include <bit>
#include <cstdint>

#include "ljx/core/Types.hpp"

namespace ljx::vm {

// Tag values are the 32-bit complements of their rank (mirrors ORDER LJ_T).
enum class EValueTag : std::uint32_t {
    Nil        = ~0u,
    False      = ~1u,
    True       = ~2u,
    LightUd    = ~3u,   // segmented: 8-bit segment + 39-bit offset
    String     = ~4u,
    UpValue    = ~5u,
    Thread     = ~6u,
    Proto      = ~7u,
    Function   = ~8u,
    Trace      = ~9u,
    CData      = ~10u,
    Table      = ~11u,
    UserData   = ~12u,
    NumInt     = ~13u,  // dual-number integer; any smaller "tag" is a double
};

inline constexpr std::uint64_t kPayloadMask = (std::uint64_t{1} << 47) - 1;

struct TValue_t {
    std::uint64_t uRaw = ~std::uint64_t{0};  // nil

    // ---- constructors (constexpr: constants are foldable at compile time) --
    [[nodiscard]] static constexpr TValue_t Nil() noexcept { return {~std::uint64_t{0}}; }

    [[nodiscard]] static constexpr TValue_t Boolean(bool bValue) noexcept {
        return {~(std::uint64_t{bValue ? 2u : 1u} << 47)};
    }

    [[nodiscard]] static constexpr TValue_t Number(double flValue) noexcept {
        return {std::bit_cast<std::uint64_t>(flValue)};
    }

    [[nodiscard]] static constexpr TValue_t Integer(std::int32_t nValue) noexcept {
        return {Box(EValueTag::NumInt, static_cast<std::uint32_t>(nValue))};
    }

    // pGcObject must satisfy the arena contract (address < 2^47).
    [[nodiscard]] static TValue_t GcObject(EValueTag eTag, const void* pGcObject) noexcept {
        return {Box(eTag, reinterpret_cast<std::uintptr_t>(pGcObject))};
    }

    [[nodiscard]] static constexpr std::uint64_t Box(EValueTag eTag,
                                                     std::uint64_t uPayload) noexcept {
        return (static_cast<std::uint64_t>(static_cast<std::uint32_t>(eTag)) << 47) |
               (uPayload & kPayloadMask);
    }

    // ---- classification ----------------------------------------------------
    // Extracted "tag" of a double sorts strictly below every real tag, so all
    // predicates below are single unsigned compares after one arithmetic shift.
    [[nodiscard]] constexpr std::uint32_t TagBits() const noexcept {
        return static_cast<std::uint32_t>(
            static_cast<std::int64_t>(uRaw) >> 47);
    }

    [[nodiscard]] constexpr bool IsNil() const noexcept { return uRaw == ~std::uint64_t{0}; }
    [[nodiscard]] constexpr bool IsDouble() const noexcept {
        return TagBits() < static_cast<std::uint32_t>(EValueTag::NumInt);
    }
    [[nodiscard]] constexpr bool IsInteger() const noexcept {
        return TagBits() == static_cast<std::uint32_t>(EValueTag::NumInt);
    }
    [[nodiscard]] constexpr bool IsNumber() const noexcept {
        return TagBits() <= static_cast<std::uint32_t>(EValueTag::NumInt);
    }
    [[nodiscard]] constexpr bool IsTruthy() const noexcept {
        return TagBits() < static_cast<std::uint32_t>(EValueTag::False);
    }
    [[nodiscard]] constexpr bool IsGcObject() const noexcept {
        const std::uint32_t uTag = TagBits();
        return uTag >= static_cast<std::uint32_t>(EValueTag::UserData) &&
               uTag <= static_cast<std::uint32_t>(EValueTag::String);
    }
    [[nodiscard]] constexpr bool Is(EValueTag eTag) const noexcept {
        return TagBits() == static_cast<std::uint32_t>(eTag);
    }

    // ---- extraction --------------------------------------------------------
    [[nodiscard]] constexpr double AsDouble() const noexcept {
        return std::bit_cast<double>(uRaw);
    }
    [[nodiscard]] constexpr std::int32_t AsInteger() const noexcept {
        return static_cast<std::int32_t>(uRaw);
    }
    [[nodiscard]] void* AsGcPointer() const noexcept {  // strip tag: 2 ALU ops
        return reinterpret_cast<void*>(uRaw & kPayloadMask);
    }

    [[nodiscard]] constexpr bool operator==(const TValue_t&) const noexcept = default;
};

static_assert(core::IsBitCastableTo64<TValue_t>, "TValue_t must stay one 64-bit word");
static_assert(alignof(TValue_t) == 8);

// ---- ORDER pin (any edit to EValueTag must keep these true) ----------------
static_assert(static_cast<std::uint32_t>(EValueTag::NumInt) <
                  static_cast<std::uint32_t>(EValueTag::UserData),
              "numbers must sort below all GC tags");
static_assert(static_cast<std::uint32_t>(EValueTag::Table) <
                  static_cast<std::uint32_t>(EValueTag::String),
              "table/userdata lowest among GC types (range tests)");
static_assert(TValue_t::Nil().uRaw == ~std::uint64_t{0}, "nil must be the all-ones word");
static_assert(TValue_t::Boolean(false).uRaw == ~(std::uint64_t{1} << 47));
static_assert(TValue_t::Boolean(true).uRaw == ~(std::uint64_t{2} << 47));

}  // namespace ljx::vm
