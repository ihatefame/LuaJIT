// LJX — platform configuration, target selection, and compiler attributes.
// Layer 0: depends on nothing.
//
// LJX is 64-bit little-endian only (see ARCHITECTURE.md §1): GC64-style value
// encoding and FR2 two-slot frames are unconditional, which removes the
// combinatorial build-mode space of LuaJIT (non-GC64/non-FR2/softfp/x87).
#pragma once

#include <cstddef>
#include <cstdint>

namespace ljx::core {

enum class EArch : std::uint8_t {
    X64,
    Arm64,
};

enum class EOs : std::uint8_t {
    Linux,
    Darwin,
    Windows,
    OtherPosix,
};

#if defined(__x86_64__) || defined(_M_X64)
inline constexpr EArch kArch = EArch::X64;
#elif defined(__aarch64__) || defined(_M_ARM64)
inline constexpr EArch kArch = EArch::Arm64;
#else
#error "LJX targets x86-64 and AArch64 only."
#endif

#if defined(__linux__)
inline constexpr EOs kOs = EOs::Linux;
#elif defined(__APPLE__)
inline constexpr EOs kOs = EOs::Darwin;
#elif defined(_WIN32)
inline constexpr EOs kOs = EOs::Windows;
#else
inline constexpr EOs kOs = EOs::OtherPosix;
#endif

// ---------------------------------------------------------------------------
// Hot-path attributes.
//
// LJX_MUSTTAIL is load-bearing: the CPS interpreter (vm/Interpreter.hpp) is
// only competitive with the hand-written assembly VM if every dispatch is a
// guaranteed tail call. Building the interpreter without it selects the
// computed-goto fallback kernel (slower; see ARCHITECTURE.md §6).
// ---------------------------------------------------------------------------

#if defined(__clang__) && __has_cpp_attribute(clang::musttail)
#define LJX_MUSTTAIL [[clang::musttail]]
#define LJX_HAS_MUSTTAIL 1
#else
#define LJX_MUSTTAIL
#define LJX_HAS_MUSTTAIL 0
#endif

// preserve_none frees nearly all registers for the interpreter's pinned state
// (the modern replacement for DynASM's fixed register file).
#if defined(__clang__) && __has_attribute(preserve_none)
#define LJX_PRESERVE_NONE __attribute__((preserve_none))
#else
#define LJX_PRESERVE_NONE
#endif

#if defined(_MSC_VER) && !defined(__clang__)
#define LJX_NOINLINE __declspec(noinline)
#define LJX_FORCEINLINE __forceinline
#else
#define LJX_NOINLINE [[gnu::noinline]]
#define LJX_FORCEINLINE [[gnu::always_inline]] inline
#endif

// ---------------------------------------------------------------------------
// Memory-system constants.
// ---------------------------------------------------------------------------

inline constexpr std::size_t kCacheLineSize = 64;

// Granule of C_VirtualArena reservations; 2 MB alignment makes the GC heap and
// the C_Universe transparently huge-page-backed on Linux.
inline constexpr std::size_t kArenaGranule = std::size_t{2} << 20;

// The NaN-boxing payload width. C_VirtualArena guarantees every GC address
// fits, making the 47-bit assumption a contract instead of a hope (survives
// 5-level paging / LAM hosts).
inline constexpr unsigned kBoxedPointerBits = 47;

}  // namespace ljx::core
