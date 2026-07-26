// LJX — global string interning.
// Layer 2.
//
// Contracts kept from LuaJIT (ARCHITECTURE.md §8):
//  * strings are immutable, interned, unique — POINTER equality is string
//    equality VM-wide; table hashing uses the dense reseedable StrId, never
//    the content hash (HREFK soundness);
//  * O(1) sparse-sampling keyed hash for interning (constant time even for
//    huge strings), per-chain escalation to a keyed dense hash when a chain
//    exceeds kMaxChainCollisions (hash-flood defense);
//  * dead-but-interned strings are resurrected by flipping the white bit;
//  * the empty string is embedded in the universe (never allocated).
// Modernized: dense hash is hardware CRC32C/AES-based; comparisons are SIMD
// (legal because payloads are zero-padded to 16 bytes).
#pragma once

#include <cstdint>
#include <string_view>

#include "ljx/core/Types.hpp"
#include "ljx/vm/Object.hpp"

namespace ljx::rt {

enum class EStringHashAlgo : std::uint8_t {
    Sparse,   // keyed O(1) sampling (default)
    Dense,    // keyed full hash (collision-attacked chains only)
};

class C_StringInterner {
public:
    static constexpr std::uint32_t kMinTableSize = 256;        // power of two
    static constexpr std::uint32_t kMaxChainCollisions = 32;   // dense-hash trigger

    C_StringInterner() noexcept = default;
    void Init(vm::C_Universe& uni);   // allocates chains, interns ""

    // THE hot entry: intern (or resurrect) a byte string.
    [[nodiscard]] vm::C_GcString* Intern(std::string_view svBytes);
    [[nodiscard]] vm::C_GcString* Empty() const noexcept { return m_pEmpty; }

    // GC integration (v1: stop-the-world sweep over all chains).
    void SweepAll() noexcept;
    void MaybeResize();          // grow at 100% load

    [[nodiscard]] std::uint32_t Count() const noexcept { return m_uCount; }
    [[nodiscard]] std::uint32_t ChainCount() const noexcept { return m_uMask + 1; }

private:
    [[nodiscard]] core::StrHash_t HashSparse(std::string_view svBytes) const noexcept;

    vm::C_Universe* m_pUniverse = nullptr;
    core::GcRef_t* m_pChains = nullptr;
    std::uint32_t m_uMask = kMinTableSize - 1;
    std::uint32_t m_uCount = 0;
    std::uint64_t m_uSeed = 0;      // from the universe PRNG (secure)
    core::StrId_t m_uNextSid = 0;
    vm::C_GcString* m_pEmpty = nullptr;
};

}  // namespace ljx::rt
