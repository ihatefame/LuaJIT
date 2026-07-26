// LJX — metamethod resolution.
// Layer 2.
//
// The fast path is ABSENCE: table loads consult the metatable only on the nil
// edge, and then only test one bit of the per-table negative cache before any
// lookup. Metamethod names are interned once and pinned; lookup is pointer
// equality. Base-type metatables live in a small array indexed by value tag.
#pragma once

#include <cstddef>
#include <cstdint>

#include "ljx/vm/Object.hpp"

namespace ljx::rt {

// ORDER MM (frozen): the first kNegativeCacheableCount metamethods are
// negative-cacheable in C_GcTable's uNoMm byte (bit i set = metamethod i
// definitely absent). The cacheable set is the first SIX — Index..Len,
// matching LuaJIT's MM_FAST cutoff; the uNoMm byte has capacity for 8 bits,
// leaving two spare.
enum class EMetaMethod : std::uint8_t {
    Index, NewIndex, Gc, Mode, Eq, Len,          // ← negative-cacheable set
    Lt, Le, Concat, Call, Add, Sub, Mul, Div, Mod, Pow, Unm, ToString,
    Metatable,   // __metatable protection (getmetatable/setmetatable)
    New,         // __new (ffi.metatype constructors)
    Count_,
};

inline constexpr std::uint8_t kNegativeCacheableCount = 6;
inline constexpr std::uint8_t kNegativeCacheCapacity = 8;   // uNoMm byte width

static_assert(static_cast<std::uint8_t>(EMetaMethod::Len) == kNegativeCacheableCount - 1,
              "Index..Len are exactly the negative-cacheable set");
static_assert(kNegativeCacheableCount <= kNegativeCacheCapacity);

class C_MetaResolver {
public:
    C_MetaResolver() noexcept = default;
    void Init(vm::C_Universe& uni);   // interns and pins metamethod names

    // Fast absence check + cached lookup. Returns nullptr on definite absence
    // (negative-cache hit costs one byte test on the metatable header).
    [[nodiscard]] const vm::TValue_t* Lookup(vm::C_GcTable* pMetatable,
                                             EMetaMethod eMethod) noexcept;

    // Metamethod for any value (base-type metatables by tag for non-tables).
    [[nodiscard]] const vm::TValue_t* LookupForValue(const vm::TValue_t& tvValue,
                                                     EMetaMethod eMethod) noexcept;

    // The interned name string for a metamethod (pinned at init).
    [[nodiscard]] vm::C_GcString* Name(EMetaMethod eMethod) const noexcept;

    // Invalidation: ANY key store into a table clears its whole uNoMm byte
    // (the table might be someone's metatable). Called from table set paths.
    static void InvalidateNegativeCache(vm::C_GcTable* pTable) noexcept;

    // Base-type metatables (strings get the string library via __index).
    void SetBaseMetatable(vm::EValueTag eTag, vm::C_GcTable* pMetatable) noexcept;
    [[nodiscard]] vm::C_GcTable* BaseMetatable(vm::EValueTag eTag) noexcept;

private:
    vm::C_Universe* m_pUniverse = nullptr;
    vm::C_GcString* m_vNames[static_cast<std::size_t>(EMetaMethod::Count_)]{};
    core::GcRef_t m_vBaseMetatables[16]{};  // indexed by ~tag
};

}  // namespace ljx::rt
