// LJX — string buffers (concat, string.format, string.buffer objects).
// Layer 2.
//
// Four-word header, growth check = one pointer compare; flag bits hide in the
// low bits of the owner reference (zero space overhead) — LuaJIT's SBuf shape.
#pragma once

#include <cstddef>
#include <cstdint>
#include <string_view>

#include "ljx/core/Memory.hpp"

namespace ljx::rt {

enum class EStrBufFlag : std::uint8_t {
    Extended = 1,  // has read cursor / dictionaries (string.buffer object)
    CopyOnWrite = 2,
    Borrowed = 4,
};

struct StrBuf_t {
    char* pWrite = nullptr;
    char* pEnd = nullptr;
    char* pBase = nullptr;
    std::uintptr_t uOwnerAndFlags = 0;  // owner thread ref | EStrBufFlag bits

    [[nodiscard]] std::size_t Length() const noexcept {
        return static_cast<std::size_t>(pWrite - pBase);
    }
    [[nodiscard]] std::size_t Remaining() const noexcept {
        return static_cast<std::size_t>(pEnd - pWrite);
    }
};

class C_StringBuffer {
public:
    // Ensures space for uBytes; the fast path is a single compare.
    [[nodiscard]] LJX_FORCEINLINE char* Reserve(StrBuf_t& buf, std::size_t uBytes) {
        if (Remaining(buf) < uBytes) [[unlikely]] {
            GrowSlow(buf, uBytes);
        }
        return buf.pWrite;
    }
    static void Commit(StrBuf_t& buf, std::size_t uBytes) noexcept { buf.pWrite += uBytes; }
    static void Append(StrBuf_t& buf, std::string_view svBytes);
    static void Reset(StrBuf_t& buf) noexcept { buf.pWrite = buf.pBase; }
    void ShrinkTemporary(StrBuf_t& buf) noexcept;  // once per GC cycle

private:
    [[nodiscard]] static std::size_t Remaining(const StrBuf_t& buf) noexcept {
        return static_cast<std::size_t>(buf.pEnd - buf.pWrite);
    }
    LJX_NOINLINE void GrowSlow(StrBuf_t& buf, std::size_t uBytes);
};

}  // namespace ljx::rt
