// LJX — embedding API.
// Layer 7.
//
// C_LuaEngine is the modern C++ facade. A separate compatibility layer
// (api/Compat51.hpp, not yet designed here) exports the Lua 5.1 C API +
// LuaJIT extensions with ABI-identical semantics: GLOBALSINDEX/ENVIRONINDEX
// pseudo-indices (with two scratch slots, defusing LuaJIT's single-tmptv
// aliasing trap), lua_cpcall, luaJIT_setmode, bytecode dump/load, the shared
// nil sentinel LUA_TNONE convention, and ≥ LUA_MINSTACK growable slots for C
// functions. FR2 frame internals stay invisible to embedders.
#pragma once

#include <cstddef>
#include <cstdint>
#include <string_view>

#include "ljx/vm/Interpreter.hpp"

namespace ljx::api {

enum class EStatus : std::int32_t {
    Ok = 0, Yield, RuntimeError, SyntaxError, OutOfMemory, ErrorInHandler,
};

enum class EJitMode : std::uint8_t { Off, On, Flush };

class C_LuaEngine {
public:
    struct Options_t {
        // Capped by core::kMaxArenaReserve (32 GB): the 32-bit granule-scaled
        // compressed-ref width is a hard contract (core/Memory.hpp).
        std::size_t uArenaReserveBytes = core::kMaxArenaReserve;
        bool bJitEnabled = true;
        bool bGenerationalGc = false;
    };

    [[nodiscard]] static C_LuaEngine* Create(const Options_t& options) noexcept;
    void Destroy() noexcept;

    // --- loading & execution ------------------------------------------------
    [[nodiscard]] EStatus LoadString(std::string_view svSource, std::string_view svChunkName);
    [[nodiscard]] EStatus LoadBytecodeDump(std::string_view svDump);
    [[nodiscard]] EStatus ProtectedCall(std::int32_t nArgs, std::int32_t nResults);

    // --- typed stack access (subset shown; mirrors the classic discipline) --
    void PushNil();
    void PushNumber(double flValue);
    void PushString(std::string_view svValue);
    void PushCFunction(vm::CFunction_f fnFunction);
    [[nodiscard]] double ToNumber(std::int32_t nIndex) const noexcept;
    [[nodiscard]] std::string_view ToString(std::int32_t nIndex) noexcept;
    [[nodiscard]] bool IsNil(std::int32_t nIndex) const noexcept;
    void SetGlobal(std::string_view svName);
    [[nodiscard]] std::int32_t Top() const noexcept;
    void SetTop(std::int32_t nTop);

    // --- engine control -----------------------------------------------------
    void SetJitMode(EJitMode eMode) noexcept;
    void CollectGarbage() noexcept;
    [[nodiscard]] vm::C_Universe& Universe() noexcept { return *m_pUniverse; }
    [[nodiscard]] vm::C_LuaThread* MainThread() noexcept;

private:
    vm::C_Universe* m_pUniverse = nullptr;
};

}  // namespace ljx::api
