// LJX — FFI: C type system, ABI classification, calls and callbacks.
// Layer 6.
//
// Kept from LuaJIT: the flat interned CType_t cell array with small integer
// ids (info embeds the child id, so structural equality is one integer
// compare and every cdata type check — interpreter or JIT guard — is one id
// compare); append-only with snapshot/rollback around speculative parses.
//
// Changed: cells allocate in STABLE chunks (deque-of-blocks) so CType_t*
// never invalidates (killing LuaJIT's recurring "table may reallocate" bug
// class), and ONE ABI classifier produces an ArgPlan_t consumed by all three
// clients — FFI calls, C callbacks, and the JIT's call recorder — instead of
// three per-target macro sets that must agree by discipline.
#pragma once

#include <cstdint>
#include <string_view>

#include "ljx/vm/Object.hpp"

namespace ljx::ffi {

using CTypeId_t = std::uint32_t;   // widened from LuaJIT's 16-bit (real apps hit 64K)
inline constexpr CTypeId_t kCTypeIdNone = 0;

enum class ECTypeKind : std::uint8_t {
    Num, Struct, Ptr, Array, Void, Enum, Func, TypeDef, Attrib,
    Field, BitField, ConstVal, Extern, KeyWord,
};

// One interned type cell. info packs kind + flags + child id; size carries
// bytes / arg count / field offset / enum value depending on kind.
// uInfo layout: kind[31:28] | flags[27:20] | child CTypeId[19:0] — the child
// field caps the id space at kMaxCTypeId (1M types; LuaJIT's 64K ceiling was
// reachable by FFI-heavy applications, this one is not). Structural equality
// stays a single uInfo compare.
inline constexpr CTypeId_t kMaxCTypeId = (1u << 20) - 1;

struct CType_t {
    std::uint32_t uInfo = 0;    // kind(4) | flags(8) | child id(20)
    std::uint32_t uSize = 0;
    CTypeId_t uSibling = 0;     // field / enum-const / parameter chain
    CTypeId_t uHashNext = 0;    // intern chain
    core::GcRef_t rName;
    std::uint32_t uPad = 0;

    [[nodiscard]] constexpr ECTypeKind Kind() const noexcept {
        return static_cast<ECTypeKind>(uInfo >> 28);
    }
};
static_assert(core::IsFrozenLayout<CType_t> && sizeof(CType_t) == 24);

class C_CTypeRegistry {
public:
    // Stable-address lookup (chunked storage: pointers never invalidate).
    [[nodiscard]] const CType_t& Get(CTypeId_t uId) const noexcept;

    // Interning: structurally identical (info,size) pairs return the same id.
    // Fails (throws the OOM/limit path) past kMaxCTypeId.
    [[nodiscard]] CTypeId_t Intern(std::uint32_t uInfo, std::uint32_t uSize);
    [[nodiscard]] CTypeId_t ParseDeclaration(std::string_view svDecl);  // cdef/new

    // Snapshot/rollback around speculative or aborted parses.
    struct Checkpoint_t { CTypeId_t uTop; };
    [[nodiscard]] Checkpoint_t Save() const noexcept;
    void Restore(Checkpoint_t checkpoint) noexcept;

    // Metatables / callback slots (miscmap equivalent).
    [[nodiscard]] vm::C_GcTable* MiscMap() noexcept;

private:
    static constexpr std::uint32_t kCellsPerChunk = 4096;
    CType_t** m_pChunks = nullptr;      // deque-of-blocks: stable addresses
    std::uint32_t m_uChunkCount = 0;
    CTypeId_t m_uTop = 1;               // 0 reserved for kCTypeIdNone sentinel
    CTypeId_t m_vHashAnchors[128]{};
};

// ---------------------------------------------------------------------------
// ABI classification — ONE classifier, three consumers.
// ---------------------------------------------------------------------------

enum class EArgClass : std::uint8_t {
    Gpr, Fpr, GprPair, FprPair, Stack, IndirectByPointer, HfaFloat, HfaDouble,
};

struct ArgSlot_t {
    EArgClass eClass{};
    std::uint8_t uRegOrOffset = 0;
    std::uint8_t uExtension = 0;   // zero/sign-extend width fixups (ABI-mandated)
    std::uint8_t uPad = 0;
};

// The complete placement recipe for one call signature.
struct ArgPlan_t {
    static constexpr std::uint32_t kMaxArgs = 32;
    ArgSlot_t vArgs[kMaxArgs]{};
    ArgSlot_t retSlot{};
    std::uint8_t uArgCount = 0;
    std::uint8_t uGprCount = 0, uFprCount = 0;
    std::uint8_t uStackSlots = 0;
    bool bReturnByPointer = false;
    bool bVararg = false;
};

// Per-ABI policy classes satisfy this concept; classification is pure and
// unit-tested against libffi as an oracle.
template <typename TAbi>
concept IsAbiPolicy = requires(const C_CTypeRegistry& reg, CTypeId_t uFuncType,
                               ArgPlan_t& plan) {
    { TAbi::kName } -> std::convertible_to<const char*>;
    { TAbi::Classify(reg, uFuncType, plan) } -> std::same_as<bool>;
};

class C_SysVX64Abi;    // two-eightbyte struct classification
class C_WinX64Abi;     // positional 4/4, by-value 1/2/4/8 only
class C_Arm64Abi;      // 8/8, HFA splitting, Darwin packed stack args

// ---------------------------------------------------------------------------
// Call execution: portable classification fills a register/stack image; a
// tiny per-target asm stub (~50 instructions) block-loads it and calls.
// ---------------------------------------------------------------------------

struct CCallState_t {
    void (*fnTarget)() = nullptr;
    std::uint32_t uStackAdjust = 0;
    std::uint8_t uStackSlotCount = 0;
    std::uint8_t uReturnByPointer = 0;  // byte, not bool: asm-stub-addressed field
    std::uint8_t uGprCount = 0, uFprCount = 0;
    alignas(16) std::uint64_t vFpr[8][2]{};   // vector-width FPR images
    std::uint64_t vGpr[8]{};
    std::uint64_t vStack[32]{};
};

class C_FfiCallEngine {
public:
    // Marshal Lua arguments per the plan, invoke through the asm stub,
    // convert the result back (boxing 64-bit values as cdata as needed).
    [[nodiscard]] std::int32_t Call(vm::C_LuaThread* pThread, CTypeId_t uFuncType,
                                    vm::TValue_t* pArgs, std::int32_t nArgs);
};

// ---------------------------------------------------------------------------
// Callbacks: one RX stub page per universe; slot number → (ctype, function).
// Stubs carry BTI/PAC/IBT landing pads; the universe pointer lives in an
// adjacent DATA page (stub code is position-identical and shareable, and the
// page pair works under dual-mapping W^X policies). On-trace callback entry
// is a hard abort; functions observed to re-enter Lua via callbacks are
// blacklisted for trace-compiled calls (protocol preserved from LuaJIT).
// ---------------------------------------------------------------------------

class C_CallbackBridge {
public:
    static constexpr std::uint32_t kMaxSlots = 512;

    [[nodiscard]] void* AcquireSlot(CTypeId_t uFuncType, const vm::TValue_t& tvFunc);
    void ReleaseSlot(std::uint32_t uSlot) noexcept;

    // Entry from the asm thunk: re-classify via the SAME ArgPlan_t, convert
    // register/stack images to Lua values, run the Lua function, marshal the
    // result back with the ABI-mandated extensions.
    [[nodiscard]] std::int32_t Enter(std::uint32_t uSlot, void* pRegisterImage);

    // Validates a code address as one of ours and recovers its slot.
    [[nodiscard]] std::int32_t SlotFromAddress(const void* pAddress) const noexcept;

private:
    void* m_pStubPage = nullptr;     // RX
    void* m_pDataPage = nullptr;     // RW: universe pointer + slot table
    CTypeId_t m_vSlotTypes[kMaxSlots]{};
    std::uint32_t m_uTopSlot = 0;
};

}  // namespace ljx::ffi
