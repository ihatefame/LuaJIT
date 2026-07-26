// LJX — GC object model (frozen layouts).
// Layer 1.
//
// Every type here is standard-layout, vtable-free, and offset-pinned: the
// interpreter, GC, and JIT address these fields by compile-time offsets baked
// into handlers and traces. The offset-aliasing contracts LuaJIT relies on
// (metatable shared between table/userdata; env between function/userdata;
// gclist across thread/proto/function/table) are preserved and static_asserted
// at the bottom of this header.
#pragma once

#include <cstddef>
#include <cstdint>

#include "ljx/core/Memory.hpp"
#include "ljx/vm/Value.hpp"

// The Lua 5.1 compat layer defines ::lua_State as an alias of the thread
// object; declaring it at global scope here keeps CFunction_f's parameter
// THE type embedders' C functions are written against.
struct lua_State;

namespace ljx::vm {

class C_GcProto;
class C_GcTable;
class C_Universe;

enum class EGcObjectType : std::uint8_t {
    String,
    UpValue,
    Thread,
    Proto,
    Function,
    Trace,
    CData,
    Table,
    UserData,
};

// Mark-byte bits (per-type overloading of the high bits is intentional and
// documented at each use site, as in LuaJIT's `marked`).
enum class EGcMark : std::uint8_t {
    White0    = 0x01,
    White1    = 0x02,
    Black     = 0x04,
    Finalized = 0x08,   // udata/cdata; on TABLES this bit is WeakKey
    WeakKey   = 0x08,   // tables only (__mode 'k'; ephemeron clearing in atomic)
    WeakVal   = 0x10,   // tables (__mode 'v'); overloaded on cdata as has-finalizer
    Fixed     = 0x20,
    SuperFixed = 0x40,
};

// 8-byte header. The two trailing bytes are the per-type "header hole" — each
// object type packs its non-32-bit fields there (see each class).
struct GcHeader_t {
    core::GcRef_t rNextGc;   // type-specific chain (intern chain for strings)
    std::uint8_t uMarked = 0;
    EGcObjectType eType{};
    std::uint8_t uExtra1 = 0;  // per-type (see owning class)
    std::uint8_t uExtra2 = 0;  // per-type
};
static_assert(core::IsBitCastableTo64<GcHeader_t>);

// ---------------------------------------------------------------------------
// C_GcString — interned, immutable; payload follows the header inline,
// NUL-terminated and zero-padded to a 16-byte multiple (SIMD compare/hash
// contract: reads up to 15 bytes past the logical end are sanctioned).
// uExtra1 = reserved-word id (lexer); uExtra2 = hash algorithm (sparse/dense).
// ---------------------------------------------------------------------------

class C_GcString {
public:
    GcHeader_t m_Header;
    core::StrId_t m_uSid = 0;     // dense interning id — THE table-hash key
    core::StrHash_t m_uHash = 0;  // content hash (intern chains only)
    std::uint32_t m_uLength = 0;

    [[nodiscard]] const char* Data() const noexcept {
        return reinterpret_cast<const char*>(this + 1);
    }
    [[nodiscard]] std::uint32_t Length() const noexcept { return m_uLength; }

    // Total allocation size incl. NUL + 16-byte padding.
    [[nodiscard]] static constexpr std::size_t AllocSize(std::uint32_t uLength) noexcept {
        return sizeof(C_GcString) + ((uLength + 16u) & ~std::size_t{15});
    }
};

// ---------------------------------------------------------------------------
// C_GcTable — hybrid array + hash.
//
// Contracts carried over verbatim (JIT-load-bearing, see ARCHITECTURE.md §4.3):
//  * hash part uses main-position chaining with Brent's eviction;
//  * DEAD KEYS KEEP THEIR NODE until an explicit resize (HREFK stability);
//  * string keys hash by m_uSid & m_uHashMask — no mixing at lookup time;
//  * empty hash part aliases the universe's shared nil-node sentinel, so the
//    lookup path has no hmask==0 branch;
//  * arrays of <= kMaxColocatedArray slots are colocated after the table;
//  * m_uNoMm (header hole uExtra1) is the negative metamethod cache: bit i set
//    = metamethod i definitely absent; any key store clears the whole byte.
// Open addressing / Swiss tables are rejected by design (HREFK).
// ---------------------------------------------------------------------------

struct TableNode_t {
    TValue_t tvValue;   // MUST stay at offset 0 (Node* used as TValue*)
    TValue_t tvKey;
    core::MRef_t rNext;
    std::uint32_t uPad = 0;
};
static_assert(sizeof(TableNode_t) == 24);
static_assert(offsetof(TableNode_t, tvValue) == 0);

inline constexpr std::uint32_t kMaxColocatedArray = 16;

class C_GcTable {
public:
    GcHeader_t m_Header;        // uExtra1 = uNoMm cache; uExtra2 = colocation
    core::MRef_t m_rArray;      // TValue_t[m_uArraySize], keys [0, size-1]
    core::GcRef_t m_rGcList;    // (field order pins the aliasing contracts)
    core::GcRef_t m_rMetatable; // offset-aliased with C_GcUserData
    core::MRef_t m_rNodes;      // TableNode_t[m_uHashMask+1] or nil-node
    std::uint32_t m_uArraySize = 0;  // exclusive bound, 0-based
    std::uint32_t m_uHashMask = 0;   // size-1; 0 = shared nil-node
    core::MRef_t m_rFreeTop;
    // Bumped by every binding change. Inline caches record the version they
    // were filled at, so a single compare invalidates them precisely — no
    // global flush, no shape objects.
    std::uint32_t m_uVersion = 0;

    // Creation: uArraySizeHint slots in the array part (0-based exclusive),
    // 1<<uHashBits hash nodes (0 = shared nil-node). Fresh tables start with
    // uNoMm = 0xFF (empty table: every metamethod definitely absent).
    [[nodiscard]] static C_GcTable* New(C_Universe& uni, std::uint32_t uArraySizeHint,
                                        std::uint32_t uHashBits);

    // Core operations (implementations in rt/; declared here as the contract).
    [[nodiscard]] const TValue_t* Get(C_Universe& uni, const TValue_t& tvKey) const noexcept;
    [[nodiscard]] TValue_t* Set(C_Universe& uni, const TValue_t& tvKey);
    [[nodiscard]] const TValue_t* GetInt(C_Universe& uni, std::uint32_t uKey) const noexcept;
    [[nodiscard]] const TValue_t* GetStr(C_Universe& uni, const C_GcString* pKey) const noexcept;
    void Resize(C_Universe& uni, std::uint32_t uArraySize, std::uint32_t uHashBits);
    [[nodiscard]] std::uint32_t Length(C_Universe& uni) const noexcept;  // nil-border search

    // Any change to what a key maps to must go through this.
    LJX_FORCEINLINE void BumpVersion() noexcept { ++m_uVersion; }
};

// ---------------------------------------------------------------------------
// C_GcProto — ONE colocated allocation:
//   [C_GcProto][bytecode…][kgc: GcRef_t, grows down][knum: TValue_t, grows up]
//   [upvalue descriptors][compressed debug info]
// The bytecode array starts immediately after the header, so proto fields are
// reachable at fixed NEGATIVE offsets from the ENTRY PC — the PC every
// callable's dispatch field points at (function headers load constants and
// param counts this way without materializing a proto pointer). Mid-function
// PC→proto recovery goes through the frame's function slot, never the PC.
// m_rConstants points BETWEEN kgc and knum (GC constants at negative
// indices — bytecode encodes them complemented; numbers at positive indices).
// uExtra1 = numparams, uExtra2 = framesize (<= 255: bytecode ISA limit).
// ---------------------------------------------------------------------------

enum class EProtoFlag : std::uint8_t {
    HasChild   = 0x01,
    IsVararg   = 0x02,
    UsesFfi    = 0x04,
    NoJit      = 0x08,
    HasILoop   = 0x10,  // contains blacklisted (I-variant-patched) bytecode
    UsesBitOps = 0x20,
};

class C_GcProto {
public:
    GcHeader_t m_Header;              // uExtra1=numparams, uExtra2=framesize
    std::uint32_t m_uBcCount = 0;
    core::GcRef_t m_rGcList;
    core::MRef_t m_rConstants;        // middle pointer: kgc down / knum up
    core::MRef_t m_rUpvalDescs;       // uint16 descriptors
    std::uint32_t m_uGcConstCount = 0;
    std::uint32_t m_uNumConstCount = 0;
    std::uint32_t m_uTotalSize = 0;
    std::uint8_t m_uUpvalCount = 0;
    std::uint8_t m_uFlags = 0;        // EProtoFlag bits + 3-bit closure counter
    std::uint16_t m_uRootTrace = 0;   // root-trace chain anchor (16-bit id)
    core::GcRef_t m_rChunkName;       // interned chunk-name string
    core::MRef_t m_rLineInfo;         // BcLine_t[m_uBcCount] (debug; may be null)
    // Whole-function JIT state: null = not tried, 1 = permanently rejected,
    // otherwise a jit::CompiledFunc_t* valid for one specific closure.
    void* m_pNative = nullptr;
    std::uint32_t m_uJitCount = 0;
    std::uint32_t m_uPadJit = 0;
    core::MRef_t m_rInlineCache;   // InlineCache_t[m_uBcCount]
    std::uint32_t m_uPadIc = 0;

    [[nodiscard]] const std::uint32_t* Bytecode() const noexcept {
        return reinterpret_cast<const std::uint32_t*>(this + 1);
    }
    [[nodiscard]] std::uint8_t FrameSize() const noexcept { return m_Header.uExtra2; }
    [[nodiscard]] std::uint8_t ParamCount() const noexcept { return m_Header.uExtra1; }

    // Proto recovered from the ENTRY PC only (bytecode colocation contract);
    // precondition: pBc == Bytecode() of some proto.
    [[nodiscard]] static const C_GcProto* FromBytecode(const std::uint32_t* pBc) noexcept {
        return reinterpret_cast<const C_GcProto*>(
                   reinterpret_cast<const char*>(pBc)) - 1;
    }
};

// ---------------------------------------------------------------------------
// C_GcFunction — closures. Every callable carries m_pPc: Lua closures point at
// their proto's bytecode; C closures at a shared one-instruction "call C"
// bytecode; builtins at synthetic fast-function opcodes — the interpreter
// calls EVERYTHING through one dispatch on the callee's first instruction.
// uExtra1 = fast-function id (0 = Lua closure, 1 = plain C), uExtra2 = #upvals.
// ---------------------------------------------------------------------------

using CFunction_f = int (*)(struct lua_State*);

class C_GcFunction {
public:
    GcHeader_t m_Header;        // uExtra1=ffid, uExtra2=nupvalues
    core::GcRef_t m_rEnv;       // offset-aliased with C_GcUserData
    core::GcRef_t m_rGcList;
    // Uniform call dispatch target. A RAW pointer, not a compressed ref: the
    // call path would otherwise pay an arena-base load, a shift and an add on
    // every single call, and the field lands in padding that a compressed ref
    // would waste anyway (the struct size is identical either way).
    const std::uint32_t* m_pPc = nullptr;
    // Lua closure: GcRef_t upvalue pointers follow inline.
    // C closure:   CFunction_f + inline TValue_t upvalues follow.

    [[nodiscard]] bool IsLua() const noexcept { return m_Header.uExtra1 == 0; }
    [[nodiscard]] std::uint8_t UpvalCount() const noexcept { return m_Header.uExtra2; }

    [[nodiscard]] core::GcRef_t* UpvalRefs() noexcept {  // Lua closures
        return reinterpret_cast<core::GcRef_t*>(this + 1);
    }
    [[nodiscard]] CFunction_f& CFunc() noexcept {        // C closures
        return *reinterpret_cast<CFunction_f*>(this + 1);
    }
    [[nodiscard]] TValue_t* CUpvalues() noexcept {       // C closures
        return reinterpret_cast<TValue_t*>(reinterpret_cast<char*>(this + 1) +
                                           sizeof(CFunction_f));
    }

    [[nodiscard]] static constexpr std::size_t LuaAllocSize(std::uint32_t uUpvals) noexcept {
        return sizeof(C_GcFunction) + uUpvals * sizeof(core::GcRef_t);
    }
    [[nodiscard]] static constexpr std::size_t CAllocSize(std::uint32_t uUpvals) noexcept {
        return sizeof(C_GcFunction) + sizeof(CFunction_f) + uUpvals * sizeof(TValue_t);
    }
};

class C_GcUpvalue {
public:
    GcHeader_t m_Header;        // uExtra1=bClosed, uExtra2=bImmutable
    core::MRef_t m_rValue;      // -> stack slot (open) or &m_tvClosed (closed)
    std::uint32_t m_uDHash = 0; // disambiguation hash (JIT alias analysis)
    TValue_t m_tvClosed;        // also overlays open-list prev/next links
};

enum class EUserDataKind : std::uint8_t {
    Plain, IoFile, FfiClib, Buffer,
};

class C_GcUserData {
public:
    GcHeader_t m_Header;        // uExtra1 = EUserDataKind
    core::GcRef_t m_rEnv;       // offset-aliased with C_GcFunction
    std::uint32_t m_uLength = 0;
    core::GcRef_t m_rMetatable; // offset-aliased with C_GcTable
    std::uint32_t m_uAlignPad = 0;  // payload follows, 8-byte aligned

    [[nodiscard]] EUserDataKind Kind() const noexcept {
        return static_cast<EUserDataKind>(m_Header.uExtra1);
    }
};

// ---------------------------------------------------------------------------
// C_LuaThread — coroutine state. The value stack doubles as the frame chain
// (frame links stored in slots, vm/Frame.hpp); stacks reserve max VA up front
// and grow by commit, so growth never moves TValues (no base/upvalue fixups).
// ---------------------------------------------------------------------------

class C_LuaThread {
public:
    GcHeader_t m_Header;        // uExtra1=dummy ffid, uExtra2=status
    core::MRef_t m_rGlobal;     // owning universe (compressed)
    core::GcRef_t m_rGcList;
    C_Universe* m_pUniverse = nullptr;  // direct pointer (hot C-API paths)
    TValue_t* m_pBase = nullptr;   // current frame base (synced at C boundaries)
    TValue_t* m_pTop = nullptr;    // first free slot (not maintained in frames)
    TValue_t* m_pMaxStack = nullptr;
    TValue_t* m_pStack = nullptr;
    // Deepest frame extent reached since the last collection. Call frames do
    // NOT nil their temp slots (that cost shows up on every call); instead the
    // collector clears everything between the live top and this mark, so a
    // slot can never hand the marker a pointer that was already freed.
    TValue_t* m_pHighWater = nullptr;
    core::GcRef_t m_rOpenUpvals;   // address-sorted open-upvalue list
    core::GcRef_t m_rEnv;
    void* m_pCFrame = nullptr;     // C frame chain; low bits = resume/unwind flags
    std::uint32_t m_uStackSize = 0;
};

// ---- offset-aliasing contracts (single-load fast paths) --------------------
static_assert(offsetof(C_GcTable, m_rMetatable) == offsetof(C_GcUserData, m_rMetatable),
              "__eq / metatable fast path does one load for table|userdata");
static_assert(offsetof(C_GcFunction, m_rEnv) == offsetof(C_GcUserData, m_rEnv));
static_assert(offsetof(C_GcTable, m_rGcList) == offsetof(C_GcFunction, m_rGcList) &&
              offsetof(C_GcTable, m_rGcList) == offsetof(C_GcProto, m_rGcList) &&
              offsetof(C_GcTable, m_rGcList) == offsetof(C_LuaThread, m_rGcList),
              "generic gray-list threading");

// ---- frozen sizes (traces/handlers bake these offsets in) ------------------
static_assert(core::IsFrozenLayout<C_GcString> && sizeof(C_GcString) == 20);
static_assert(core::IsFrozenLayout<C_GcTable> && sizeof(C_GcTable) == 40);
static_assert(core::IsFrozenLayout<C_GcProto> && sizeof(C_GcProto) == 72);

// One cache line per bytecode site. Filled when a field lookup misses on the
// receiver and resolves through its metatable's __index — the shape of every
// method call. A hit costs two identity compares and two version compares,
// replacing a metamethod lookup plus a second table search.
struct InlineCache_t {
    core::GcRef_t rMeta;          // receiver's metatable when filled
    std::uint32_t uMetaVersion;
    core::GcRef_t rIndexTable;    // the table __index resolved to
    std::uint32_t uIndexVersion;
    TValue_t tvValue;             // what the lookup produced
};
static_assert(core::IsFrozenLayout<InlineCache_t> && sizeof(InlineCache_t) == 24);
static_assert(core::IsFrozenLayout<C_GcFunction> && sizeof(C_GcFunction) == 24);
static_assert(core::IsFrozenLayout<C_GcUpvalue> && sizeof(C_GcUpvalue) == 24);
static_assert(core::IsFrozenLayout<C_GcUserData> && sizeof(C_GcUserData) == 24);
static_assert(core::IsFrozenLayout<C_LuaThread> && sizeof(C_LuaThread) == 88);

// ---- EGcObjectType ↔ EValueTag lockstep ------------------------------------
// The dense object-type enum mirrors the value-tag complement order (LuaJIT's
// gct == ~itype identity, expressed as rank arithmetic). Every conversion and
// the GC's tag-driven dispatch depend on this correspondence; pin it fully.
namespace detail {
constexpr std::uint32_t TagRank(EValueTag eTag) {
    return ~static_cast<std::uint32_t>(eTag);  // Nil→0, …, String→4, …
}
}  // namespace detail

static_assert(static_cast<std::uint32_t>(EGcObjectType::String) ==
                  detail::TagRank(EValueTag::String) - 4 &&
              static_cast<std::uint32_t>(EGcObjectType::UpValue) ==
                  detail::TagRank(EValueTag::UpValue) - 4 &&
              static_cast<std::uint32_t>(EGcObjectType::Thread) ==
                  detail::TagRank(EValueTag::Thread) - 4 &&
              static_cast<std::uint32_t>(EGcObjectType::Proto) ==
                  detail::TagRank(EValueTag::Proto) - 4 &&
              static_cast<std::uint32_t>(EGcObjectType::Function) ==
                  detail::TagRank(EValueTag::Function) - 4 &&
              static_cast<std::uint32_t>(EGcObjectType::Trace) ==
                  detail::TagRank(EValueTag::Trace) - 4 &&
              static_cast<std::uint32_t>(EGcObjectType::CData) ==
                  detail::TagRank(EValueTag::CData) - 4 &&
              static_cast<std::uint32_t>(EGcObjectType::Table) ==
                  detail::TagRank(EValueTag::Table) - 4 &&
              static_cast<std::uint32_t>(EGcObjectType::UserData) ==
                  detail::TagRank(EValueTag::UserData) - 4,
              "EGcObjectType rank == EValueTag complement rank - 4, for every type");

[[nodiscard]] constexpr EGcObjectType TagToGcType(EValueTag eTag) noexcept {
    return static_cast<EGcObjectType>(detail::TagRank(eTag) - 4);
}
[[nodiscard]] constexpr EValueTag GcTypeToTag(EGcObjectType eType) noexcept {
    return static_cast<EValueTag>(~(static_cast<std::uint32_t>(eType) + 4));
}

}  // namespace ljx::vm
