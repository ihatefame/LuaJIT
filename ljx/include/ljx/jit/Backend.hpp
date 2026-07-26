// LJX — register allocation, machine-code backends, executable memory.
// Layer 5.
//
// The assembler is a SINGLE BACKWARD PASS fusing register allocation and code
// emission (LuaJIT's design, kept because it is structurally superior for
// linear trace IR): walking the IR in reverse gives exact liveness with no
// interval bookkeeping (a use is seen before its def; the def frees the
// register), makes DCE free, lets guards emit after operand registers are
// final, and enables address-mode fusion by peeking at not-yet-emitted
// operand instructions.
//
// Backends are compile-time plug-ins: a concept + CRTP-style template driver
// replaces lj_asm.c's textual #include composition. Division of labor is
// unchanged — the generic driver owns regalloc/snapshots/guard bookkeeping;
// the backend owns instruction selection with PREPEND-style emitters.
#pragma once

#include <bit>
#include <concepts>
#include <cstddef>
#include <cstdint>

#include "ljx/jit/Ir.hpp"
#include "ljx/jit/Snapshot.hpp"

namespace ljx::jit {

// ---------------------------------------------------------------------------
// Register sets and allocation records.
// ---------------------------------------------------------------------------

struct RegSet_t {
    std::uint64_t uBits = 0;

    [[nodiscard]] constexpr bool Contains(std::uint8_t uReg) const noexcept {
        return (uBits >> uReg) & 1;
    }
    constexpr void Add(std::uint8_t uReg) noexcept { uBits |= std::uint64_t{1} << uReg; }
    constexpr void Remove(std::uint8_t uReg) noexcept { uBits &= ~(std::uint64_t{1} << uReg); }
    [[nodiscard]] constexpr std::uint8_t PickLowest() const noexcept {
        return static_cast<std::uint8_t>(std::countr_zero(uBits));
    }
    [[nodiscard]] constexpr bool Empty() const noexcept { return uBits == 0; }
};

inline constexpr std::uint8_t kRegNone = 0x80;   // high bit: no register
inline constexpr std::uint8_t kRegSink = 0xfe;   // allocation was sunk
inline constexpr std::uint8_t kRegInit = 0xff;

// Blended eviction cost: (ref + weights) << 16 | owning ref. Reverse-order
// eviction picks the LOWEST ref — the value whose def is furthest away in the
// remaining (earlier) code; constants naturally lose and rematerialize free.
struct RegCost_t {
    std::uint32_t uRaw = 0;

    [[nodiscard]] static constexpr RegCost_t Make(IrRef_t rRef, std::uint32_t uWeight) noexcept {
        return {(static_cast<std::uint32_t>(rRef) + uWeight) << 16 | rRef};
    }
    [[nodiscard]] constexpr IrRef_t Ref() const noexcept {
        return static_cast<IrRef_t>(uRaw);
    }
};

inline constexpr std::uint32_t kPhiWeight = 64;  // PHIs are ~64 instructions stickier

// ---------------------------------------------------------------------------
// C_AsmContext — shared assembler state visible to backends.
// ---------------------------------------------------------------------------

class C_AsmContext {
public:
    // Machine-code cursor: backends PREPEND (emit backwards).
    std::uint8_t* m_pMcCursor = nullptr;
    std::uint8_t* m_pMcBottom = nullptr;   // redzone-checked sparse limit

    // Allocation state.
    RegSet_t m_FreeGpr, m_FreeFpr;
    RegSet_t m_ModifiedInLoop;             // for loop-head reconciliation
    RegSet_t m_WeakRefs;                   // snapshot-only refs (lose fights)
    RegCost_t m_vCost[64];                 // per machine register

    C_IrBuffer* m_pIr = nullptr;
    const Snapshot_t* m_pCurrentSnapshot = nullptr;
    std::uint32_t m_uCurrentSnapshotIndex = 0;

    // --- allocator services used by backends --------------------------------
    [[nodiscard]] std::uint8_t AllocRef(IrRef_t rRef, RegSet_t allowed);
    [[nodiscard]] std::uint8_t ScratchReg(RegSet_t allowed);
    void EvictReg(std::uint8_t uReg);
    void RematerializeConst(IrRef_t rRef, std::uint8_t uReg);
    // x86 two-operand fusion: propagate dest into left operand when legal.
    void HintLeft(IrRef_t rRef, std::uint8_t uReg) noexcept;

    // May this load fuse into the consumer's address mode? (Never across the
    // loop-invariant boundary; the fusion legality oracle lives here.)
    [[nodiscard]] bool MayFuse(IrRef_t rRef) const noexcept;
};

// ---------------------------------------------------------------------------
// The backend concept — the complete generic↔target contract.
// ---------------------------------------------------------------------------

enum class ECondCode : std::uint8_t {
    Eq, Ne, Lt, Ge, Le, Gt, ULt, UGe, ULe, UGt, Overflow, NoOverflow,
};

template <typename TBackend>
concept IsMachineBackend = requires(C_AsmContext& asmCtx, const IrIns_t* pIns,
                                    ECondCode eCc, std::uint32_t uExitNo) {
    // Code unit: byte stream on x86-64, 32-bit words on AArch64.
    typename TBackend::MCodeUnit_t;
    { TBackend::kName } -> std::convertible_to<const char*>;
    { TBackend::kGprAllocatable } -> std::convertible_to<RegSet_t>;
    { TBackend::kFprAllocatable } -> std::convertible_to<RegSet_t>;
    { TBackend::kExitStubSpacing } -> std::convertible_to<std::uint32_t>;

    // Instruction selection (prepend-emitters).
    { TBackend::EmitIns(asmCtx, pIns) };
    { TBackend::EmitGuard(asmCtx, eCc, uExitNo) };
    { TBackend::EmitMove(asmCtx, std::uint8_t{}, std::uint8_t{}) };
    { TBackend::EmitSpill(asmCtx, std::uint8_t{}, std::uint32_t{}) };
    { TBackend::EmitReload(asmCtx, std::uint8_t{}, std::uint32_t{}) };

    // Trace glue.
    { TBackend::EmitExitStubGroup(asmCtx, std::uint32_t{}) } -> std::same_as<std::uint8_t*>;
    { TBackend::EmitLoopBranch(asmCtx) };
    { TBackend::PatchExitBranch(asmCtx, uExitNo, static_cast<void*>(nullptr)) };

    { TBackend::CanRematerialize(IrRef_t{}) } -> std::convertible_to<bool>;
};

// ---------------------------------------------------------------------------
// C_TraceAssembler<TBackend> — the generic backward driver.
//
// Invariants owned here (all carried from LuaJIT):
//  * exitno == snapno for every guard of a snapshot; one RegSp map per
//    snapshot (renames inside a guard group are killed or force-spilled —
//    Bloom-filter pair over snapshot refs);
//  * the IR is IMMOVABLE during assembly (machine code embeds addresses of
//    constant payload slots); a mid-assembly rename overflow triggers one
//    reallocation + full re-assembly (rare, bounded);
//  * snapshot registers allocate lazily at the snapshot's FIRST guard and are
//    weak (lose eviction fights unless genuinely live);
//  * side-trace heads coalesce with the parent's exit register map via an
//    iterative move/cycle-break, so entering a side trace costs moves, not a
//    frame spill/reload.
// ---------------------------------------------------------------------------

template <IsMachineBackend TBackend>
class C_TraceAssembler {
public:
    explicit C_TraceAssembler(C_AsmContext& asmCtx) noexcept : m_pCtx(&asmCtx) {}

    // Assembles the finished IR into machine code; fills exit stubs, loop
    // branch (with fall-through inversion + alignment), snapshot mc offsets.
    // Returns false on mcode-area exhaustion (caller retries with a new area).
    [[nodiscard]] bool Assemble(class C_GcTrace& trace);

private:
    void SetupRegisterHints();        // forward pre-pass (call/ret constraints)
    void AssembleInstructionsBackward();
    void AssembleSnapshotGroup(std::uint32_t uSnapNo);
    void AssembleHead();              // root vs side-trace entry
    C_AsmContext* m_pCtx;
};

// Provided backends (definitions in their own TUs).
class C_X64Backend;    // + APX flavor: 32 GPRs, NDD 3-operand forms
class C_Arm64Backend;  // BTI landing pads, PAC-signed dispatch targets

// ---------------------------------------------------------------------------
// C_MachineCodeArena — W^X executable memory.
//
// Chained fixed-size areas; traces emitted top-down, exit-stub groups
// bottom-up. Reserve/commit transactions; protection state cached per area;
// jump-range-constrained placement (AArch64 ±128 MB) with PRNG probing.
// Optional dual-mapping (RW alias + RX alias) build mode for patch-heavy
// workloads; MAP_JIT/pthread_jit_write_protect_np on hardened macOS.
// ---------------------------------------------------------------------------

class C_MachineCodeArena {
public:
    static constexpr std::size_t kDefaultAreaSize = 64 << 10;
    static constexpr std::size_t kDefaultTotalLimit = 2048 << 10;

    [[nodiscard]] std::uint8_t* Reserve(std::size_t uBytes);   // begin transaction
    void Commit(std::uint8_t* pTop) noexcept;                  // publish + RX + isync
    void AbortReservation() noexcept;

    // Scoped writability for cross-trace patching (flips only the owning area).
    class C_PatchScope {
    public:
        C_PatchScope(C_MachineCodeArena& arena, void* pTarget) noexcept;
        ~C_PatchScope();
    };

    void FreeAll() noexcept;   // trace flush: invalidates all stub groups too

private:
    void* m_pAreaChain = nullptr;
    std::uint8_t* m_pTop = nullptr;
    std::uint8_t* m_pBottom = nullptr;
    std::uint32_t m_uProtectionState = 0;  // cached; no-op when already correct
};

}  // namespace ljx::jit
