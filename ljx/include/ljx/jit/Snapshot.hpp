// LJX — snapshots and deoptimization.
// Layer 5 (frozen formats).
//
// A snapshot maps IR refs back to interpreter stack slots at guard
// granularity. Kept from LuaJIT: 12-byte headers + 32-bit packed entries,
// sparse slots (unmodified slots elided), merge-when-no-guard-in-between,
// dead-slot purging via a reaching-definitions scan over the FOLLOWING
// bytecode, frame links appended after slot entries, and the mcode-offset
// field enabling binary-search unwinding from a fault address.
//
// STRUCTURAL IDENTITY: exit number == snapshot number. All guards emitted
// under one snapshot share one register/spill map; renames inside a guard
// group are killed or force a spill (Bloom-filter pair in the assembler).
#pragma once

#include <cstdint>

#include "ljx/jit/Ir.hpp"

namespace ljx::vm {
class C_Universe;
}

namespace ljx::jit {

class C_TraceRecorder;

// Packed snapshot entry: (slot << 24) | flags | ref.
// Flag values numerically EQUAL the TRef_t flag bits so entries are built by
// masking a TRef — pinned here.
struct SnapEntry_t {
    std::uint32_t uRaw = 0;

    [[nodiscard]] static constexpr SnapEntry_t Make(std::uint8_t uSlot, std::uint32_t uFlags,
                                                    IrRef_t rRef) noexcept {
        return {(std::uint32_t{uSlot} << 24) | uFlags | rRef};
    }
    [[nodiscard]] constexpr std::uint8_t Slot() const noexcept {
        return static_cast<std::uint8_t>(uRaw >> 24);
    }
    [[nodiscard]] constexpr IrRef_t Ref() const noexcept {
        return static_cast<IrRef_t>(uRaw);
    }
};

inline constexpr std::uint32_t kSnapFlagFrame = kTRefFlagFrame;
inline constexpr std::uint32_t kSnapFlagCont = kTRefFlagCont;
inline constexpr std::uint32_t kSnapFlagNoRestore = 0x040000;
inline constexpr std::uint32_t kSnapFlagKeyIndex = kTRefFlagKeyIndex;

static_assert(kSnapFlagFrame == 0x10000 && kSnapFlagCont == 0x20000 &&
                  kSnapFlagKeyIndex == 0x100000,
              "snapshot flags ARE the TRef_t flag bits (entries built by masking a TRef)");
static_assert((kSnapFlagNoRestore & (kTRefFlagFrame | kTRefFlagCont | kTRefFlagKeyIndex)) == 0,
              "snapshot-only flags must not collide with TRef bits");

struct Snapshot_t {
    std::uint32_t uMapOfs = 0;   // offset into the trace's shared entry map
    IrRef_t rFirstIns = 0;       // first IR ref belonging to this snapshot
    std::uint16_t uMcOfs = 0;    // machine-code offset (unwind binary search)
    std::uint8_t uSlotCount = 0;
    std::uint8_t uTopSlot = 0;
    std::uint8_t uEntryCount = 0;
    std::uint8_t uExitTaken = 0; // saturates; drives side-trace hotness
};
static_assert(sizeof(Snapshot_t) == 12, "snapshot headers are 12 bytes, frozen");

inline constexpr std::uint8_t kSnapshotExitDone = 0xff;  // side trace exists/blacklisted

// ---------------------------------------------------------------------------
// C_SnapshotWriter — recorder-side construction.
// ---------------------------------------------------------------------------

class C_SnapshotWriter {
public:
    // Adds (or merges into) the current snapshot for the recorder state.
    void Capture(C_TraceRecorder& rec);

    // Dead-slot purge: reaching-defs over the following bytecode + upvalue
    // closure scan; keeps snapshots byte-cheap.
    void PurgeDeadSlots(C_TraceRecorder& rec);

    [[nodiscard]] std::uint16_t Count() const noexcept { return m_uCount; }

private:
    Snapshot_t* m_pSnapshots = nullptr;
    SnapEntry_t* m_pEntryMap = nullptr;
    std::uint16_t m_uCount = 0;
    bool m_bMergePending = false;   // no IR/guard since last capture
    std::uint8_t m_uGuardsEmitted = 0;
};

// ---------------------------------------------------------------------------
// Trace exit / deopt.
//
// ExitState_t is the register image the exit handler writes: every GPR and
// vector register PLUS the trace's spill area, all at fixed per-target
// offsets — the contract between generated guard code, the exit handler, and
// snapshot restoration. Register-file extents are per-architecture constants
// (x86-64: 16/16, 32 GPRs under the APX build flavor; AArch64: 32/32) so the
// layout is frozen per target, not shared across targets.
// ---------------------------------------------------------------------------

inline constexpr std::uint32_t kNumExitGprs =
    (core::kArch == core::EArch::X64) ? 16 : 32;   // APX flavor overrides to 32
inline constexpr std::uint32_t kNumExitFprs =
    (core::kArch == core::EArch::X64) ? 16 : 32;
inline constexpr std::uint32_t kNumSpillSlots = 256;  // matches the RA's spill space

struct ExitState_t {
    double vFpr[kNumExitFprs];
    std::uint64_t vGpr[kNumExitGprs];
    std::int32_t vSpill[kNumSpillSlots];  // trace stack frame spill image
    std::uint32_t uExitNumber;            // == snapshot number
    std::uint32_t uTraceNumber;
};
static_assert(core::IsFrozenLayout<ExitState_t>,
              "exit stubs address this by raw fixed offsets");

class C_DeoptEngine {
public:
    // Restores interpreter state from a snapshot: reconstructs stack slots
    // (rematerializing sunk allocations), rebuilds frame links, re-enters the
    // interpreter. Runs under a protected frame; GC-safe.
    [[noreturn]] void RestoreAndReenter(vm::C_Universe& uni, const ExitState_t& exitState);

    // Side-trace support: replay a snapshot as the entry state of a new
    // recording (parent register map → SLoad coalescing hints).
    void ReplayIntoRecorder(C_TraceRecorder& rec, std::uint32_t uParentTrace,
                            std::uint32_t uExitNumber);
};

}  // namespace ljx::jit
