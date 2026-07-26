// LJX — the trace compiler.
//
// Unlike the loop and function tiers, this one does not recognize shapes. It
// records whatever bytecodes actually execute at a hot back-edge, specializes
// on the types actually observed, and leaves through a side exit when an
// assumption breaks. That makes it indifferent to whether the code is
// arithmetic, table traffic or method dispatch.
#pragma once

#include <cstdint>
#include <unordered_map>
#include <vector>

#include "ljx/jit/TraceIr.hpp"
#include "ljx/vm/Bytecode.hpp"

namespace ljx::vm {
class C_Universe;
class C_GcProto;
class C_GcTable;
struct InlineCache_t;
}

namespace ljx::jit {

enum class ETraceState : std::uint8_t { Idle, Recording };

enum class EAbort : std::uint8_t {
    None, Unsupported, TooLong, LeftFrame, BadType, NoMemory, GuardImpossible,
};

// A finished trace. Entered from the interpreter at its start PC; returns the
// bytecode index to resume at (always through a snapshot restore).
struct Trace_t {
    void* pCode = nullptr;
    const vm::BcIns_t* pStartPc = nullptr;
    std::uint32_t uExitCount = 0;
};

// Entry ABI of compiled trace code:
//   uint32 fn(TValue_t* pBase, C_Universe* pUni)
// returns the bytecode index the interpreter must resume at.
using TraceEntry_f = std::uint32_t (*)(vm::TValue_t* pBase, vm::C_Universe* pUni);

class C_TraceJit {
public:
    static constexpr std::uint32_t kHotLoopThreshold = 56;
    static constexpr std::uint32_t kMaxRecordedIns = 400;
    static constexpr std::uint32_t kMaxIrIns = 2048;
    static constexpr std::uint32_t kMaxSlots = 250;
    static constexpr std::uint32_t kMaxInlineDepth = 6;
    static constexpr std::size_t kCodeArenaSize = 4u << 20;

    explicit C_TraceJit(vm::C_Universe& uni) noexcept : m_pUniverse(&uni) {}
    ~C_TraceJit();

    // --- interpreter hooks --------------------------------------------------
    // A hot back-edge was reached. Returns a trace to enter, or null (and may
    // start recording).
    [[nodiscard]] const Trace_t* OnLoopEdge(const vm::BcIns_t* pPc, vm::TValue_t* pBase);
    // Called for every instruction while recording (from the record dispatch).
    void RecordInstruction(const vm::BcIns_t* pPc, vm::TValue_t* pBase);
    [[nodiscard]] bool IsRecording() const noexcept { return m_eState == ETraceState::Recording; }

    [[nodiscard]] std::uint64_t TraceCount() const noexcept { return m_vTraces.size(); }

private:
    // --- recording ----------------------------------------------------------
    void StartRecording(const vm::BcIns_t* pPc, vm::TValue_t* pBase);
    void AbortRecording(EAbort eReason, const char* sDetail);
    void CloseLoop();

    [[nodiscard]] IrRef SlotRef(std::int32_t nSlot);          // load-on-demand + guard
    void SetSlot(std::int32_t nSlot, IrRef rValue);
    [[nodiscard]] IrRef Emit(EIrOp eOp, EIrType eType, IrRef rOp1, IrRef rOp2);
    [[nodiscard]] IrRef Constant(const vm::TValue_t& tvValue);
    [[nodiscard]] IrRef ConstantNum(double flValue);
    [[nodiscard]] std::uint32_t TakeSnapshot(std::uint32_t uResumePc);
    [[nodiscard]] EIrType ObservedType(const vm::TValue_t& tvValue) const noexcept;

    // Per-bytecode recorders (return false to abort).
    [[nodiscard]] bool RecordArith(const vm::BcIns_t& ins, vm::TValue_t* pBase);
    [[nodiscard]] bool RecordCompare(const vm::BcIns_t& ins, vm::TValue_t* pBase);
    [[nodiscard]] bool RecordTableGet(const vm::BcIns_t& ins, vm::TValue_t* pBase);
    [[nodiscard]] bool RecordTableSet(const vm::BcIns_t& ins, vm::TValue_t* pBase);
    [[nodiscard]] bool RecordCall(const vm::BcIns_t& ins, vm::TValue_t* pBase);
    [[nodiscard]] bool RecordReturn(const vm::BcIns_t& ins, vm::TValue_t* pBase);

    // --- backend ------------------------------------------------------------
    [[nodiscard]] const Trace_t* Assemble();
    [[nodiscard]] std::uint8_t* AllocCode(std::size_t uBytes);

    vm::C_Universe* m_pUniverse;
    ETraceState m_eState = ETraceState::Idle;

    // Recording state.
    const vm::BcIns_t* m_pStartPc = nullptr;
    vm::TValue_t* m_pEntryBase = nullptr;
    std::uint32_t m_uRecorded = 0;
    std::int32_t m_nBaseOffset = 0;                 // current frame, vs entry base
    std::vector<IrIns_t> m_vIns;                    // instructions (index + kIrBias = ref)
    std::vector<IrConst_t> m_vConst;                // constants (kIrBias - 1 - index = ref)
    std::vector<Snapshot_t> m_vSnapshots;
    std::vector<SnapSlot_t> m_vSnapSlots;
    IrRef m_vChain[static_cast<std::size_t>(EIrOp::Count_)]{};   // CSE chains
    std::vector<IrRef> m_vSlotValue;                // slot -> current IR value
    std::vector<EIrType> m_vSlotType;               // slot -> type already guarded

    // Inlined-frame bookkeeping so a snapshot can rebuild every frame.
    struct InlineFrame_t {
        std::int32_t nBaseOffset;
        const vm::BcIns_t* pReturnPc;
        IrRef rFunc;
    };
    std::vector<InlineFrame_t> m_vFrames;

    // Blacklist: sites that failed to record are not retried forever.
    std::unordered_map<const vm::BcIns_t*, std::uint32_t> m_mapHot;
    std::unordered_map<const vm::BcIns_t*, Trace_t*> m_mapTraces;
    std::vector<Trace_t*> m_vTraces;

    std::uint8_t* m_pCodeArena = nullptr;
    std::size_t m_uCodeUsed = 0;
};

}  // namespace ljx::jit
