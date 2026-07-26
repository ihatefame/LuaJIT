// LJX — the trace compiler.
//
// Unlike the loop and function tiers, this one does not recognize shapes. It
// records whatever bytecodes actually execute at a hot back edge, specializes
// on the types actually observed, and leaves through a side exit when an
// assumption breaks. That makes it indifferent to whether the code is
// arithmetic, table traffic or method dispatch. See docs/TRACE_DESIGN.md.
#pragma once

#include <cstdint>
#include <unordered_map>
#include <vector>

#include "ljx/jit/TraceIr.hpp"
#include "ljx/vm/Bytecode.hpp"
#include "ljx/vm/Value.hpp"

namespace ljx::vm {
class C_Universe;
class C_GcProto;
class C_GcTable;
class C_GcFunction;
}

namespace ljx::jit {

enum class ETraceState : std::uint8_t { Idle, Recording };

enum class EAbort : std::uint8_t {
    None, Unsupported, TooLong, LeftFrame, BadType, NoMemory, RegPressure, Blacklist,
};

// Where a side exit lands: the bytecode to resume at and the frame it belongs
// to. The exit stub has already written every live value back to the Lua
// stack, so the interpreter needs nothing else.
struct TraceExit_t {
    const vm::BcIns_t* pResumePc = nullptr;
    std::int32_t nBaseOffset = 0;
};

// Entry ABI of compiled trace code:
//   uint32 fn(TValue_t* pBase, std::uintptr_t uArenaBase)
// returns the index of the exit that was taken.
using TraceEntry_f = std::uint32_t (*)(vm::TValue_t* pBase, std::uintptr_t uArenaBase);

struct Trace_t {
    void* pCode = nullptr;
    const vm::BcIns_t* pStartPc = nullptr;
    std::vector<TraceExit_t> vExits;
    std::int32_t nTopSlot = 0;      // highest stack slot the trace touches
    std::uint32_t uNumber = 0;
    std::uint64_t uEntries = 0;                 // diagnostics (LJX_TRACEDEBUG)
    std::vector<std::uint64_t> vExitCounts;
};

class C_TraceJit {
public:
    static constexpr std::uint32_t kHotLoopThreshold = 53;
    static constexpr std::uint32_t kBlacklistCount = 0x4000'0000u;
    static constexpr std::uint32_t kMaxRecordedIns = 600;
    static constexpr std::uint32_t kMaxIrIns = 1500;
    static constexpr std::int32_t kMaxSlot = 180;
    static constexpr std::int32_t kSlotBias = 4;   // frames reach to base-2
    static constexpr std::uint32_t kMaxInlineDepth = 5;
    static constexpr std::size_t kCodeArenaSize = 4u << 20;

    explicit C_TraceJit(vm::C_Universe& uni) noexcept : m_pUniverse(&uni) {}
    ~C_TraceJit();

    C_TraceJit(const C_TraceJit&) = delete;
    C_TraceJit& operator=(const C_TraceJit&) = delete;

    // --- interpreter hooks --------------------------------------------------
    // A loop back edge was taken and the interpreter is about to dispatch
    // `pHeadPc`. Returns a trace to enter, or null (and may start recording).
    [[nodiscard]] const Trace_t* OnLoopEdge(const vm::BcIns_t* pHeadPc, vm::TValue_t* pBase,
                                            const vm::TValue_t* pKBase);
    // Called for every instruction while recording, before it executes.
    void RecordInstruction(const vm::BcIns_t* pPc, vm::TValue_t* pBase,
                           const vm::TValue_t* pKBase);
    [[nodiscard]] bool IsRecording() const noexcept { return m_eState == ETraceState::Recording; }
    // Trace installed at a loop head, or null. Used by the J-variant loop ops.
    [[nodiscard]] const Trace_t* TraceAt(const vm::BcIns_t* pHeadPc) const noexcept {
        const auto it = m_mapTraces.find(pHeadPc);
        return it == m_mapTraces.end() ? nullptr : it->second;
    }
    // The interpreter re-entered itself (a metamethod, a C function): whatever
    // runs inside that nested call is not part of this trace.
    void AbortForReentry() { AbortRecording(EAbort::Unsupported, "interpreter re-entry"); }

    [[nodiscard]] std::size_t TraceCount() const noexcept { return m_vTraces.size(); }

private:
    // --- recording ----------------------------------------------------------
    void StartRecording(const vm::BcIns_t* pPc, vm::TValue_t* pBase,
                        const vm::TValue_t* pKBase);
    void AbortRecording(EAbort eReason, const char* sDetail);
    void CloseLoop();
    [[nodiscard]] bool RecordOne(const vm::BcIns_t& ins, const vm::BcIns_t* pPc);

    [[nodiscard]] IrRef SlotRef(std::int32_t nSlot);          // load-on-demand + guard
    void SetSlot(std::int32_t nSlot, IrRef rValue);
    [[nodiscard]] IrRef Emit(EIrOp eOp, EIrType eType, IrRef rOp1, IrRef rOp2);
    [[nodiscard]] IrRef EmitGuard(EIrOp eOp, IrRef rOp1, IrRef rOp2, const vm::BcIns_t* pResumePc);
    [[nodiscard]] IrRef Constant(const vm::TValue_t& tvValue);
    [[nodiscard]] IrRef ConstantNum(double flValue);
    [[nodiscard]] IrRef ConstantInt(std::int64_t nValue);
    [[nodiscard]] IrRef ConstantPtr(const void* pPtr);
    [[nodiscard]] std::uint32_t TakeSnapshot(const vm::BcIns_t* pResumePc);
    [[nodiscard]] EIrType ObservedType(const vm::TValue_t& tvValue) const noexcept;
    [[nodiscard]] EIrType TypeOf(IrRef rRef) const noexcept;
    [[nodiscard]] vm::TValue_t ConstValue(IrRef rRef) const noexcept;

    // Bytecode groups.
    [[nodiscard]] bool RecordArith(const vm::BcIns_t& ins, EIrOp eOp, int nKind);
    [[nodiscard]] bool RecordCompare(const vm::BcIns_t& ins, const vm::BcIns_t* pPc);
    [[nodiscard]] bool RecordTest(const vm::BcIns_t& ins, const vm::BcIns_t* pPc);
    [[nodiscard]] bool RecordTableGet(const vm::BcIns_t& ins, const vm::BcIns_t* pPc,
                                      vm::TValue_t tvKey, std::int32_t nTabSlot,
                                      IrRef rKeyDynamic);
    [[nodiscard]] bool RecordTableSet(const vm::BcIns_t& ins, const vm::BcIns_t* pPc,
                                      vm::TValue_t tvKey, std::int32_t nTabSlot);
    [[nodiscard]] bool RecordForL(const vm::BcIns_t& ins, const vm::BcIns_t* pPc);
    [[nodiscard]] bool RecordCall(const vm::BcIns_t& ins, const vm::BcIns_t* pPc);
    [[nodiscard]] bool RecordReturn(const vm::BcIns_t& ins, std::uint32_t uFirst,
                                    std::uint32_t uCount);

    // Hash-part access shared by get and set. Returns the node pointer ref for
    // a key that is PRESENT, or kIrNone (with bAbsent set) when the recorded
    // lookup proved the key absent.
    [[nodiscard]] IrRef HashNodeRef(IrRef rTabPtr, const vm::C_GcTable* pTab,
                                    vm::TValue_t tvKey, IrRef rKeyRef,
                                    const vm::BcIns_t* pResumePc, bool& bAbsent);

    // --- backend ------------------------------------------------------------
    [[nodiscard]] Trace_t* Assemble();
    [[nodiscard]] std::uint8_t* AllocCode(std::size_t uBytes);

    vm::C_Universe* m_pUniverse;
    ETraceState m_eState = ETraceState::Idle;

    // Recording state.
    const vm::BcIns_t* m_pStartPc = nullptr;
    vm::TValue_t* m_pEntryBase = nullptr;
    vm::TValue_t* m_pBase = nullptr;                // current frame while recording
    const vm::TValue_t* m_pKBase = nullptr;         // current frame's constants
    std::uint32_t m_uRecorded = 0;
    std::int32_t m_nBaseOffset = 0;                 // current frame, vs entry base
    std::int32_t m_nTopSlot = 0;
    std::vector<IrIns_t> m_vIns;                    // instructions (index + kIrBias = ref)
    std::vector<std::uint16_t> m_vInsSnap;          // guard -> snapshot index
    std::vector<IrConst_t> m_vConst;                // constants (kIrBias - 1 - index = ref)
    std::vector<Snapshot_t> m_vSnapshots;
    std::vector<SnapSlot_t> m_vSnapSlots;
    IrRef m_vChain[static_cast<std::size_t>(EIrOp::Count_)]{};   // CSE chains
    std::vector<IrRef> m_vSlotValue;                // slot -> current IR value
    std::vector<IrRef> m_vSlotEntry;                // slot -> entry SLoad (loop carried)

    // Inlined frames, innermost last.
    struct InlineFrame_t {
        std::int32_t nBaseOffset;
        const vm::BcIns_t* pReturnPc;
        const vm::TValue_t* pKBase;
    };
    std::vector<InlineFrame_t> m_vFrames;

    void PinValue(const vm::TValue_t& tvValue);

    std::unordered_map<const vm::BcIns_t*, std::uint32_t> m_mapHot;
    std::unordered_map<const vm::BcIns_t*, Trace_t*> m_mapTraces;
    std::vector<Trace_t*> m_vTraces;

    std::uint8_t* m_pCodeArena = nullptr;
    std::size_t m_uCodeUsed = 0;
};

[[nodiscard]] bool TraceDebug() noexcept;

}  // namespace ljx::jit
