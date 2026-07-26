// LJX — VM observation events.
// Layer 3.
//
// The substrate for jit.dump/jit.v/jit.p equivalents, trace-abort
// diagnostics, and external tooling (perf map / jitdump / GDB JIT
// registration hang off a sink). A null sink costs one predictable branch on
// COLD paths only — no event fires from an interpreter fast path.
#pragma once

#include <cstdint>

#include "ljx/vm/Bytecode.hpp"

namespace ljx::vm {

class C_GcProto;
class C_LuaThread;

enum class EVmEvent : std::uint8_t {
    BytecodeNew,     // a proto finished loading/parsing
    TraceStart,
    TraceStop,       // published (payload: trace number, link type)
    TraceAbort,      // payload: trace error + blame PC
    TraceExit,       // side exit taken (payload: trace, exit number)
    Record,          // one bytecode recorded (verbose tooling only)
    GcStateChange,
    FinalizerError,  // error thrown from a __gc handler (reported, not unwound)
};

struct VmEventData_t {
    EVmEvent eEvent{};
    std::uint16_t uTraceNumber = 0;
    std::uint32_t uAuxCode = 0;        // ETraceError / EGcState / exit number
    const C_GcProto* pProto = nullptr;
    const BcIns_t* pPc = nullptr;
};

// Abstract interface (cold-path virtual dispatch is acceptable here — and
// only here; no hot-path type in LJX has a vtable).
class C_IVmEventSink {
public:
    virtual ~C_IVmEventSink() = default;
    virtual void OnVmEvent(C_LuaThread* pThread, const VmEventData_t& eventData) = 0;
};

}  // namespace ljx::vm
