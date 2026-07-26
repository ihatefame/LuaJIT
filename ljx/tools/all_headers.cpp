// LJX — aggregation TU: every public header must compile standalone and
// together under -std=c++20. This file is the CI syntax gate for the
// interface-only phase (no implementations exist yet).

#include "ljx/core/Config.hpp"
#include "ljx/core/Memory.hpp"
#include "ljx/core/Types.hpp"

#include "ljx/vm/Bytecode.hpp"
#include "ljx/vm/FastFunc.hpp"
#include "ljx/vm/Frame.hpp"
#include "ljx/vm/Interpreter.hpp"
#include "ljx/vm/Object.hpp"
#include "ljx/vm/Value.hpp"
#include "ljx/vm/VmEvent.hpp"

#include "ljx/gc/GarbageCollector.hpp"

#include "ljx/rt/Meta.hpp"
#include "ljx/rt/StringBuffer.hpp"
#include "ljx/rt/StringInterner.hpp"

#include "ljx/fe/Lexer.hpp"
#include "ljx/fe/Parser.hpp"

#include "ljx/jit/Backend.hpp"
#include "ljx/jit/Fold.hpp"
#include "ljx/jit/Ir.hpp"
#include "ljx/jit/Recorder.hpp"
#include "ljx/jit/Snapshot.hpp"
#include "ljx/jit/Trace.hpp"

#include "ljx/ffi/Ffi.hpp"

#include "ljx/api/Engine.hpp"

int main() {
    // Compile-time sanity only; behavior arrives with the implementation.
    static_assert(sizeof(ljx::vm::TValue_t) == 8);
    static_assert(sizeof(ljx::jit::IrIns_t) == 8);
    static_assert(sizeof(ljx::jit::Snapshot_t) == 12);
    return 0;
}
