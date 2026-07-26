# LJX — A Ground-Up C++20 Redesign of LuaJIT

**Status:** Architecture proposal + core interfaces (no implementation yet)
**Scope:** 64-bit little-endian targets only (x86-64, AArch64). Lua 5.1 semantics + LuaJIT
extensions, `ffi`, `string.buffer`, and this fork's syntax extensions.

---

## 1. Goals and Design Philosophy

LJX is a clean-room re-architecture of LuaJIT as a modular C++20 system. The goal is not to
transliterate `lj_*.c` into classes — it is to **keep every measured performance mechanism of
LuaJIT** (most of which live in data layout, not code) while replacing the three things C forced
on the original:

1. **The hand-written assembly interpreter** → a guaranteed-tail-call (`[[clang::musttail]]` +
   `preserve_none`) continuation-passing interpreter, one C++ function per opcode, generated from
   a single opcode registry via templates.
2. **The `buildvm`/DynASM offline metaprogram for tables** → `constexpr`/`consteval` generation
   of dispatch metadata, fold-rule perfect hashes, library registration streams, and opcode
   tables, checked by `static_assert` instead of by four generated headers agreeing by luck.
3. **Textual-macro polymorphism** (`lj_asm_x86.h` inclusion, `CCALL_HANDLE_*` `#elif` chains) →
   C++20 **concepts + CRTP** backends and per-ABI policy classes, with zero virtual dispatch on
   any hot path.

The guiding rule, in tension order:

> **Data layout first. Mechanical sympathy second. Object-orientation at module boundaries
> only.** A class may organize a subsystem; it may never insert an indirection, a virtual call,
> or a fatter struct into a hot path. Every hot structure has a frozen, `static_assert`-pinned
> layout.

"Beat LuaJIT" concretely means: match the interpreter within noise (musttail + pinned context),
match trace codegen (same IR, same reverse-LSRA, same fold rules), then win on the new headroom:
generational arena GC with bitmap sweeps, SIMD string operations, huge-page VM state, wider ISA
use (BMI2, AArch64 `ldp`/`stp`, optionally APX), and PGO/BOLT-friendly per-opcode functions.

### Non-goals (v1)

- 32-bit targets, big-endian, MIPS/PPC, softfp, x87, non-FR2 frames, DUALNUM-off builds
  (LJX is **GC64+FR2, dual-number always** — this deletes roughly half of the conditional
  complexity in `lj_obj.h`/`lj_frame.h`).
- Multi-threaded mutation of one VM. (A clean seam for background trace *assembly* is designed
  in, see §7.8, but the mutator is single-threaded, as in LuaJIT.)
- Bytecode-dump compatibility with stock LuaJIT (format keeps the concepts, bumps the version).

---

## 2. Naming Conventions (normative for the whole codebase)

| Entity                       | Convention              | Example                              |
|------------------------------|-------------------------|--------------------------------------|
| Class (has behavior)         | `C_` + CamelCase        | `C_TraceRecorder`, `C_IrBuffer`      |
| Abstract interface class     | `C_I` + CamelCase       | `C_IVmEventSink`                     |
| Struct (POD / layout type)   | CamelCase + `_t`        | `TValue_t`, `IrIns_t`, `Snapshot_t`  |
| Enum (always `enum class`)   | `E` + CamelCase         | `EBcOp`, `EIrOp`, `EGcState`         |
| Enumerator                   | CamelCase               | `EGcState::Propagate`                |
| Concept                      | `Is` + CamelCase        | `IsMachineBackend`                   |
| Function type alias          | CamelCase + `_f`        | `BcHandler_f`                        |
| Template parameter           | `T` + CamelCase         | `TBackend`, `TPolicy`                |
| Compile-time constant        | `k` + CamelCase         | `kRefBias`, `kCacheLineSize`         |
| Class member                 | `m_` + Hungarian        | `m_pIrBuffer`, `m_nTop`, `m_flStep`  |
| Static class member          | `s_` + Hungarian        | `s_pDefaultAllocator`                |
| Global                       | `g_` + Hungarian        | `g_pActiveUniverse`                  |
| Local / parameter            | Hungarian               | `nSlot`, `flValue`, `sChunkName`     |

Hungarian type tags: `n` int, `u` unsigned, `fl` float/double, `b` bool, `s` string, `p` pointer,
`e` enum value, `fn` function/callable, `it` iterator, `v` container, `by` byte, `h` handle,
`tv` TValue, `r` reference/ref-id (e.g. `rOp1` for an `IrRef_t`; **reserved** for ref-ids —
never used to mean "C++ reference parameter"), `sv` string_view, `tr` TRef, `ins` bytecode/IR
instruction, `pos` bytecode position, `tok` token.

Supplementary rules (normative):

- **Aggregate-typed variables carry no scalar tag.** A variable whose type is a class/struct
  (by value or reference) uses a descriptive CamelCase name: parameters `asmCtx`, `uni`,
  `exitState`; members `m_Dispatch`, `m_Recorder`, `m_Stats`. Scalar tags are for scalars,
  pointers (`p`), and the typed tags listed above.
- **Enums use a trailing-underscore `Count_` sentinel** where a dense count is needed; the
  sentinel is not a real enumerator and is exempt from the CamelCase rule.
- **GC-heap object types are always `C_Gc*` / `C_LuaThread`**, even while an individual type
  has few or no methods yet — they are identity-bearing objects whose behavior accretes with
  the implementation. Interior/value PODs (`GcHeader_t`, `TableNode_t`, `TValue_t`,
  `Snapshot_t`, …) use `_t`.

Namespaces are lowercase and mirror the directory layout: `ljx::core`, `ljx::vm`, `ljx::gc`,
`ljx::rt`, `ljx::fe`, `ljx::jit`, `ljx::ffi`, `ljx::api`.

---

## 3. Module Map

```
                    ┌───────────────────────────────────────────────┐
                    │  api/    C_LuaEngine, Lua 5.1 C-API shim      │
                    └───────────────┬───────────────────────────────┘
     ┌──────────────┐   ┌───────────┴───────────┐   ┌──────────────┐
     │ fe/          │   │ vm/                   │   │ ffi/         │
     │ C_Lexer      │──▶│ TValue_t  Object model│◀──│ C_CTypeReg   │
     │ C_Parser     │   │ EBcOp     BcIns_t     │   │ C_AbiPlan    │
     │ (→ bytecode) │   │ C_Interpreter (CPS)   │   │ C_Callback   │
     └──────────────┘   │ C_DispatchTable       │   └──────┬───────┘
                        └───────────┬───────────┘          │
     ┌──────────────┐   ┌───────────┴───────────┐   ┌──────┴───────┐
     │ rt/          │   │ jit/                  │   │ gc/          │
     │ C_StringIntern│◀─│ C_TraceRecorder       │──▶│ C_GarbageCol │
     │ table ops    │   │ C_IrBuffer + C_Fold   │   │ arenas, maps │
     │ C_MetaResolver│  │ C_LoopOptimizer       │   │ barriers     │
     └──────────────┘   │ C_SnapshotWriter      │   └──────────────┘
                        │ C_AsmContext (LSRA)   │
     ┌──────────────┐   │ backends (concept)    │
     │ core/        │   │ C_MachineCodeArena    │
     │ Config Types │   │ C_TraceCache C_Jit    │
     │ C_VirtualArena│  └───────────────────────┘
     └──────────────┘
```

Dependency rule: arrows only point downward/leftward in the table below. `core` depends on
nothing; `api` depends on everything. **No module reaches into another's data — cross-module
contracts are the frozen layouts in `vm/` plus the interfaces in each module's header.**

| Layer | Directory | Contents |
|-------|-----------|----------|
| 0 | `core/` | platform config, attributes, `C_VirtualArena`, compressed refs, bit utils |
| 1 | `vm/` (data) | `TValue_t`, GC object model, `BcIns_t`, frames, `C_Universe` |
| 2 | `gc/`, `rt/` | collector, string interner, table/meta ops, buffers |
| 3 | `vm/` (exec) | dispatch tables, hot counters, the CPS interpreter |
| 4 | `fe/` | lexer, single-pass parser → bytecode |
| 5 | `jit/` | recorder → IR → fold/opt → snapshots → regalloc → backend → trace cache |
| 6 | `ffi/` | ctype registry, ABI classifier, callbacks |
| 7 | `api/` | embedding facade + Lua 5.1 C API compatibility layer |

---

## 4. Core Value & Object Model (`vm/Value.hpp`, `vm/Object.hpp`)

### 4.1 `TValue_t`: 8-byte NaN box, GC64 layout, frozen

The single most load-bearing decision in LuaJIT is the 8-byte NaN-boxed value, and LJX keeps it
bit-for-bit:

```
 |----------- 64-bit word ------------|
 | 1..1 (13) | tag (4) | payload (47) |    non-double values
 |            IEEE-754 double         |    numbers (verbatim)
```

- Tags are **totally ordered complements** (`EValueTag::Nil` = all-ones … numbers lowest) so one
  unsigned compare classifies {double | int | primitive | GC object}, `nil` tests compile to
  `cmp reg, -1`, and primitives are synthesizable in two ALU ops. Dozens of range predicates
  (`isTruthy`, `isTableOrUdata`, `isGcObject`) are single compares; the tag order is pinned by
  `static_assert` next to the predicates that exploit it.
- All reinterpretation goes through `std::bit_cast` — no unions, no strict-aliasing UB, and every
  encode/decode helper is `constexpr` (constants can be built at compile time).
- A 16-byte "clean" variant was considered and rejected: it halves stack/table cache density and
  loses the one-instruction copy (`mov`) and the fused type+identity compare on table keys.
- Double→int conversions with out-of-range/NaN inputs are UB in C++; LJX routes **all** of them
  through named wrappers (`ljx::core::NumToInt32Trunc` etc.) implemented with intrinsics
  (`_mm_cvttsd_si32`, AArch64 `fcvtzs`) so interpreter, runtime, and JIT agree bit-for-bit —
  including the `0x80000000` INDEFINITE result the array-bounds trick relies on.

### 4.2 Pointer strategy: one reserved arena, 47-bit boxing, optional 32-bit refs

LuaJIT's non-GC64 mode proves 32-bit refs buy real cache density (smaller headers everywhere),
but its "keep the whole heap under 2/4 GB" allocator (`MAP_32BIT`, probing, NT hacks) is its most
fragile OS dependence, and the 47-bit GC64 assumption breaks under 5-level paging/LAM.

LJX: at startup, `core::C_VirtualArena` **reserves one contiguous VA range below 2^47**
(commit-on-demand; capped at `kMaxArenaReserve` = 32 GB — the ref-width contract below).
All GC memory comes from it. Consequences:

- 47-bit NaN boxing is a *guaranteed contract*, not a hope about `mmap` behavior.
- `GcRef_t` is a 32-bit **arena-relative granule index**: referents are 8-byte aligned (MRefs
  must reach TValue-aligned interiors — stack slots, constant areas), so `base + (uint64)idx*8`
  (one addressing-mode operand on both ISAs) covers the full 32 GB reservation — which is why
  the reservation is capped there, with the cap and the ref width tied together by one
  constant. `MRef_t` is the same for non-GC memory that lives in-arena.
  This generalizes LuaJIT's non-GC64 density win to arbitrarily placed heaps,
  V8-pointer-compression-style, and makes multiple isolated VMs trivial (per-VM base register
  = the existing context register).
- Full 64-bit `TValue_t` payloads still carry real pointers (tag strip = 2 ALU ops), so the
  interpreter fast paths are unchanged; compression applies to **object-internal** references.
- Arena granules are 2 MB-aligned → transparent huge pages for the heap and the VM universe.

### 4.3 GC object model

Kept from LuaJIT (all pinned by `static_assert`):

- 8-byte object headers: `GcHeader_t { GcRef_t rNextGc; uint8 uMarked; EGcObjectType eType;
  uint8 uExtra1, uExtra2 }` — the two "hole" bytes are per-type fields exactly as in LuaJIT
  (string keyword id / hash algo, table `nomm` / colocation, proto params/framesize…).
- **Offset-aliasing contracts**: `m_rMetatable` at the same offset in `C_GcTable`/
  `C_GcUserData`; `m_rEnv` aliased in function/userdata; `m_rGcList` aliased across
  thread/proto/func/table. These let the `__eq` fast path and the marker use one load for
  multiple types.
- `C_GcProto` is **one colocated allocation**: header, bytecode array immediately after (so
  proto metadata is at negative offsets from any PC — one register is both PC and proto handle),
  split constant array (`k` points to the middle; GC constants at negative indices, numbers at
  positive; bytecode encodes GC-constant operands complemented so `not` produces the index),
  upvalue descriptors, compressed debug info.
- Strings: interned, immutable, NUL-terminated, zero-padded to 16 bytes (up from LuaJIT's 4) —
  the padding is what makes word/SIMD-at-a-time compare and hash legal. Table hashing uses the
  dense, reseedable `uSid` (interning ID), never the content hash — HREFK soundness depends on it.
- Tables: hybrid array+hash; `TableNode_t { tvValue, tvKey, rNext }` (24 B) with
  **main-position chaining + Brent's eviction**; **dead keys keep their slot until an explicit resize** (the JIT
  contract behind HREFK stability and hoisting HREF across GC steps); shared `nilnode` sentinel
  removes the `hmask==0` branch; ≤16-slot arrays colocated in the table allocation; 1-byte
  negative-metamethod cache `uNoMm` invalidated wholesale on any store. **Explicitly rejected:**
  open-addressing/Swiss tables — probe-sequence relocation breaks HREFK, LuaJIT's strongest
  table optimization.

### 4.4 `C_Universe`: the one-allocation VM

LuaJIT's GG_State trick is kept and formalized: `C_Universe` colocates the main thread state,
global state, JIT state, hot-count table, and both dispatch tables in one 2 MB huge page, so a
**single pinned context register** addresses all of it with `[reg + disp32]`. Every offset the
interpreter or JIT bakes in is exported as a `constexpr` and `static_assert`-pinned. No GC-visible
object has a vtable or RTTI; type dispatch in the GC is by `EGcObjectType` switch or CRTP,
resolved at compile time.

---

## 5. Bytecode & Frames (`vm/Bytecode.hpp`, `vm/Frame.hpp`)

- 32-bit fixed-width instructions, opcode in the low byte, `A`/`B`/`C` 8-bit and `D` 16-bit
  fields exactly as LuaJIT (the 250-slot frame limit is a feature: it disciplines register
  pressure and keeps snapshots byte-cheap).
- The opcode set is defined **once** in an X-macro/`constexpr` registry that generates: the enum,
  operand-mode table, metamethod association, dispatch table, disassembler names, and the
  **opcode algebra** `static_assert`s (`ISLT^1==ISGE`, `FORL+1==IFORL`, comparison ops followed
  by `JMP`, `RETM+1==RET` adjacency…). The recorder and dispatch-mode machinery compute variants
  by arithmetic, so the algebra is enforced at compile time, not by comments.
- Comparison+JMP fusion, `KSHORT` sign-extension, complemented GC-constant operands, `FORL`
  slot quadruple, `ITERN`/`ISNEXT` speculation — all kept.
- One deliberate change: `ISNEXT`/`ITERN` **despecialization patches a per-site inline-cache
  word, not the bytecode itself**, keeping bytecode pages read-only/shareable (mmap'd caches);
  trace-entry patching (`FORL→JFORL` etc.) stays as self-modifying bytecode because it is
  fundamental to zero-overhead trace entry.
- **Frames stay on the value stack** (no CallInfo array): two-slot FR2 frames
  `[func][link]`, link low bits encode `EFrameType` (Lua/C/Cont/Varg/CPCall/PCall/PCallHook),
  previous Lua frame recovered by decoding the `A` field of the call instruction before the
  saved PC. `LUAI_MAXSTACK < 64K` so slot deltas stay 16-bit. Coroutine stacks reserve max VA
  up front and grow by commit — the delta-fixup of base/top/upvalues on stack realloc disappears.

---

## 6. The Interpreter (`vm/Interpreter.hpp`)

The hand-written assembly VM is replaced by a **continuation-passing tail-call interpreter**:

```cpp
// One function per opcode; ~all hot state pinned in argument registers.
using BcHandler_f = void (*)(vm::TValue_t*        pBase,     // frame base
                             const vm::BcIns_t*   pPc,       // next instruction
                             uint64_t             uRa,       // decoded A
                             uint64_t             uRd,       // decoded D (B/C cracked lazily)
                             vm::C_Universe*      pUni,      // context (dispatch/g/J/hot)
                             const vm::TValue_t*  pKBase);   // current constants

template <vm::EBcOp TOp>
LJX_PRESERVE_NONE void OpHandler(...) {
    ...fast path...
    LJX_MUSTTAIL return pUni->DispatchNext(pBase, pPc, pUni, pKBase); // macro-expanded decode
}
```

Design points, each mirroring a measured property of the assembly VM (see survey):

1. **Guaranteed tail calls** (`[[clang::musttail]]`; GCC 15 equivalent) + `preserve_none`
   calling convention: handlers get ~12 effectively-pinned registers and zero prologue —
   the modern equivalent of DynASM's fixed register file (validated by CPython 3.14, upb/wasm3,
   LuaJIT-Remake). Computed-goto is the documented fallback; a plain switch is not acceptable.
2. **Replicated dispatch**: each handler ends in its *own* indirect jump — per-opcode BTB
   entries, exactly like `ins_NEXT` macro expansion. Never centralize dispatch.
3. **Lazy operand cracking**: decode OP, A, and D up front (mirroring the assembly VM's
   dispatch sequence — see the `BcHandler_f` signature above); B/C are cracked from D inside
   the handlers that need them, with BMI2/`ubfx` (no partial-register games needed on modern
   cores).
4. **Slow-path discipline**: every fallback is a separate `[[gnu::noinline]]` function that
   *re-enters via tail call*, never a call that returns into the handler — otherwise handler
   prologues grow spills and the fast path pays for the slow path. This is *the* failure mode of
   naive C interpreters and is a hard review rule (`docs: SLOWPATH-RULE`).
5. **Dual dispatch tables** (dynamic + static) remain the *only* instrumentation mechanism:
   hooks, profiling, and trace recording swap `std::atomic<BcHandler_f>` entries (relaxed loads
   in the hot path — free on x86/ARM64; release stores in `SetDispatchMode`, async-signal-safe
   for `lua_sethook`).
6. **Hashed hot counters**: the 64×16-bit shared table, decrement 2 per loop edge / 1 per call,
   trigger on borrow — 3 instructions, no per-proto storage; collision decay is a feature.
   Constants (`kHotLoopThreshold=56`, penalties, blacklisting) are preserved numerically (§7.6).
7. **Uniform call dispatch through `pPc`**: every callable's first "instruction" selects its
   header handler (Lua header ops, C-closure op, or one of the ~57 synthetic builtin fast-path
   ops) — one call path for everything, builtins dispatched at bytecode cost, with the
   `FfhRetry/FfhRes/FfhTailcall` C-fallback protocol kept.
8. **Interpreter as deopt target**: the handler signature *is* the deopt ABI. `ExitState_t`
   (all GPRs/vector regs at fixed offsets) + the restore path re-enter the interpreter at an
   arbitrary bytecode with reconstructed `pBase/pPc/pKBase` — same contract as `vm_exit_interp`,
   including the `MULTRES` slot and KBASE rematerialization rules.
9. **pcall/coroutines**: the frame-type-bits-in-link scheme is kept (cheapest known pcall — no
   setjmp on the non-throwing path); unwinding uses the C++ EH personality only on cold paths,
   and resume/yield keep the C-frame-chain flag-bit protocol.

MSVC has no musttail: Windows v1 ships clang-cl. (A DynASM-style asm kernel remains a
contingency, sequestered behind the same `BcHandler_f` ABI.)

---

## 7. The JIT (`jit/*.hpp`)

Trace compilation is LuaJIT's crown jewel; LJX keeps the pipeline shape **exactly** —
record → on-the-fly fold/CSE/forward → loop copy-substitution → sink/DCE → backward
regalloc+codegen — and modernizes the expression of each stage.

### 7.1 IR (`jit/Ir.hpp`)

- `IrIns_t` is **exactly 64 bits**: `(rOp1, rOp2, uOpAndType, rPrev)` with accessor views
  (`Op()`, `Type()`, `Op12()`). The `rPrev` field is the per-opcode CSE chain link before
  regalloc and the `(reg,spill)` pair after — phase reuse expressed as named accessors
  (`ChainPrev()` / `AllocatedReg()`+`SpillSlot()`), not separate storage. SoA was considered
  and rejected: fold/CSE touch `(o, op12, t, prev)` of one instruction per emit; the 8-byte
  cell is one line.
- **16-bit biased refs**: constants grow down from `kRefBias`, instructions up. One magnitude
  compare implements: is-constant, DCE marking, loop-invariant classification, and regalloc
  eviction priority. 64-bit constants take two slots. All literal operands `< kRefBias`
  (compile-time-checked for `EIrConvMode`/field ids/call ids).
- **Per-opcode skip lists** (`m_vChain[EIrOp::Count_]` + `rPrev`) for CSE/forwarding/alias
  analysis, search bounded by `max(rOp1, rOp2)`. Constants interned per opcode → reference
  equality is value equality everywhere.
- Opcode algebra (`Lt^1==Ge`, load→store delta, `CallN..CArg` contiguity) `static_assert`-pinned
  against the X-macro registry.

### 7.2 Fold engine (`jit/Fold.hpp`)

The ~500 `LJFOLD` rules become `constexpr FoldRule_t` descriptors; a **`consteval` builder**
runs the same smallest-semi-perfect-hash search buildvm does (multiply-shift & double-rotate
families, primary+secondary slot) and emits the table as `constinit` data. The runtime lookup —
24-bit key, branchless wildcard cascade, two probes, `NextFold/RetryFold/KIntFold/FailFold/
DropFold` protocol — is kept **verbatim**; it is measured, minimal, and branch-predictable.
Rule invariants (type preservation, PHI barrier, monotonic strength reduction, "every
load/store/alloc has an any/any rule or it falls through to CSE illegally") are documented on
the descriptor type and spot-checked by `consteval` validators where expressible.

### 7.3 Recorder (`jit/Recorder.hpp`)

`C_TraceRecorder` keeps: the slot-map mirror of the interpreter stack (`TRef_t` per slot, type
in the top byte), record-one-instruction-behind with `EPostProc` fixups, runtime-value peeking,
scalar-evolution cache for `FORL`, narrowing via the backpropagation stack machine with the
16-entry round-robin cache, per-opcode recording handlers (split out of the 2000-line switch
into a table, same contract), and abort-by-unwind through a protected call so a user error
thrown mid-recording aborts the trace for free. `rec_check_slots` becomes a debug-mode
executable invariant checker (`CheckSlotMap()`).

### 7.4 Loop optimization

Copy-substitution unrolling through the *full* fold pipeline (invariant guards hoisted by CSE,
store-to-load forwarding across the back edge), PHI placement below the body, type-instability
retry contract with the recorder (`LoopUndo` + unroll budget) — kept as-is; the essay at
`lj_opt_loop.c:22-90` explains why classic LICM loses, and nothing in C++20 changes that.
Allocation sinking keeps the `RegSink`-in-`uReg` piggyback and the snapshot re-materialization
triad (sink ↔ snapshot-alloc ↔ replay must be touched in lockstep — enforced by putting all
three in one module with a shared descriptor per sinkable op).

### 7.5 Snapshots & deopt (`jit/Snapshot.hpp`)

12-byte `Snapshot_t` headers + 32-bit packed `SnapEntry_t` (`slot<<24 | flags | ref`), sparse
slots, merge-when-no-guard, purge via the reaching-defs scan over following bytecode, frame links
appended, `mcofs` for binary-search unwind — all kept. `exitno == snapno` stays a structural
identity (all guards of a snapshot share one RegSp map; renames inside a guard group are killed
or force-spilled via the Bloom-filter pair). Exit stubs stay "code, not data" (`push imm8; jmp`
groups on x86-64; per-trace stubs on AArch64).

### 7.6 Trace management (`jit/Trace.hpp`)

`C_GcTrace` is a GC object holding compacted IR/snapshots/mcode metadata and the trace-graph
links (root/side/next chains, link types incl. stitching). Hot-count constants, penalty
doubling with 4 PRNG bits, `kPenaltyMax=60000` blacklisting via I-variant bytecode patching,
side-exit hotness (`kHotSideExitThreshold=10`), per-root side cap — **numerically preserved**;
these embody years of tuning and are exposed as `constexpr` config, not "cleaned up".

### 7.7 Register allocation & backends (`jit/Backend.hpp`)

- **Reverse linear-scan** with `RegCost_t` (blended cost | owning ref), lowest-ref eviction
  (constants evict first, rematerialized free), PHI weighting, weak sets, hints in the register
  byte, `ra_left` two-operand fusion — kept exactly. `RegSet_t` wraps `uint64_t` with
  `std::countr_zero`; the unrolled min-scan becomes an unrolled `constexpr` fold over a register
  pack (with codegen parity checked against a scalar loop in tests). The randomized-RA fuzz mode
  is kept as an invariant checker.
- **Backward single-pass codegen** (exact liveness, free DCE, guard emission after operands
  final, address-mode fusion by peeking at unemitted operands) — kept.
- Backends are selected by **concept, not virtuals**:

```cpp
template <typename TB>
concept IsMachineBackend = requires(C_AsmContext& asmCtx, const IrIns_t* pIns) {
    typename TB::MCodeUnit_t;                      // uint8_t (x86) / uint32_t (a64)
    { TB::kExitStubSpacing } -> std::convertible_to<uint32_t>;
    { TB::EmitIns(asmCtx, pIns) };
    { TB::EmitGuard(asmCtx, ECondCode{}) };
    { TB::CanRematerialize(IrRef_t{}) } -> std::convertible_to<bool>;
    ...
};
```

  The generic driver (`C_TraceAssembler<TBackend>`) owns regalloc/snapshots; backends own
  instruction selection with prepend-style emitters — the same division of labor as
  `lj_asm.c` + `lj_asm_x86.h`, expressed as a template instead of textual inclusion.
- `C_MachineCodeArena`: chained W^X areas, reserve/commit transactions, protection cache,
  jump-range-constrained placement with PRNG probing (AArch64 ±128 MB), `MAP_JIT` support, and an
  optional **dual-mapping mode** (RW alias + RX alias) for patch-heavy workloads — measured
  against the whole-area flip before becoming default.

### 7.8 Background assembly seam

Recording and optimization must stay synchronous (they read live stack state). But
`C_TraceAssembler` consumes only the finished IR/snapshot buffers plus the mcode arena — the
interfaces are written so trace *assembly* can later run on a helper thread with transactional
mcode publish and bytecode patch, without changing any hot-path contract.

---

## 8. GC (`gc/GarbageCollector.hpp`)

The one subsystem where LJX goes materially beyond LuaJIT (following the direction of the
unshipped LuaJIT 3.0 GC design):

1. **Arena heap + mark bitmaps** replace the intrusive all-objects *sweep* list: objects
   allocate from size-segregated bump regions inside `C_VirtualArena` blocks; mark state lives
   in per-arena bitmaps; **sweep becomes word-scans with `popcount`** and empty arenas are
   released without touching object memory. `rNextGc` remains in the header as the per-type
   chain field (string intern chains, the finalizer registry), and gray/gray-again membership
   stays intrusive via `m_rGcList` — an external worklist would make the table back-barrier
   allocating.
2. **Incremental tri-color with LuaJIT's exact barrier taxonomy**: back-barrier for tables
   (gray-again list ≡ remembered set), forward barrier elsewhere, make-white degradation outside
   propagate/atomic, threads never black, and the full elision rule list (stack slots, new
   objects, self-stores…). Any barrier redesign that adds a fence or second branch to a table
   store is rejected by design.
3. **Generational mode** built on the same barriers: sticky mark bits, minor collections that
   re-traverse only the gray-again/remembered set. The barrier cost model is unchanged — this is
   the low-risk path to beating LuaJIT on allocation-heavy benchmarks.
4. **Bounded atomic phase**: per-thread dirty flags (skip clean stack rescans), incremental
   weak-table clearing, dedicated finalizer registry instead of the mainthread-list ordering
   trick. Finalizers keep the mark-then-resurrect-once protocol and run scheduled, never from
   sweep; the trace interlock (`no atomic/finalize while on-trace; force trace exit instead`)
   is preserved verbatim.
5. **Exact accounting** (`uTotal` maintained by sized alloc/free — the `lua_Alloc` sized-free
   contract finally exploited), incremental pacing math (stepmul/pause percentages) kept with
   units rebased on real bytes marked/swept.

String interning keeps: chain-anchored interning with the low-bit secondary-hash tag, O(1)
sparse sampling hash for long strings with per-chain escalation to a keyed dense hash under
attack (chain > 32), StrID reseeding, secure-seed-or-refuse-to-start. Modernization: the dense
hash becomes hardware CRC32C/AES-based at memory bandwidth; comparisons go SIMD (legal because
of 16-byte padding).

---

## 9. Frontend (`fe/Lexer.hpp`, `fe/Parser.hpp`)

- **No AST, ever.** The single-pass parser emitting registerized bytecode through `ExpDesc_t`
  (a POD tagged struct — deliberately *not* `std::variant`: discharge is a hand-rolled in-place
  state machine) is the reason load time is negligible. Scope objects live on the native stack;
  the variable stack and bytecode stack are two shared growable arenas exposed as spans.
- Kept verbatim: keyword detection via the interned-string `uReserved` byte; constants interned
  through a real Lua table with the slot-index encoding; parse-time folding through the VM's own
  fold kernel (parse-time and run-time arithmetic can never disagree; never fold to NaN/-0);
  `KSHORT`/`KNIL`/`CAT` peepholes; jump-list threading through `JMP` D-fields; test-and-copy
  materialization avoidance; loop inversion; constructor template tables + `TDUP`; `ITERN`
  speculation with runtime despecialization; this fork's O(1) `uSid`-hashed variable lookup and
  syntax extensions (`?.`, `??`, bit-op bytecodes, compound assignment, lambdas, `const`/
  `continue`).
- Bytecode dump: same concepts (ULEB128, 33-bit tagged numbers, depth-first children, per-proto
  length framing, endian-swap on load), **new version byte**, deterministic output by default.

---

## 10. FFI (`ffi/Ffi.hpp`)

- Flat, append-only, interned `CType_t` cells (info embeds child id → structural equality is
  one integer compare; cdata type checks are one 16/32-bit id compare, which is what makes JIT
  guards cheap). Cells allocate in **stable chunks** (deque-of-blocks) killing the
  "`cts->tab` may reallocate" pointer-invalidation hazard class. Snapshot/rollback for aborted
  parses kept.
- **One ABI classifier** (`C_AbiClassifier<TAbi>` per-ABI policy classes) produces an
  `ArgPlan_t` (register/stack placement, extensions, splits) consumed by all three clients —
  FFI calls, callbacks, and the JIT's call recorder — instead of three macro sets that must
  agree; unit-tested against libffi as an oracle.
- Call execution keeps the `CCallState_t` memory-image + ~50-instruction asm stub split;
  callbacks keep the numbered-stub RX page (with BTI/PAC/IBT), the slot→(ctype, Lua function)
  mapping, and the on-trace blacklisting protocol. The universe pointer moves from
  baked-into-code to a data page adjacent to the stub page (shareable code pages, dual-map
  friendly).
- `crec`-style trace integration: FFI calls compile away into typed IR behind one ctype-id
  guard; declaration-string parsing at record time behind a string-identity guard.

---

## 11. Embedding API (`api/Engine.hpp`)

- `api::C_LuaEngine` is the C++ facade: owns a `C_Universe`, exposes typed push/get, protected
  calls, coroutines, and the JIT/GC control surface.
- A **compatibility layer** exports the Lua 5.1 C API + LuaJIT extensions (`lua_State*`
  aliases the main thread; `GLOBALSINDEX`/`ENVIRONINDEX` pseudo-indices with *two* scratch
  slots to defuse the aliasing trap; `lua_cpcall`; `luaJIT_setmode`; bcdump via `lua_dump`;
  the shared-niltv `LUA_TNONE` convention; ≥ `LUA_MINSTACK` growable slots for C functions).
  FR2 frame shifting stays invisible to embedders.
- Panic/error semantics stay ABI-identical; internal unwinding may use C++ EH on cold paths
  (the compiler now generates the per-arch unwind info the .dasc files maintained by hand).

---

## 12. Performance Engineering Checklist (how LJX intends to win)

| Area | Parity mechanism | Headroom beyond LuaJIT |
|------|------------------|------------------------|
| Interpreter | musttail CPS + preserve_none, replicated dispatch, pinned context | PGO/BOLT over per-op functions; BMI2/`ubfx` decode; APX (32 GPRs) build flavor |
| Values | 8-byte NaN box, ordered tags | AArch64 `ldp/stp` pair moves for frame setup/returns |
| Tables/strings | sid hashing, Brent chaining, dead-key stability | SIMD strcmp/hash on 16-byte-padded strings; CRC32C/AES dense hash |
| GC | identical barrier fast paths | arena+bitmap sweep, generational minor GC, huge pages, bounded atomic |
| JIT | same IR/fold/LSRA/backward codegen | consteval fold table (no build drift), dual-mapped mcode option, background-assembly seam |
| FFI | ctype-guard specialization | finished vector-arg classification; stable ctype chunks |
| Build | — | LTO+PGO reference builds; per-benchmark flamegraph parity gates vs. LuaJIT in CI |

Benchmark methodology: LuaJIT 2.1 (this fork) is the baseline; the CI perf suite runs the
classic LuaJIT bench set + real-world workloads, pinned cores, ≥30 runs, and reports per-bench
deltas. **A regression against LuaJIT on any tier-1 benchmark blocks the change** once the
corresponding subsystem is declared at-parity.

---

## 13. Implementation Roadmap

| Phase | Deliverable | Exit criterion |
|-------|-------------|----------------|
| 0 | `core/` + `vm/` data model, `C_VirtualArena`, `TValue_t`, object model | layout tests + constexpr encode/decode tests green |
| 1 | Lexer/parser → bytecode, disassembler | passes Lua 5.1 + fork syntax suite; dumps match reference semantics |
| 2 | CPS interpreter + GC v1 (incremental, arena) + stdlib | full test suite green; interpreter ≥ 0.85× LuaJIT-interp |
| 3 | Recorder + IR + fold + snapshots + x86-64 backend | tier-1 benches ≥ 0.9× LuaJIT-JIT |
| 4 | Loop opt, sinking, side traces, stitching, AArch64 backend | full bench parity; deopt torture suite green |
| 5 | FFI + callbacks + `string.buffer` | FFI bench parity; libffi-oracle ABI tests green |
| 6 | Generational GC, SIMD strings, PGO/BOLT builds, APX flavor | measurable wins on target benches |

---

## 14. Core Interface Index

| Header | Key interfaces |
|--------|----------------|
| `core/Config.hpp` | platform/attribute macros, `EArch`, `kCacheLineSize` |
| `core/Types.hpp` | fixed types, Hungarian tag reference, base concepts |
| `core/Memory.hpp` | `C_VirtualArena`, `GcRef_t`, `MRef_t`, `C_SegregatedAllocator` |
| `vm/Value.hpp` | `EValueTag`, `TValue_t` |
| `vm/Object.hpp` | `GcHeader_t`, `C_GcString`, `C_GcTable`, `TableNode_t`, `C_GcProto`, `C_GcFunction`, `C_GcUpvalue`, `C_GcUserData`, `C_LuaThread` |
| `vm/Bytecode.hpp` | `EBcOp`, `BcIns_t`, `C_BytecodeEmitter`, `C_BytecodeSerializer` |
| `vm/Frame.hpp` | `EFrameType`, `FrameLink_t`, frame walkers |
| `vm/Interpreter.hpp` | `BcHandler_f`, `EDispatchMode`, `EVmState`, `C_DispatchTable`, `C_HotCountTable`, `C_Interpreter`, `C_Universe` |
| `vm/FastFunc.hpp` | `EFastFunc`, `EFfhResult`, `LibFuncDef_t`, `C_LibraryRegistry` |
| `vm/VmEvent.hpp` | `EVmEvent`, `VmEventData_t`, `C_IVmEventSink` |
| `gc/GarbageCollector.hpp` | `EGcState`, `C_GarbageCollector`, barrier API, `C_GcArena` |
| `rt/StringInterner.hpp` | `C_StringInterner`, `EStringHashAlgo` |
| `rt/Meta.hpp` | `EMetaMethod`, `C_MetaResolver` |
| `rt/StringBuffer.hpp` | `StrBuf_t`, `C_StringBuffer` |
| `fe/Lexer.hpp` | `ETokenKind`, `Token_t`, `C_Lexer` |
| `fe/Parser.hpp` | `EExpKind`, `ExpDesc_t`, `C_FuncState`, `C_Parser` |
| `jit/Ir.hpp` | `EIrOp`, `EIrType`, `IrIns_t`, `IrRef_t`, `TRef_t`, `C_IrBuffer` |
| `jit/Fold.hpp` | `FoldRule_t`, `C_FoldEngine`, `SearchFoldHashParams`/`BuildFoldTable` (consteval) |
| `jit/Recorder.hpp` | `C_TraceRecorder`, `ETraceError`, `EPostProc`, `ETraceLink`, `C_LoopOptimizer`, `C_SinkOptimizer` |
| `jit/Snapshot.hpp` | `Snapshot_t`, `SnapEntry_t`, `C_SnapshotWriter`, `ExitState_t`, `C_DeoptEngine` |
| `jit/Backend.hpp` | `RegSet_t`, `RegCost_t`, `C_AsmContext` (reverse-LSRA services), `IsMachineBackend`, `C_TraceAssembler`, `C_MachineCodeArena` |
| `jit/Trace.hpp` | `C_GcTrace`, `C_TraceCache`, `C_JitEngine`, penalty/blacklist tuning constants |
| `ffi/Ffi.hpp` | `CType_t`, `C_CTypeRegistry`, `ArgPlan_t`, `IsAbiPolicy`, `C_CallbackBridge` |
| `api/Engine.hpp` | `C_LuaEngine`, compat-layer surface |

All headers compile standalone under `-std=c++20` with clang ≥ 17 and GCC ≥ 13
(`ljx/tools/all_headers.cpp` is the aggregation TU used as the CI syntax gate).
