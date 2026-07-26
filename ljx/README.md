# LJX — a ground-up C++20 reimplementation of LuaJIT

LJX is a clean-room Lua 5.1 VM built in modern C++20, following the architecture
in [`ARCHITECTURE.md`](ARCHITECTURE.md). This tree is a **working interpreter
plus three native-code tiers**: a loop JIT and a whole-function JIT for the
shapes they recognize, and a **trace compiler** that records whatever actually
executes and specializes on the types it observes (roadmap phases 0–2 and most
of phase 3).

## What works today

A complete, self-contained Lua front-end and runtime:

- **Values** — 8-byte NaN-boxed `TValue_t` (GC64 layout), ordered complement
  tags, `std::bit_cast` everywhere (no union UB), intrinsic FP→int kernels.
- **Memory** — one reserved virtual arena; 32-bit granule-scaled compressed
  references (`GcRef_t`/`MRef_t`); bump + segregated-list allocator.
- **Objects** — interned immutable strings (pointer identity; sid-hashed),
  hybrid array+hash tables with main-position chaining, Brent's eviction, and
  dead-key slot stability; closures with open/closed upvalues; protos as one
  colocated allocation with a split constant array.
- **Inline caches** — every bytecode site carries a cache line, and every table
  carries a version counter. A field lookup that misses on the receiver and
  resolves through its metatable's `__index` — the shape of *every* method call
  in object-oriented Lua — is recorded and replayed with four compares instead
  of a metamethod lookup plus a second table search. Precise invalidation: any
  binding change bumps the owning table's version, so a stale line can never be
  used and nothing is ever globally flushed.
- **GC** — precise mark-sweep from interpreter safe points (v1; the barrier
  call sites for the incremental/generational collector are already planted).
- **Front end** — a single-pass, no-AST recursive-descent parser emitting
  registerized bytecode through an `ExpDesc_t` discharge state machine
  (delayed codegen, jump-list threading, test-and-copy materialization,
  parse-time constant folding, concat fusion, operand-kind `VN`/`NV`/`VV`
  variants).
- **Interpreter** — a continuation-passing tail-call interpreter
  (`[[clang::musttail]]`), one function per opcode, replicated dispatch, FR2
  two-slot frames on the value stack, errors via setjmp/longjmp (see below).
  Call frames do not clear their temp slots; the collector clears everything
  above the live top instead, so that cost is paid once per GC rather than on
  every call.
- **Language** — locals/upvalues/globals, `if`/`while`/`repeat`/numeric &
  generic `for`, functions/closures/recursion/method calls, multiple returns
  and tailcalls, `and`/`or` short-circuit, metatables (`__index`,
  `__newindex`, `__add` … , `__eq`), `pcall`/`error`.
- **Function JIT** (`jit/FuncJit.hpp`) — *numeric-closed* functions (numbers
  in ⇒ numbers throughout: arithmetic, comparisons, branches, self-recursion)
  compile to **native code with a plain `double f(double, …)` ABI**, so
  recursion becomes a machine `call`. See "The function JIT" below.
- **Loop JIT** (`jit/LoopJit.hpp`) — hot counted `for` loops whose bodies are
  straight-line number arithmetic are compiled to **x86-64 machine code**: entry
  type guards, all live slots promoted to xmm registers for the whole loop,
  register-to-register SSE, ascending/descending variants selected on the step
  sign, and a clean bail-back to the interpreter when a guard fails. See
  "The loop JIT" below.
- **Stdlib (subset)** — `print type tostring tonumber pairs ipairs next
  setmetatable getmetatable rawget rawset assert error pcall collectgarbage`,
  plus `math`, `string` (`len sub rep byte char`, method syntax), `table`
  (`insert remove`), `os.clock/time`, `io.write`.

- **Trace compiler** — hot back edges are recorded through a dispatch-table
  swap, lowered to a linear typed SSA IR, hoisted, register-allocated and
  emitted as x86-64 with per-snapshot side exits. Records arithmetic,
  comparisons, array and hash access (including variable string keys),
  `__index` method dispatch, upvalue and global reads, and inlines calls whose
  callee is constant under a guard.

### Not yet implemented

Varargs (`...`), coroutines, `goto`, the trace JIT/FFI, and the full stdlib.
Errors use setjmp/longjmp rather than C++ exceptions because exceptions cannot
unwind reliably through `musttail` + `preserve_none` frames — this is also the
scheme the architecture prescribes (LuaJIT does the same). One consequence:
a longjmp does not run C++ destructors, so the few cold library helpers that
build a `std::string` immediately before raising can leak on that error edge
(tracked; the hot path holds no unwind-sensitive C++ objects).

## Build & test

```sh
make            # builds ./ljx and ./ljx_test (requires clang for musttail)
make check      # unit tests + golden-output Lua regression tests
./ljx script.lua
```

## The loop JIT

`C_LoopJit` compiles a counted `for` loop the first time it is entered (a loop
entered once may still iterate millions of times, so entry-counting would miss
exactly the loops worth compiling; rejection is cached, so one-shot loops pay
only microseconds).

What it does:

1. **Scans** the bytecode from `FORI+1` to the matching `FORL`. Accepted:
   straight-line number arithmetic (`Add/Sub/Mul/Div` in all `VV`/`VN`/`NV`
   forms, `Mov`, `Unm`, `KShort`, `KNum`) and **array reads/writes indexed by
   the loop variable** (`t[i]`, one table per loop). Anything else — a call, a
   branch, an upvalue — rejects the loop permanently for that site.
2. **Guards on entry**: the induction triple and every read-before-write input
   slot must hold doubles. Guards run before any state is touched, so a failed
   guard returns `1` and the interpreter runs the loop with no cleanup needed.
3. **Promotes every live slot to an xmm register** for the whole loop —
   loaded once before, stored back once after, so the interpreter and the GC
   observe exactly what they would have. When the body never assigns the loop
   variable (the common case) it reads straight out of the induction register.
4. **Emits** register-to-register SSE, with `dst == lhs` collapsing to a single
   instruction and commutativity exploited for `add`/`mul`. Register copies use
   `movaps`, never `movsd reg,reg` — the latter *merges* the upper 64 bits,
   creating a false dependency that serializes the loop (this one detail was
   worth ~1.9× on the loop benchmark).
5. **Specializes on step sign** at entry, emitting separate ascending and
   descending loops so the iteration test is a single `ucomisd` + `jae`.
6. **Array access** keeps a parallel *integer* induction variable in a GP
   register (so an index needs no per-iteration float→int conversion), hoists
   the array pointer and length out of the loop, and checks the bound once per
   iteration with a 64-bit unsigned compare — which catches negative indices
   for free. Nothing inside compiled code can reallocate the array, so the
   hoisted pointer stays valid.
7. **Pre-grows** the array once at entry, through a runtime call emitted before
   any value is cached in a register (so the VM state is wholly in memory and
   the call is a safe point). Without this, a loop that fills a table from
   empty would deoptimize on its very first store.
8. **Deoptimizes mid-loop** when a bound or element-type guard fails: every
   register-resident value and the induction variable are flushed back to the
   stack and the *remaining* iterations run interpreted. The guard sits at the
   top of the iteration, before any side effect, so the flushed state is
   exactly an iteration boundary.

`LJX_JITDEBUG=1` traces every compile decision (accepted, or why rejected);
`LJX_JITDUMP=<file>` writes the raw machine code for
`objdump -D -b binary -m i386:x86-64 -M intel`; `LJX_NOJIT=1` forces the
interpreter.

**Correctness.** `make check` runs every script in `tests/lua` twice — once
compiled, once with `LJX_NOJIT=1` — and requires byte-identical output. That
differential test, plus a dedicated edge-case suite (zero-iteration loops in
both directions, operand aliasing on non-commutative ops, unary minus,
constant-on-the-left division, out-of-range and wrong-typed array elements,
non-integral bounds, descending array walks), is what keeps the compiler
honest.

## The function JIT

The loop JIT cannot help call-heavy code — there is no loop to compile. The
function JIT closes that gap by exploiting one property:

> A function is **numeric-closed** when every value it computes is a number
> given number arguments.

That single property buys two things the loop JIT has to work for:

* **One guard, at the boundary.** Numbers in ⇒ numbers throughout, so the body
  needs no guards at all and there is no mid-function deoptimization.
* **No allocation ⇒ no GC.** Compiled frames therefore need no presence on the
  Lua stack, so compiled code uses a plain native ABI — `double f(double, …)`
  with arguments in `xmm0..`, slots pinned to `xmm8..xmm15`, and self-recursion
  emitted as a machine `call`.

Compilation happens on the 8th call. The type check the interpreter already
performs at the call boundary *is* the guard; a non-number argument simply
falls through to the interpreter.

Two correctness obligations come with running on the native stack, and both are
covered by tests that fail loudly without the fix:

* **A reassigned self-reference.** Compiled code bakes in the closure it was
  compiled against. If the upvalue holding it is reassigned, the compiled entry
  must not be used — so it is re-validated once per outermost entry (nothing
  inside compiled code can reassign an upvalue, so once is enough).
* **Unbounded recursion.** Native recursion has no Lua-side depth check, so
  every compiled entry tests `rsp` against a limit derived from `RLIMIT_STACK`
  and raises a normal Lua `stack overflow` error instead of running off the C
  stack. Abandoning the native frames via longjmp is safe precisely because
  compiled functions hold no destructors and no VM state.

## Benchmarks

Best-of-5, this machine, against the LuaJIT 2.1 built in `../src`:

| bench | LJX | LuaJIT `-joff` | vs. interp | LuaJIT (JIT) | vs. LJ JIT |
|-------|----:|---------------:|-----------:|-------------:|-----------:|
| fib    | **0.055s** | 0.346s | **6.3× faster** | 0.042s | 1.31× |
| loop   | **0.077s** | 0.353s | **4.6× faster** | 0.061s | 1.26× |
| array  | **0.0092s** | 0.041s | **4.5× faster** | 0.0076s | 1.21× |
| tab    | **0.020s** | 0.053s | **2.6× faster** | 0.025s | **0.83× — faster** |
| str    | **0.095s** | 0.113s | **1.19× faster** | 0.055s | 1.72× |
| real   | **0.0050s** | 0.041s | **8.3× faster** | 0.0020s | 2.4× |
| branch | **0.042s** | 0.066s | **1.57× faster** | 0.026s | 1.67× |

`real` is objects, `__index` method dispatch and string-keyed dictionaries;
`branch` is a branch-heavy loop plus nested loops. Before the trace tier
existed, `real` was **17.7× behind** LuaJIT's trace compiler and `branch`-like
shapes ran in the interpreter. Every benchmark's output is checked against the
interpreter (`LJX_NOJIT=1`) and against LuaJIT itself.

### The three tiers, measured separately

`LJX_NOLOOPJIT=1` and `LJX_NOFUNCJIT=1` disable the two pattern tiers, so each
can be measured on its own. That gives an uncomfortable but useful number:

| bench | all tiers | trace tier only | LuaJIT `-joff` |
|-------|----------:|----------------:|---------------:|
| loop  | 0.076s | 0.101s | 0.353s |
| array | 0.0099s | 0.020s | 0.041s |
| tab   | 0.027s | 0.076s | 0.067s |
| real  | 0.0068s | 0.0067s | 0.048s |

On counted numeric loops the pattern tiers are still **1.3× to 2.8× ahead** of
the trace tier, which is why they keep their precedence. What the trace tier
buys is that it has no shape requirement at all: on `real` it is the only tier
that fires, and it is the whole difference between 0.048s and 0.0068s.

### What each tier can and cannot do

The **loop JIT** recognizes a counted numeric `for` whose body is arithmetic
and array access. The **function JIT** recognizes a *numeric-closed* function —
numbers in, numbers throughout — and compiles it to a plain native
`double f(double, …)`, so recursion becomes a native `call`. Both are pattern
matchers: fast on the shape they recognize, silent everywhere else.

The **trace compiler** recognizes nothing. It records the bytecodes that
actually execute at a hot back edge, specializes on the types actually
observed, and leaves through a side exit when an assumption breaks. It inlines
calls whose callee is constant under a guard, so a trace spans many Lua frames.
`docs/TRACE_DESIGN.md` describes the mechanism; the part that makes it general
rather than a fourth pattern matcher is worth stating here:

> A field access on a class-style object misses on the receiver and resolves
> through the metatable's `__index`. The receiver's own hash chain is emitted
> as **real guarded code** — main position computed from the loaded mask and
> the key's string id, node key compares, chain-terminator guard — so an
> instance that later acquires that field leaves the trace instead of being
> ignored. Only the metatable resolution *above* that miss is folded to a
> constant, under guards on the metatable's identity and the two tables'
> versions. That constant is what makes the call site monomorphic and
> inlinable.

What the trace tier still refuses: varargs, `pcall`, coroutines, string
concatenation, `#`, anything that allocates (a table constructor, a closure, a
new table key), C functions, and metamethods other than a table `__index`.
Those abort recording and blacklist the loop.

Every benchmark result is checked against the interpreter (`LJX_NOJIT=1`) and
must match exactly. `tests/lua/trace.lua` additionally checks the trace tier
against LuaJIT's own output, line for line.

**A note on `preserve_none`.** The architecture calls for it, and the code is
written to use it, but clang 18 does not implement the attribute — the macro
expands to nothing here, so the numbers above are what the interpreter achieves
on the plain SysV convention. Inspecting the emitted handlers shows clang is
already doing the right thing (no prologue, a clean indirect tail jump, the
metamethod path split out of line), so `preserve_none` is upside, not a
prerequisite.

## Layout

```
include/ljx/   public headers (the frozen contracts; compile standalone)
src/core/      arena, allocator, PRNG, FP kernels
src/rt/        string interner, tables, metamethods
src/gc/        garbage collector
src/vm/        universe, bytecode metadata, interpreter, stdlib
src/fe/        lexer, parser
tests/         C++ unit tests + Lua golden-output regression tests
```
