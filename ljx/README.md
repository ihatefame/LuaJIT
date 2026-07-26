# LJX — a ground-up C++20 reimplementation of LuaJIT

LJX is a clean-room Lua 5.1 VM built in modern C++20, following the architecture
in [`ARCHITECTURE.md`](ARCHITECTURE.md). This tree is a **working interpreter
plus a native-code compiler for hot numeric loops** (roadmap phases 0–2, and
the first increment of phase 3).

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

Best-of-7, this machine, against the LuaJIT 2.1 built in `../src`:

| bench | LJX | LuaJIT `-joff` | vs. interp | LuaJIT (JIT) | vs. LJ JIT |
|-------|----:|---------------:|-----------:|-------------:|-----------:|
| fib   | **0.061s** | 0.335s | **5.45× faster** | 0.063s | **0.98× — faster** |
| loop  | **0.075s** | 0.327s | **4.39× faster** | 0.062s | 1.21× |
| array | **0.0094s** | 0.041s | **4.35× faster** | 0.0073s | 1.28× |
| tab   | **0.027s** | 0.069s | **2.58× faster** | 0.033s | **0.80× — faster** |
| str   | **0.126s** | 0.142s | **1.13× faster** | 0.063s | 1.98× |
| real  | 0.051s | 0.049s | 0.95× | 0.0029s | 17.7× |

Reading this honestly:

- **`tab`** (fill a table, then sum it) is the one benchmark where LJX beats
  LuaJIT's full trace compiler, and the reason is a strategy difference, not
  raw codegen: pre-growing the array once at loop entry avoids the incremental
  reallocation LuaJIT does as the table grows.
- **`array`** and **`loop`** are 4.3× faster than LuaJIT's hand-written
  assembly interpreter and land within **1.22–1.25×** of its trace compiler.
- **`str`** edges past the assembly interpreter thanks to an allocation-free
  concat path and a hand-rolled integer formatter.
- **`fib`** went from 0.434s to **0.062s** once the function JIT landed — 5.4×
  faster than LuaJIT's assembly interpreter and level with its trace compiler.
  Recursion is now real machine recursion.
- **`str`** is the remaining soft spot: string building still allocates and
  interns per operation, which the JIT does not touch.

### Where the JIT does *not* reach — and what that costs

`real` is deliberately in the table: objects with methods, string-keyed
dictionaries, nested tables. **Both JIT tiers reject every region in it** —
the loop JIT because the bodies contain calls and string-keyed access, the
function JIT because the functions touch tables. So `real` measures the
interpreter alone, and against LuaJIT's trace compiler that is a **17.7× gap**.

That number is the honest state of this project. The two tiers are pattern
matchers: they are fast on the shapes they recognize and contribute nothing
elsewhere. Closing it needs the tier `ARCHITECTURE.md` actually specifies — a
trace compiler that records whatever executes, specializes on observed types,
and side-exits on a guard failure — because that mechanism is indifferent to
the *shape* of the code it compiles.

What has been done for `real` so far is architectural rather than
pattern-matched, and applies everywhere: link-time optimization (so runtime
fast paths inline into interpreter handlers), inline caches for
metatable-resolved lookups, and raw-store fast paths for table writes. Together
they took it from 0.075s to 0.051s — from 1.55× *slower* than LuaJIT's assembly
interpreter to 0.95× of it — without a line of code that knows what a benchmark
looks like.

Every benchmark result is checked against the interpreter (`LJX_NOJIT=1`) and
must match exactly.

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
