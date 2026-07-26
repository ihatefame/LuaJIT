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
- **GC** — precise mark-sweep from interpreter safe points (v1; the barrier
  call sites for the incremental/generational collector are already planted).
- **Front end** — a single-pass, no-AST recursive-descent parser emitting
  registerized bytecode through an `ExpDesc_t` discharge state machine
  (delayed codegen, jump-list threading, test-and-copy materialization,
  parse-time constant folding, concat fusion, operand-kind `VN`/`NV`/`VV`
  variants).
- **Interpreter** — a continuation-passing tail-call interpreter
  (`[[clang::musttail]]` + `preserve_none`), one function per opcode,
  replicated dispatch, FR2 two-slot frames on the value stack, errors via
  setjmp/longjmp (see below).
- **Language** — locals/upvalues/globals, `if`/`while`/`repeat`/numeric &
  generic `for`, functions/closures/recursion/method calls, multiple returns
  and tailcalls, `and`/`or` short-circuit, metatables (`__index`,
  `__newindex`, `__add` … , `__eq`), `pcall`/`error`.
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

## Benchmarks

Best-of-7, this machine, against the LuaJIT 2.1 built in `../src`:

| bench | LJX (+loop JIT) | LuaJIT `-joff` | vs. interp | LuaJIT (JIT) | vs. LJ JIT |
|-------|----------------:|---------------:|-----------:|-------------:|-----------:|
| tab   | **0.025s** | 0.068s | **2.72× faster** | 0.032s | **0.77× — faster** |
| array | **0.0095s** | 0.041s | **4.34× faster** | 0.0076s | 1.25× |
| loop  | **0.075s** | 0.323s | **4.28× faster** | 0.062s | 1.22× |
| str   | **0.133s** | 0.142s | **1.06× faster** | 0.065s | 2.05× |
| fib   | 0.434s | 0.330s | 0.76× | 0.059s | 7.32× |

Reading this honestly:

- **`tab`** (fill a table, then sum it) is the one benchmark where LJX beats
  LuaJIT's full trace compiler, and the reason is a strategy difference, not
  raw codegen: pre-growing the array once at loop entry avoids the incremental
  reallocation LuaJIT does as the table grows.
- **`array`** and **`loop`** are 4.3× faster than LuaJIT's hand-written
  assembly interpreter and land within **1.22–1.25×** of its trace compiler.
- **`str`** edges past the assembly interpreter thanks to an allocation-free
  concat path and a hand-rolled integer formatter.
- **`fib`** is the honest gap: recursion means function calls, which compiled
  code does not cover, so it runs purely interpreted — and there the musttail
  CPS interpreter still sits ~31% behind hand-written assembly. Calls in
  compiled code are the next increment.

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
