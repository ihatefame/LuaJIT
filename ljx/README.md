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

1. **Scans** the bytecode from `FORI+1` to the matching `FORL`. Only
   straight-line number arithmetic is accepted (`Add/Sub/Mul/Div` in all
   `VV`/`VN`/`NV` forms, `Mov`, `Unm`, `KShort`, `KNum`); anything else — a
   call, a branch, a table op — rejects the loop, permanently, for that site.
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

`LJX_JITDEBUG=1` traces every compile decision (accepted, or why rejected).

## Benchmarks

Best-of-7, this machine, against the LuaJIT 2.1 built in `../src`:

| bench | LJX (+loop JIT) | LuaJIT `-joff` | vs. interp | LuaJIT (JIT) | vs. LJ JIT |
|-------|----------------:|---------------:|-----------:|-------------:|-----------:|
| loop  | **0.075s** | 0.323s | **4.28× faster** | 0.061s | 1.23× |
| str   | **0.131s** | 0.142s | **1.08× faster** | 0.066s | 1.99× |
| fib   | 0.415s | 0.325s | 0.78× | 0.058s | 7.20× |
| tab   | 0.088s | 0.062s | 0.70× | 0.031s | 2.80× |

Reading this honestly:

- **`loop`** is what the JIT was built for: 4.3× faster than LuaJIT's
  hand-written assembly interpreter and within **1.23×** of its full trace
  compiler.
- **`str`** now edges past the assembly interpreter thanks to an allocation-free
  concat path and a hand-rolled integer formatter.
- **`fib`** (recursion) and **`tab`** (table stores) have no JIT coverage yet —
  they run purely interpreted, where the musttail CPS interpreter sits 22–43%
  behind hand-written assembly. Function calls and table access in compiled
  code are the next two increments.

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
