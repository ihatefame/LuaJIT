# LJX — a ground-up C++20 reimplementation of LuaJIT

LJX is a clean-room Lua 5.1 VM built in modern C++20, following the architecture
in [`ARCHITECTURE.md`](ARCHITECTURE.md). This tree is a **working interpreter**
(phases 0–2 of the roadmap), not yet the trace JIT.

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

## Benchmarks

Interpreter-vs-interpreter is the honest comparison for this phase (LJX has no
JIT yet). Best-of-5, this machine, against the LuaJIT 2.1 in `../src`:

| bench | LJX (interp) | LuaJIT `-joff` | ratio | LuaJIT (JIT) |
|-------|-------------:|---------------:|------:|-------------:|
| fib   | 0.31s | 0.27s | 1.17× | 0.04s |
| loop  | 0.27s | 0.25s | 1.06× | 0.03s |
| tab   | 0.06s | 0.06s | 1.13× | 0.02s |
| str   | 0.29s | 0.09s | 3.19× | 0.06s |

On compute- and table-bound code the musttail CPS interpreter lands within
**6–17%** of LuaJIT's hand-written assembly interpreter — validating the core
thesis of the redesign. Strings are the current weak spot (per-op `std::string`
construction + `snprintf` number formatting + stop-the-world interning churn);
a rope/buffer path and a fast number formatter are the obvious next wins.
Against full LuaJIT with the trace compiler, LJX is 5–9× slower, as expected
until the JIT lands (roadmap phase 3).

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
