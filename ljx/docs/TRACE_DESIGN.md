# LJX Trace Compiler — design

The two existing tiers are pattern matchers: they recognize *shapes* (a counted
numeric loop; a numeric-closed function) and contribute nothing outside them.
On realistic object-oriented code both reject every region, which is why
`bench_real` sits ~18× behind LuaJIT's trace compiler.

A trace compiler is indifferent to shape. It records the bytecodes that
*actually execute*, specializes on the types it *actually observes*, and leaves
through a side exit when an assumption breaks. That is the mechanism this
document specifies.

## 1. Pipeline

```
hot back-edge ──► record ──► optimize ──► regalloc ──► emit ──► link
                    │                                            │
                    └──── abort (blacklist the site) ◄───────────┘
       guard fails at run time ──► side exit ──► restore ──► interpreter
```

## 2. Trigger and recording

* **Trigger.** Loop back-edges (`Loop`, `ForL`, `IterL`) decrement a hot
  counter. On underflow, recording starts at the loop header.
* **Mechanism.** The dynamic dispatch table is swapped for one whose every
  entry is `RecordAndExecute`. That single function records the instruction
  about to run — reading the *live operand values* so it can specialize — and
  then tail-calls the real handler. This is the architecture's stated
  instrumentation contract: no per-instruction check exists in the normal
  dispatch path.
* **Stop conditions.** Returning to the start PC closes the loop (success);
  exceeding the instruction budget, hitting an unrecordable operation, or
  leaving the enclosing frame aborts. Aborts blacklist the site with an
  exponential penalty so a pathological loop is not re-recorded forever.

## 3. IR

Linear, typed, SSA. One 64-bit instruction per operation:

```
IrIns_t { uint16 rOp1; uint16 rOp2; uint8 eOp; uint8 eType; uint16 rPrev; }
```

`rPrev` chains instructions of the same opcode for CSE; constants are interned
below a bias so `ref < kBias` identifies a constant with one compare — the
scheme `ARCHITECTURE.md` §7.1 specifies.

Operations: `SLoad` (read a stack slot, carrying a type guard), `KNum/KGc/KPri`
(constants), arithmetic, comparison guards, `ARef/HRefK` (table element
address), `ALoad/HLoad/AStore/HStore`, `FLoad` (object field), `Guard*`
(metatable identity, table version), `Phi` (loop-carried), `Loop` (the
back-edge marker).

## 4. Specialization decisions

| observed | recorded as |
|----------|-------------|
| slot holds a number | `SLoad` + number guard; later reads reuse the ref |
| table array element in range | index guard + `ALoad` |
| table hash hit at its main position | `HRefK` (constant node index) + key guard + `HLoad` |
| **method lookup that misses the receiver and resolves through `__index`** | guard receiver's metatable identity + both table versions, then the result is a **constant** |
| constant callee | inline the callee's bytecode into the trace |

The method-lookup row is the important one: it is the shape of every call in
OO Lua, and the interpreter's inline cache has already established exactly the
predicate that makes it foldable — identity plus version of the two tables
involved. Under those guards the resolved method is a compile-time constant,
which in turn makes the call site monomorphic and inlinable.

## 5. Inlining calls

Recording continues through a call to a Lua function whose identity is
constant under guards. The recorder pushes a frame (base offset `+= A + 2`) and
keeps going; `Ret` pops it. A trace therefore spans many Lua frames, which is
what removes call overhead from method-heavy code.

## 6. Snapshots and exits

Every guard is associated with a snapshot: the list of
(slot index, IR ref) pairs that are live, plus the resume PC and frame depth.
On a side exit the stub writes those values back into the Lua stack, restores
the frame links for any inlined frames, sets the thread base, and returns the
resume PC — the interpreter simply continues from there. Because the whole
machine state is reconstructible from the snapshot, guards are free to fail at
any point.

## 7. Optimization

Applied while emitting, not as separate passes:

* constant folding,
* CSE over per-opcode chains (bounded by operand refs, as in `ARCHITECTURE.md`),
* **redundant guard elimination** — a slot already guarded as a number, or a
  table whose version was already guarded, needs no second guard,
* load forwarding — a load from a location already loaded (and not stored to
  since) reuses the earlier ref.

Guard elimination is the one that matters most: in a loop body the same
receiver type, metatable and versions are re-checked on every iteration by the
interpreter, and the trace checks them once.

## 8. Register allocation and code

Linear scan over the finished IR: numbers to xmm, tagged values and pointers to
GP registers, spilling to a native frame when pressure demands. Guards emit a
conditional branch to a per-snapshot exit stub. The loop back-edge becomes a
real backwards jump, so an iteration executes with no dispatch, no type checks
that were already proven, and no call overhead for inlined frames.

## 9. Safety obligations

* **GC.** Compiled code may allocate (table stores can grow). Every point that
  can allocate is a safe point: live values are written back to the Lua stack
  first, exactly as a snapshot restore would.
* **Deoptimization must be exact.** The differential test (`LJX_NOJIT=1`)
  compares compiled and interpreted output byte-for-byte on every script; a
  trace that restores the wrong state fails it.
* **Invalidation.** A trace guards the table versions it depends on, so a
  mutated class invalidates it through an ordinary guard failure rather than
  through any separate bookkeeping.
