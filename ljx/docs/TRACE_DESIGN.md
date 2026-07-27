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
| table array element in range | exactness guard on the index, unsigned bound guard, `LoadTV` |
| hash lookup, any string key | main position from the **loaded** mask and the key's string id; node key guard; `LoadTV` |
| key absent from the receiver | the same walk, ending in a chain-terminator guard — absence is **proved**, not assumed |
| method resolved through `__index` above a proved miss | guards on the metatable's identity and the two tables' versions; the result is then a **constant** |
| constant callee | inline the callee's bytecode into the trace |

Two of these rows deserve their exact wording.

**The hash walk is real code, not a fold.** The main position is computed the
way the runtime computes it, from the mask loaded out of the table and the id
loaded out of the key. Nothing is guarded about the table's shape, so one trace
serves every instance of a class and survives a rehash instead of exiting on
one. A constant node index would have needed a mask guard and would have been
wrong without one — the node array is reallocated on growth, and an empty hash
part aliases a shared one-node sentinel.

**Only the resolution above the miss is folded.** It is tempting to state the
predicate as "metatable identity plus both versions" and delete the receiver
lookup entirely. That is unsound: neither version belongs to the receiver, so
an instance that acquires its own binding for the key — a per-instance method
override, a memoized field — would be ignored and the trace would keep
returning the class's value forever. The receiver's own version cannot be added
to the predicate either; that would pin the trace to a single object. So the
receiver's chain stays as guarded code and only what sits above it folds.
`tests/lua/trace.lua` installs an override mid-loop and checks the trace
notices.

## 5. Inlining calls

Recording continues through a call to a Lua function whose identity is constant
under a guard. The recorder does not model the frame itself — it reads the
base the interpreter is actually running on, so the frame chain follows for
free — and the two words a call writes (the callee at `base-2`, the link at
`base-1`) are ordinary slot assignments whose values are trace constants. A
snapshot therefore reconstructs every inlined frame without any special
machinery, and a trace spans many Lua frames.

## 5a. Iterators

`for i, v in ipairs(t)` records as the array access it is: bump the index,
guard it below the array size, load the element — the bound guard doubling as
the loop exit. `for k, v in pairs(t)` cannot be an address computation (the
successor of a hash key is wherever the node array says it is), so it records
as a `CallIter` helper: the recorder *simulates* one `next` step at record
time to learn the result types, emits the call, and re-reads the two result
slots with `SReload` — a typed in-place load that is never hoisted and never
CSE'd, because its whole meaning is "what the call just wrote". The key's
type guard is also the loop's exit: a different key kind, or the iteration
ending in nil, resumes the interpreter at the `IterC`, which re-runs the
(idempotent) step and carries on.

A mixed table walks two phases — numeric array keys, then hash-part keys —
and one trace can only specialize on one key type. The failing entry guard
grows a *variant*: a second trace at the same head PC, specialized to the
types present now, reached through the first trace's patched exit stub. Two
rules keep a variant family from degenerating: the PC→trace map keeps the
FIRST trace (the head of the chain — a newer variant in the map would put
the wrong specialization first and grow the chain by one trace per phase
change, without bound), and an exit is never *linked* to a trace with the
same start PC (its entry checks are exactly what just failed; a link would
be a native cycle of type checks with no body between them — only a fresh
variant makes progress).

At run time the helper steps the table directly (`vm::TableNext`), skipping
the C-call framing, and a one-entry position hint in the universe remembers
which node the previous key was found at — validated by re-reading that
node's key, so a stale hint misses instead of misdirecting. What this tier
still lacks against LuaJIT's `ITERN` is *positional* iteration: LuaJIT
carries a hidden node index through the loop and never re-derives the
position from the key at all. Carrying that index as a trace-internal SSA
value (slots keep the key, so every deopt stays exact) is the designed next
step if pairs-heavy workloads warrant it.

## 6. Snapshots, exits, and the two kinds of slot

A slot that is **read before it is written** gets an entry `SLoad`, is carried
in a register across the back edge, and appears in every snapshot from that
point on. A slot that is **only ever written** is not carried in a register at
all; it is stored to the Lua stack once, at the back edge. That single store is
what makes deoptimization exact without a loop-carry analysis: a guard *after*
the assignment is covered by the snapshot, and a guard *before* it needs
whatever the previous iteration left in the slot — which is exactly what the
back-edge store put there.


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

## 8. Loop-invariant hoisting

The IR is emitted in three groups: the entry `SLoad`s, then everything whose
operands are loop-invariant, then the body. For a field access on an invariant
receiver the second group absorbs the entire address computation — mask load,
node-array load, main-position arithmetic, node address, and the node **key**
load with its guard, since a raw store rewrites a node's value word and never
its key. What is left in the loop is the value load, its type guard, the
arithmetic, and the store.

Three rules keep it sound:

* a stack slot is invariant only when the recorded iteration left it holding
  the value it was entered with;
* only header words a trace can never write are hoistable loads — array,
  metatable, next, node array, array size, hash mask. The version word is
  excluded, because a hash store bumps it;
* a guard that moves into the pre-roll is re-pointed at the entry snapshot. It
  runs before the body has changed anything, so resuming at the trace head with
  the interpreter's own state is always correct.

## 9. Register allocation and code

Linear scan over the reordered IR: numbers to xmm, tagged values and pointers
to GP registers, spilling to a native frame when pressure demands. Guards emit
a conditional branch to a per-snapshot exit stub that writes the live values
back to the Lua stack and returns the exit number.

Anything whose definition runs **once** — the entry loads and everything
hoisted — stays live across the back edge, or the next iteration would read a
register that the body has since reused. Hoisted values may still spill, but
their spill store is emitted at the *definition*, in the pre-roll: emitting it
at the eviction point would place it inside the loop, where on the second
iteration the register holds something else and the store would corrupt the
home slot. Entry loads are never spilled at all, because the back-edge copies
write to their registers.

## 10. Safety obligations

* **GC.** Compiled traces do not allocate at all — anything that could
  (a table constructor, a closure, a new table key, a concat, a C call) aborts
  recording — so no collection can begin while a trace is running. Two things
  still have to be arranged. The interpreter raises the thread's high-water
  mark past the highest slot a trace touches when the trace returns, since a
  trace writes inlined frames that no `FuncF` ever accounted for. And every GC
  object a trace bakes an address into — a compared constant, a metatable whose
  version word it reads, an upvalue whose cell it loads — is anchored in a
  dedicated pin table, which is an ordinary GC root; a stale object then fails
  a guard instead of being dereferenced after free. The pins deliberately do
  NOT live in the registry: pinned values become table *keys*, and the stdlib
  stores named objects in the registry under string keys — one shared table
  and a pinned `"next"` string would overwrite the `next` function `pairs`
  hands out.
* **Recording is not re-entrant.** A metamethod or C function that calls back
  into the interpreter would otherwise have its bytecodes appended to the trace
  with a bogus frame base, so `C_Interpreter::Call` aborts an in-progress
  recording. For the same reason the loop and function tiers, and the
  collector's step check, stand down while the recorder is running: a tier that
  executes a whole loop or function natively would hide those bytecodes from
  it.
* **Deoptimization must be exact.** The differential test (`LJX_NOJIT=1`)
  compares compiled and interpreted output byte-for-byte on every script; a
  trace that restores the wrong state fails it.
* **Invalidation.** A trace guards the table versions it depends on, so a
  mutated class invalidates it through an ordinary guard failure rather than
  through any separate bookkeeping.
