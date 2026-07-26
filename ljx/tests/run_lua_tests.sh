#!/usr/bin/env bash
# Golden-output regression tests for the LJX interpreter.
set -u
LJX="${LJX:-./ljx}"
DIR="$(dirname "$0")/lua"
fail=0
check() {
  local name="$1"; shift
  local expected="$1"; shift
  local got
  got="$("$LJX" "$DIR/$name.lua" 2>&1)"
  if [ "$got" != "$expected" ]; then
    echo "FAIL $name"
    echo "  expected: $expected"
    echo "  got:      $got"
    fail=1
  else
    echo "ok   $name"
  fi
}
check hello "$(printf 'hello, world\n14\nconcat: 14!')"
# JIT: array reads/writes, mid-loop deopt, guard fallback, descending walks.
check jit_array "$(printf '15150\n42925\n1010\nfalse\t10\n18\n210')"
# Inline caches: method reassignment, own-field shadowing, __index swap,
# metatable removal, and two instances of one class.
check ic_invalidate "$(printf 'A\tA\nB\nC\nB\nD\nnil\nbase/own')"
# Function JIT: recursion, arity 2, non-number fallback, reassigned
# self-reference, runaway recursion, NaN comparisons, constant-left division.
check jit_func "$(printf '6765\n9\n42\tfalse\n101\nfalse\tstack overflow\n1\t2\t2\n-22')"
# JIT: zero-iteration loops, operand aliasing, unary minus, guard bail-out.
check jit_edge "$(printf '0\n0\n-8\ntrue\n-10\n100\n0\nfalse')"
check regress "$(printf '6765\n5050\n1024\n-2\n15\ntrue\nnested-ok\n42\n3\n120')"
# Trace compiler: __index method dispatch, a per-instance override appearing
# mid-loop, two-level inheritance, a mid-iteration type change, array and field
# stores, inlined calls, descending loops and a break out of a traced loop.
# Every line here is byte-identical to LuaJIT's own output.
check trace "$(printf 'dot\t2200\noverride\tclass\tclass\tinstance\tinstance\ninherit\tbase\ndeopt\t450\narray\t98400\nfields\t301\t601\nmisc\t2406\nbreak\t201\ndict\t80\t80\npressure\t5902608.5\t116509\t3058\npoly\t6000')"
# Differential test: compiled loops must produce byte-identical output to the
# interpreter. This is the strongest correctness check on the JIT — every
# script in tests/lua is run both ways and the outputs compared.
for f in "$DIR"/*.lua; do
  name="$(basename "$f" .lua)"
  case "$name" in bench_*) continue;; esac
  jit_out="$("$LJX" "$f" 2>&1)"
  int_out="$(LJX_NOJIT=1 "$LJX" "$f" 2>&1)"
  if [ "$jit_out" != "$int_out" ]; then
    echo "FAIL differential:$name (JIT output differs from interpreter)"
    diff <(echo "$int_out") <(echo "$jit_out") | head -6
    fail=1
  else
    echo "ok   differential:$name"
  fi
done

[ $fail -eq 0 ] && echo "ALL LUA TESTS PASSED" || echo "SOME LUA TESTS FAILED"
exit $fail
