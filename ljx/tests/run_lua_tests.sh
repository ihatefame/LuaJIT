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
check regress "$(printf '6765\n5050\n1024\n-2\n15\ntrue\nnested-ok\n42\n3\n120')"
[ $fail -eq 0 ] && echo "ALL LUA TESTS PASSED" || echo "SOME LUA TESTS FAILED"
exit $fail
