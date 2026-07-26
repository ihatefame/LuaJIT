-- Whole-function JIT: correctness of compiled numeric functions.

-- 1. plain recursion
local function fib(n) if n < 2 then return n end return fib(n-1) + fib(n-2) end
for i = 1, 20 do fib(5) end               -- warm up past the compile threshold
print(fib(20))                            -- 6765

-- 2. two parameters and a non-commutative body
local function ack(m, n)
  if m < 1 then return n + 1 end
  if n < 1 then return ack(m - 1, 1) end
  return ack(m - 1, ack(m, n - 1))
end
for i = 1, 20 do ack(1, 1) end
print(ack(2, 3))                          -- 9

-- 3. a compiled function called with a NON-number must fall back cleanly
local function dbl(x) return x * 2 end
for i = 1, 20 do dbl(1) end
local okstr = pcall(dbl, "nope")
print(dbl(21), okstr)                     -- 42  false

-- 4. the self-reference upvalue is reassigned after compilation
local function inc(n) if n <= 0 then return 0 end return inc(n-1) + 1 end
local saved = inc
for i = 1, 40 do saved(3) end
inc = function(n) return 100 end
print(saved(3))                           -- 101, NOT 3

-- 5. runaway recursion raises instead of smashing the native stack
local function runaway(n) return runaway(n - 1) + 1 end
for i = 1, 40 do pcall(runaway, 1) end
print(pcall(runaway, 1))                  -- false  stack overflow

-- 6. NaN comparison semantics survive compilation
local function cmp(a, b)
  if a < b then return 1 end
  if a >= b then return 2 end
  return 3
end
for i = 1, 40 do cmp(1, 2) end
local nan = 0/0
print(cmp(1, 2), cmp(2, 1), cmp(nan, 1))  -- 1  2  2  (NaN: a<b false, a>=b true)

-- 7. division, negation, constants on the left
local function mix(x) return -(100 / x) + 3 end
for i = 1, 40 do mix(2) end
print(mix(4))                             -- -22
