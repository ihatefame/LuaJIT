-- Closures and calls under the trace tier.

-- 1. a fresh upvalue-free closure per iteration, called in a hot inner loop:
--    the call guards the dispatch PC, not the closure identity
local function apply(arr, f)
  local r = 0
  for i = 1, #arr do r = r + f(arr[i]) end
  return r
end
local arr = {}
for i = 1, 50 do arr[i] = i end
local total = 0
for pass = 1, 500 do
  total = total + apply(arr, function(x) return x * 2 + 1 end)
end
print(total)

-- 2. two different protos through the same call site: the PC guard must
--    distinguish them
local a2, b2 = 0, 0
for i = 1, 2000 do
  local f
  if i % 2 == 0 then f = function(x) return x + 1 end
  else f = function(x) return x + 100 end end
  local v = f(1)
  if v == 2 then a2 = a2 + 1 else b2 = b2 + 1 end
end
print(a2, b2)

-- 3. closures WITH upvalues created in a loop (helper-allocated; the call
--    keeps the identity guard)
local acc3 = 0
for i = 1, 300 do
  local k = i
  local g = function() return k end
  acc3 = acc3 + g()
end
print(acc3)

-- 4. memoization table: number keys in the hash part, key guarded constant
local memo = {}
local function fib(n)
  if n < 2 then return n end
  local m = memo[n]
  if m then return m end
  m = fib(n - 1) + fib(n - 2)
  memo[n] = m
  return m
end
local s4 = 0
for i = 1, 2000 do s4 = s4 + fib(20) % 97 end
print(s4)

-- 5. a mixed-key table: array part + number keys in the hash part
local mix = {}
for i = 1, 10 do mix[i] = i end
mix[1000000] = 77
mix[2.5] = 88
local s5 = 0
for i = 1, 2000 do s5 = s5 + mix[3] + mix[1000000] + mix[2.5] end
print(s5)

-- 6. leaf C builtins called from compiled loops through the generic helper
local parts = {}
for i = 1, 600 do parts[i] = tostring(i % 50) end
local ln = 0
for i = 1, 600 do ln = ln + #parts[i] end
print(ln)

local mx, mn = 0, 0
for i = 1, 2000 do
  mx = mx + math.max(i % 7, 3)
  mn = mn + math.min(i % 7, 3)
end
print(mx, mn)

local subs = 0
local base = "abcdefghij"
for i = 1, 2000 do
  local piece = string.sub(base, (i % 5) + 1, (i % 5) + 3)
  subs = subs + #piece
end
print(subs)

local built = {}
for i = 1, 1000 do table.insert(built, i * 3) end
print(#built, built[1000], table.concat({"a", "b", "c"}, "-"))
