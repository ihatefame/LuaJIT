-- Narrowing: integral loop indexes lose their per-iteration exactness guards.
-- Every case below must match the interpreter and LuaJIT exactly.

-- descending across zero
local s = 0
for i = 200, -200, -1 do s = s + i end
print(s)

-- fractional step: narrowing must NOT apply, results stay exact
local f = 0
for i = 1, 100, 0.5 do f = f + 1 end
print(f)

-- a compiled loop re-entered with a fractional start: the hoisted ChkInt32
-- entry guard must exit to the interpreter, not truncate
local t = {}
for i = 1, 300 do t[i] = i end
local function run(a, b)
  local s2 = 0
  for i = a, b do s2 = s2 + (t[i] or 0) + 1 end
  return s2
end
print(run(1, 300), run(0.5, 10.5))

-- a ~54-iteration loop re-entered many times: recording can capture the
-- LAST iteration, whose ForL takes the exit direction
local total = 0
local function inner(n)
  local s3 = 0
  for i = 1, n do s3 = s3 + i end
  return s3
end
for k = 1, 200 do total = total + inner(54) end
print(total)

-- index arithmetic stays integral through addition
local arr = {}
for i = 1, 400 do arr[i] = i end
local x = 0
for i = 1, 399 do x = x + arr[i + 1] end
print(x)

-- huge bounds: entry guard rejects, interpreter result identical
local big = 0
for i = 2147483000, 2147483040 do big = big + 1 end
print(big)
