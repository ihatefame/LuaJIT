-- Allocation inside traces: TNew, key creation, concat and # are helper
-- calls preceded by a full snapshot write-back — which is also what makes a
-- collection triggered inside them safe.

-- 1. t[#t+1] append pattern
local t = {}
for i = 1, 3000 do t[#t + 1] = i * 2 end
print(#t, t[1], t[3000])

-- 2. fresh table per iteration, filled by an inner loop
local m = {}
for i = 1, 400 do
  m[i] = {}
  for j = 1, 10 do m[i][j] = i * j end
end
print(m[1][1], m[400][10], #m)

-- 3. string building: concat of strings and numbers
local n = 0
for i = 1, 3000 do
  local s = "key" .. (i % 100) .. "!"
  n = n + #s
end
print(n)

-- 4. GC pressure: enough garbage tables to force collections INSIDE the
--    compiled loop, with live data that must survive them
local keep = {}
for i = 1, 120000 do
  local tmp = {i, i + 1, i + 2}
  keep[(i % 50) + 1] = tmp
end
local sum = 0
for i = 1, 50 do sum = sum + keep[i][1] end
print(sum % 100000)

-- 5. dictionary built with computed string keys (Cat + hash key creation)
local d = {}
for i = 1, 500 do d["k" .. i] = i end
print(d.k1, d.k500)

-- 6. growing array read back through # in the same loop
local q = {}
local steps = 0
for i = 1, 2000 do
  q[#q + 1] = i
  if #q % 500 == 0 then steps = steps + 1 end
end
print(#q, steps)
