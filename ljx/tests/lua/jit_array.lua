-- 1. plain array read
local t = {}
for i = 1, 100 do t[i] = i * 3 end
local s = 0
for i = 1, 100 do s = s + t[i] end
print(s)                                  -- 3*5050 = 15150

-- 2. array write into a pre-sized array, then verify
local u = {}
for i = 1, 50 do u[i] = 0 end            -- grow (interpreted)
for i = 1, 50 do u[i] = i * i end        -- in-range writes (JIT)
local s2 = 0
for i = 1, 50 do s2 = s2 + u[i] end
print(s2)                                 -- sum of squares 1..50 = 42925

-- 3. mid-loop bail: read past the end of the array
local v = {}
for i = 1, 10 do v[i] = 1 end
local n = 0
for i = 1, 20 do
  local x = v[i]
  if x == nil then n = n + 100 else n = n + x end
end
print(n)                                  -- 10*1 + 10*100 = 1010

-- 4. mid-loop bail: a non-number element in the middle
local w = {}
for i = 1, 10 do w[i] = i end
w[5] = "five"
local c = 0
local ok = pcall(function()
  for i = 1, 10 do c = c + w[i] end
end)
print(ok, c)                              -- false (error at i=5), c = 1+2+3+4 = 10

-- 5. non-integral loop bounds with an array (entry bail)
local z = {}
for i = 1, 10 do z[i] = 2 end
local q = 0
for i = 1.5, 9.5 do q = q + (z[math.floor(i)] or 0) end
print(q)                                  -- 8 iterations * 2 = 16

-- 6. descending array walk
local d = {}
for i = 1, 20 do d[i] = i end
local r = 0
for i = 20, 1, -1 do r = r + d[i] end
print(r)                                  -- 210
