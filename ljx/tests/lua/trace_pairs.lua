-- pairs()/next in traces: the iterator becomes a CallIter helper whose two
-- results come back through typed in-place reloads. Key-type changes and the
-- end of the iteration both surface as reload guard exits.

-- 1. pure hash part, string keys
local t = {}
for i = 1, 500 do t["k" .. i] = i end
local s, n = 0, 0
for k, v in pairs(t) do s = s + v; n = n + 1 end
print(s, n)

-- 2. mixed array + hash: key type flips from number to string mid-walk
local m = {}
for i = 1, 200 do m[i] = i end
for i = 1, 200 do m["s" .. i] = i * 10 end
local sa, sh = 0, 0
for k, v in pairs(m) do
  if type(k) == "number" then sa = sa + v else sh = sh + v end
end
print(sa, sh)

-- 3. empty table, then single-entry table
local e = {}
local ne = 0
for _ in pairs(e) do ne = ne + 1 end
local one = { only = 42 }
local vo = 0
for k, v in pairs(one) do vo = v end
print(ne, vo)

-- 4. early break out of a hot pairs loop
local big = {}
for i = 1, 400 do big["b" .. i] = i end
local cnt = 0
for k, v in pairs(big) do
  cnt = cnt + 1
  if cnt == 100 then break end
end
print(cnt)

-- 5. nested pairs: outer table of inner tables
local outer = {}
for i = 1, 60 do
  local inner = {}
  for j = 1, 20 do inner["f" .. j] = j end
  outer["o" .. i] = inner
end
local tot = 0
for _, inner in pairs(outer) do
  for _, v in pairs(inner) do tot = tot + v end
end
print(tot)

-- 6. explicit next() in a while loop — same builtin, different shape
local w = {}
for i = 1, 300 do w[i] = i end
local k, v = next(w)
local sw = 0
while k do
  sw = sw + v
  k, v = next(w, k)
end
print(sw)

-- 7. values of mixed types: reload value-type guard must sort them out
local mixed = {}
for i = 1, 150 do mixed["n" .. i] = i end
for i = 1, 150 do mixed["t" .. i] = "x" end
local nn, ns = 0, 0
for _, v in pairs(mixed) do
  if type(v) == "number" then nn = nn + 1 else ns = ns + 1 end
end
print(nn, ns)
