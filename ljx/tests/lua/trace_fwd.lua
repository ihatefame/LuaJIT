-- Store-to-load forwarding hazards: aliased references, helper calls between
-- store and load, mixed constant/computed slot addresses.

-- 1. two fields, same object: dot's loads forward from scale's stores
local Vec = {}
Vec.__index = Vec
local v = setmetatable({x = 1, y = 2}, Vec)
function Vec:scale(k) self.x = self.x * k; self.y = self.y * k; return self end
function Vec:dot(o) return self.x * o.x + self.y * o.y end
local acc = 0
for i = 1, 2000 do
  v:scale(1.001)
  acc = acc + v:dot(v)
end
print(math.floor(acc))

-- 2. the same table through TWO variables: the alias must be respected
local a = {n = 0}
local b = a
local s = 0
for i = 1, 2000 do
  a.n = i
  s = s + b.n   -- must see a.n's store
end
print(s)

-- 3. a key-creating call between store and load: no stale forwarding
local t = {v = 0}
local u = {}
local sum = 0
for i = 1, 2000 do
  t.v = i
  u["k" .. (i % 8)] = i   -- key creation / rehash in another table
  sum = sum + t.v
end
print(sum)

-- 4. store into one table, read the same-named field of another
local p, q = {f = 1}, {f = 100}
local r = 0
for i = 1, 2000 do
  p.f = i
  q.f = -i
  r = r + p.f + q.f   -- +i - i = 0
end
print(r)
