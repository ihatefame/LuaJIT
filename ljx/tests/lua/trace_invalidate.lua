-- Invalidation under STRUCTURAL versions: value stores bump nothing, so every
-- mutation below must be seen through the slot addresses themselves.

-- 1. class method REASSIGNED mid-loop: the trace's inlined call must notice
local C = {} C.__index = C
function C:f() return 1 end
local o = setmetatable({}, C)
local s = 0
for i = 1, 4000 do
  if i == 2000 then C.f = function() return 100 end end
  s = s + o:f()
end
print(s)

-- 2. field read through __index whose value changes mid-loop: the new value
--    must flow out of the same slot with no deopt at all
local D = {limit = 7} D.__index = D
local p = setmetatable({}, D)
local t = 0
for i = 1, 4000 do
  if i == 2000 then D.limit = 11 end
  t = t + p.limit
end
print(t)

-- 3. __index SWAPPED to another table mid-loop (a plain store to mt.__index)
local A1 = {v = 5}
local A2 = {v = 9}
local M = {__index = A1}
local q = setmetatable({}, M)
local u = 0
for i = 1, 4000 do
  if i == 2000 then M.__index = A2 end
  u = u + q.v
end
print(u)

-- 4. setmetatable swap mid-loop
local B1 = {} B1.__index = B1 B1.tag = 1
local B2 = {} B2.__index = B2 B2.tag = 2
local r = setmetatable({}, B1)
local w = 0
for i = 1, 4000 do
  if i == 2000 then setmetatable(r, B2) end
  w = w + r.tag
end
print(w)

-- 5. NEW KEY added to the class mid-loop, shadowing nothing (structural bump)
local E = {} E.__index = E
function E:g() return self.n or 0 end
local e = setmetatable({n = 3}, E)
local z = 0
for i = 1, 4000 do
  if i == 2000 then E.extra = 42 end
  z = z + e:g()
end
print(z)

-- 6. the receiver acquires its OWN key mid-loop (shadows the class value)
local F = {shade = 1} F.__index = F
local fo = setmetatable({}, F)
local y = 0
for i = 1, 4000 do
  if i == 2000 then fo.shade = 50 end
  y = y + fo.shade
end
print(y)

-- 7. dictionary value stores in a hot loop, then read back (no bumps at all)
local dict = {a = 0, b = 0}
for i = 1, 4000 do
  dict.a = dict.a + 1
  dict.b = dict.b + 2
end
print(dict.a, dict.b)
