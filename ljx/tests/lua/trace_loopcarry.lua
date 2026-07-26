local Q={} Q.__index=Q
function Q.new() return setmetatable({n=0,head=1},Q) end
function Q:push(v) self.n=self.n+1 self[self.n]=v end
function Q:size() return self.n-self.head+1 end
local q=Q.new() for i=1,500 do q:push(i) end
local s=0 for i=1,500 do s=s+q:size() end print(s)

-- A snapshot taken before a slot's first read must still name that slot when
-- the slot is loop-carried: the register the back edge keeps current is the
-- only live copy, and the Lua stack holds the value from trace entry.
local q2 = {n = 0}
for i = 1, 500 do q2.n = q2.n + 1; q2[q2.n] = i end
print(q2.n, q2[500])

local q3 = {n = 0}
for i = 1, 500 do local k = q3.n + 1; q3.n = k; q3[k] = i end
print(q3.n, q3[500])

-- The same hazard through an inlined method call.
local L = {} L.__index = L
local obj = setmetatable({n = 0}, L)
function L:bump() self.n = self.n + 1 return self.n end
local last = 0
for i = 1, 500 do last = obj:bump() end
print(obj.n, last)
