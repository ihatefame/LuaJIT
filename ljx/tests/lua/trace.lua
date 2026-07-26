-- Exercises the trace compiler: loops long enough to become hot, then
-- deliberate guard failures so every side exit has to restore state exactly.

-- 1. object methods resolved through __index, with a per-instance override
--    appearing MID-LOOP: the trace must not keep calling the class method.
local Vec = {}
Vec.__index = Vec
function Vec.new(x, y) return setmetatable({x = x, y = y}, Vec) end
function Vec:dot(o) return self.x * o.x + self.y * o.y end
function Vec:name() return "class" end

local a, b = Vec.new(1, 2), Vec.new(3, 4)
local acc = 0
for i = 1, 200 do acc = acc + a:dot(b) end
print("dot", acc)

local names = {}
for i = 1, 120 do
  if i == 100 then a.name = function() return "instance" end end
  names[#names + 1] = a:name()
end
print("override", names[1], names[99], names[100], names[120])

-- 2. two-level inheritance: the trace (and the interpreter's inline cache)
--    must walk the whole __index chain, not stop at the first table.
local Base = {}; Base.__index = Base
function Base:greet() return "base" end
local Derived = {}; Derived.__index = Derived
setmetatable(Derived, {__index = Base})
local obj = setmetatable({}, Derived)
local g = ""
for i = 1, 120 do g = obj:greet() end
print("inherit", g)

-- 3. a guard that fails mid-iteration: the slot type changes on iteration 150,
--    so the exit must resume with every earlier assignment already visible.
local mixed = 0
local v = 1
for i = 1, 300 do
  if type(v) == "number" then mixed = mixed + v * 2 else mixed = mixed + 1 end
  if i == 150 then v = "stop" end
end
print("deopt", mixed)

-- 4. array traffic and field stores inside the traced loop
local arr = {}
for i = 1, 40 do arr[i] = i end
local sum = 0
for pass = 1, 120 do
  for i = 1, 40 do sum = sum + arr[i] end
end
print("array", sum)

local p = Vec.new(1, 1)
for i = 1, 300 do p.x = p.x + 1; p.y = p.y + 2 end
print("fields", p.x, p.y)

-- 5. descending loops, modulo, unary minus, nested calls
local function twice(n) return n + n end
local function chain(n) return twice(n) + 1 end
local d = 0
for i = 300, 1, -1 do d = d + chain(i % 7) - (-1) end
print("misc", d)

-- 6. a loop whose body leaves through a `break` (an exit taken from the middle)
local found = 0
for i = 1, 500 do
  if i * i > 40000 then found = i break end
end
print("break", found)

-- 7. dictionary traffic with a VARIABLE string key: the main position has to
--    be computed at run time from the key's id, not folded to a constant.
local counts = {}
local keys = {"alpha", "beta", "gamma", "delta", "epsilon"}
for i = 1, 400 do
  local k = keys[(i % 5) + 1]
  counts[k] = (counts[k] or 0) + 1
end
print("dict", counts.alpha, counts.epsilon)

-- 8. more simultaneously live values than there are registers, so the
--    allocator has to spill and reload inside the loop.
local a1, a2, a3, a4, a5, a6 = 1, 2, 3, 4, 5, 6
local b1, b2, b3, b4, b5, b6 = 7, 8, 9, 10, 11, 12
local c1, c2, c3, c4 = 13, 14, 15, 16
for i = 1, 300 do
  a1 = a1 + b1 * 0.5; a2 = a2 + b2 * 0.5; a3 = a3 + b3 * 0.5
  a4 = a4 + b4 * 0.5; a5 = a5 + b5 * 0.5; a6 = a6 + b6 * 0.5
  b1 = b1 + c1 * 0.25; b2 = b2 + c2 * 0.25
  b3 = b3 + c3 * 0.25; b4 = b4 + c4 * 0.25
  c1 = c1 + 1; c2 = c2 + 2; c3 = c3 + 3; c4 = c4 + 4
end
print("pressure", a1 + a2 + a3 + a4 + a5 + a6, b1 + b2 + b3 + b4, c1 + c2 + c3 + c4)

-- 9. nested field chains and a receiver that CHANGES every iteration, which
--    the hoisting pass must not treat as loop invariant.
local objs = {Vec.new(1, 1), Vec.new(2, 2), Vec.new(3, 3), Vec.new(4, 4)}
local tot = 0
for i = 1, 400 do
  local o = objs[(i % 4) + 1]
  tot = tot + o:dot(o)
end
print("poly", tot)
