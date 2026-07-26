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
