-- Realistic mixed workload: objects with methods, string keys, nested tables.
local Vec = {}
Vec.__index = Vec
function Vec.new(x, y) return setmetatable({x = x, y = y}, Vec) end
function Vec:dot(o) return self.x * o.x + self.y * o.y end
function Vec:scale(k) self.x = self.x * k; self.y = self.y * k; return self end

local t0 = os.clock()
local acc = 0.0
local a = Vec.new(1, 2)
local b = Vec.new(3, 4)
for i = 1, 400000 do
  a:scale(1.000001)
  acc = acc + a:dot(b)
end

-- string-keyed dictionary traffic
local counts = {}
local keys = {"alpha","beta","gamma","delta","epsilon"}
for i = 1, 400000 do
  local k = keys[(i % 5) + 1]
  counts[k] = (counts[k] or 0) + 1
end
print("real="..math.floor(acc % 1000)..","..counts.alpha.." in "..(os.clock()-t0).."s")
