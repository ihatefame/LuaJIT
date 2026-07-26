local N = 3000000
local t = {}
for i = 1, N do t[i] = 0 end          -- pre-size (grow path, interpreted)
local t0 = os.clock()
for i = 1, N do t[i] = i * 2 end      -- in-range stores (JIT)
local s = 0.0
for i = 1, N do s = s + t[i] end      -- reads (JIT)
print("arraysum="..s.." in "..(os.clock()-t0).."s")
