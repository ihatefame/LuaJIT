local t0 = os.clock()
local sum = 0.0
for i = 1, 50000000 do sum = sum + (i * 2 - 1) end
print("loopsum="..sum.." in "..(os.clock()-t0).."s")
