local t0 = os.clock()
local t = {}
for i = 1, 3000000 do t[i] = i * 2 end
local s = 0
for i = 1, 3000000 do s = s + t[i] end
print("tabsum="..s.." in "..(os.clock()-t0).."s")
