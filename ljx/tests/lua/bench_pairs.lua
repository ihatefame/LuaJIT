-- pairs-heavy workload: repeated full walks of a mixed-key table
local t = {}
for i = 1, 200 do t[i] = i end
for i = 1, 200 do t["key" .. i] = i * 2 end
local start = os.clock()
local s = 0
for rep = 1, 3000 do
  for k, v in pairs(t) do s = s + v end
end
print(s)
print(os.clock() - start)
