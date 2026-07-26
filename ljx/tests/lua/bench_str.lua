local t0 = os.clock()
local n = 0
for i = 1, 2000000 do
  local s = "key" .. (i % 1000)
  n = n + #s
end
print("strn="..n.." in "..(os.clock()-t0).."s")
