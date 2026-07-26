local t0 = os.clock()
local a, b, s = 0, 0, 0
for i = 1, 5000000 do
  if i % 3 == 0 then a = a + i
  elseif i % 5 == 0 then b = b + 2
  else s = s + 1 end
end
local r = 0
for i = 1, 2000 do
  for j = 1, 500 do r = r + j end
  r = r + i
end
print("branch="..a.."/"..b.."/"..s.."/"..r.." in "..(os.clock()-t0).."s")
