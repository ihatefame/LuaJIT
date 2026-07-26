local t0 = os.clock()
local function apply(arr, f)
  local r = 0
  for i = 1, #arr do r = r + f(arr[i]) end
  return r
end
local arr = {}
for i = 1, 100 do arr[i] = i end
local total = 0
for pass = 1, 30000 do
  total = total + apply(arr, function(x) return x * 2 + 1 end)
end
print("cb="..total.." in "..(os.clock()-t0).."s")
