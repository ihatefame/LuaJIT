-- Side traces: hot exits grow their own traces and chain back, so branchy and
-- nested code stays compiled. Every output must match LuaJIT byte for byte.

-- 1. a branch inside a hot loop: the un-recorded arm becomes a side trace
local a, b = 0, 0
for i = 1, 4000 do
  if i % 3 == 0 then a = a + i else b = b + 1 end
end
print(a, b)

-- 2. both arms roughly equally hot
local x = 0
for i = 1, 4000 do
  if i % 2 == 0 then x = x + 2 else x = x - 1 end
end
print(x)

-- 3. nested numeric loops: outer stem links into the inner root trace,
--    the inner exit's side trace records the outer tail and links back
local s = 0
for i = 1, 300 do
  for j = 1, 30 do s = s + j end
  s = s + i
end
print(s)

-- 4. table built in the outer loop, filled by the inner one
local m = {}
for i = 1, 200 do
  m[i] = 100
  for j = 1, 3 do m[i] = m[i] + j end
  if m[i] ~= 106 then print("BAD", i, m[i]) end
end
print(m[1], m[200])

-- 5. while inside while
local u, v = 0, 0
local i = 0
while i < 200 do
  local j = 0
  while j < 40 do j = j + 1; v = v + 1 end
  i = i + 1
  u = u + i
end
print(u, v)

-- 6. numeric equality branches on doubles, including a -0.0 == 0.0 case the
--    raw-bits compare would get wrong
local eq, ne = 0, 0
local z = -0.0
for i = 1, 2000 do
  if z == 0.0 then eq = eq + 1 else ne = ne + 1 end
  if i == 1000 then eq = eq + 100 end
end
print(eq, ne)

-- 7. inner loop with a break, inside an outer loop
local found = 0
for pass = 1, 300 do
  for k = 1, 50 do
    if k * k > 900 then found = found + k; break end
  end
end
print(found)
