-- control flow, functions, recursion
local function fib(n)
  if n < 2 then return n end
  return fib(n-1) + fib(n-2)
end
print(fib(20))

local sum = 0
for i = 1, 100 do sum = sum + i end
print(sum)

local i, product = 0, 1
while i < 10 do i = i + 1; product = product * 2 end
print(product)

repeat i = i - 3 until i < 0
print(i)

-- tables + generic for
local t = {10, 20, 30, x = "ex", y = "why"}
print(#t, t[2], t.x, t.y)
local total = 0
for _, v in ipairs(t) do total = total + v end
print(total)
local nkeys = 0
for k in pairs(t) do nkeys = nkeys + 1 end
print(nkeys)

-- closures & upvalues
local function counter()
  local n = 0
  return function() n = n + 1 return n end
end
local c1, c2 = counter(), counter()
c1() c1() c1()
c2()
print(c1(), c2())

-- strings
local s = "hello world"
print(s:len(), s:sub(1, 5), string.rep("ab", 3))
print(("x"):byte())

-- and/or value semantics
print(nil or "default", false and "no" or "yes", 1 and 2)
print(not nil, not 0)

-- metatables
local base = {greet = function(self) return "hi " .. self.name end}
local obj = setmetatable({name = "ljx"}, {__index = base})
print(obj:greet())
local vec = setmetatable({x=1}, {__add = function(a, b) return a.x + b.x end})
print(vec + vec)

-- pcall / error
local ok, err = pcall(function() error("boom") end)
print(ok, err)
print(pcall(function() return 42 end))

-- multiple returns
local function three() return 1, 2, 3 end
local a, b, c = three()
print(a, b, c)
