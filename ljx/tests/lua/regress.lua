-- deterministic golden-output regression (no clock/random)
local function fib(n) if n < 2 then return n end return fib(n-1) + fib(n-2) end
print(fib(20))                                    -- 6765
local s = 0; for i=1,100 do s = s + i end; print(s)   -- 5050
local p = 1; local i = 0
while i < 10 do i = i + 1; p = p * 2 end; print(p)     -- 1024
i = 10; repeat i = i - 3 until i < 0; print(i)         -- -2
local t = {1,2,3,4,5}; local a = 0
for _, v in ipairs(t) do a = a + v end; print(a)       -- 15
print(pcall(function() error("x") end) == false)       -- true
local ok = pcall(function()
  local o = setmetatable({}, {__index = function() return "nested-ok" end})
  print(o.anything)                                     -- nested-ok
end); assert(ok)
local function multi() return 42 end; print(multi())    -- 42
local mt = setmetatable({v=1}, {__add=function(a,b) return a.v+b.v+1 end})
print(mt + mt)                                          -- 3
print(math.floor(math.sqrt(14400)))                     -- 120
