local function fib(n) if n < 2 then return n end return fib(n-1) + fib(n-2) end
local N = 34
local t0 = os.clock()
local r = fib(N)
print(string.rep("",0) .. "fib("..N..")="..r.." in "..(os.clock()-t0).."s")
