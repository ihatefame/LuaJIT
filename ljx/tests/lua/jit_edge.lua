-- zero-iteration loops (both directions), aliasing dst==rhs, self-ops
local a = 0 for i = 5, 1 do a = a + 1 end print(a)          -- 0
local b = 0 for i = 1, 5, -1 do b = b + 1 end print(b)      -- 0
local c = 10.0 for i = 1, 3 do c = i - c end print(c)       -- non-commutative alias
local d = 2.0 for i = 1, 10 do d = d * d end print(d == d)  -- self-multiply -> inf
local e = 0.0 for i = 1, 4 do e = -i + e end print(e)       -- unary minus
local f = 1.0 for i = 1, 3 do f = 100 / f end print(f)      -- const-left div
local g = 0 for i = 1, 3 do local t = i i = i end print(g)   -- write to loop var
-- guard fallback: string in a slot the loop reads
local h = "notanumber"
local ok = pcall(function() for i = 1, 3 do h = h + 1 end end)
print(ok)                                                    -- false, no crash
