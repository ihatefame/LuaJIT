-- Inline caches must never survive a change to what they resolved through.
local Class = {}
Class.__index = Class
function Class:who() return "A" end
local obj = setmetatable({}, Class)
local out = {}
for i = 1, 50 do out[#out+1] = obj:who() end     -- warms the method cache
print(out[1], out[50])                            -- A  A

-- 1. reassign the method on the class (existing key, raw store)
Class.who = function() return "B" end
print(obj:who())                                  -- B, not A

-- 2. add the field directly on the receiver: it must win over the class
obj.who = function() return "C" end
print(obj:who())                                  -- C
obj.who = nil
print(obj:who())                                  -- B again

-- 3. swap __index to a different table
local Other = { who = function() return "D" end }
Class.__index = Other
print(obj:who())                                  -- D

-- 4. drop the metatable entirely
setmetatable(obj, nil)
print(obj.who)                                    -- nil

-- 5. plain field caching must not confuse two instances
local P = {}; P.__index = P; P.tag = "base"
local p1 = setmetatable({}, P)
local p2 = setmetatable({tag = "own"}, P)
local s = ""
for i = 1, 30 do s = p1.tag .. "/" .. p2.tag end
print(s)                                          -- base/own
