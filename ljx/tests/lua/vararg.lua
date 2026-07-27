-- Vararg functions: FuncV frame reshaping, VarG, select, unpack. The frame
-- moves up past the supplied arguments; returns and tailcalls hop the
-- Vararg link back down.

-- 1. basics: ignore, forward, count
local function f(...) return 7 end
print(f(), f(1, 2, 3))
local function id(...) return ... end
print(id(1, "two", 3))
local function count(...) return select('#', ...) end
print(count(), count(1), count(nil, nil, nil))

-- 2. mixed fixed + varargs, missing and surplus arguments
local function mixed(a, b, ...) return a, b, select('#', ...) end
print(mixed(1, 2, 3, 4, 5))
print(mixed(9))

-- 3. select indexing, including negative
local function pick(n, ...) return (select(n, ...)) end
print(pick(2, "a", "b", "c"), pick(-1, "x", "y", "z"))

-- 4. the c10 pattern: sum via select in a hot loop
local function sum(...)
  local s = 0
  for i = 1, select('#', ...) do s = s + select(i, ...) end
  return s
end
local a = 0
for i = 1, 500 do a = a + sum(i, i + 1, i + 2) end
print(a)

-- 5. tail-recursive vararg function: the vararg gap must not leak stack
local function loop(n, ...)
  if n == 0 then return select('#', ...) end
  return loop(n - 1, ...)
end
print(loop(50000, "a", "b", "c"))

-- 6. forwarding through calls and pcall
local function fwd(...) return sum(...) end
print(fwd(10, 20, 30))
local function boom(...) error("boom") end
local bOk, sErr = pcall(boom, 1, 2)
print(bOk, sErr:sub(-4))   -- position prefix is path-dependent; check the tail

-- 7. unpack / table.unpack
print(unpack({ 10, 20, 30 }))
print(table.unpack({ 4, 5, 6 }, 2, 3))
local function apply(fn, t) return fn(unpack(t)) end
print(apply(sum, { 1, 2, 3, 4, 5 }))

-- 8. varargs in returned tables and multiple assignment
local function pack2(...) local x, y = ...; return { x, y } end
local p = pack2("p", "q", "r")
print(p[1], p[2])
local function tail(...) return select(2, ...) end
print(tail(1, 2, 3))
