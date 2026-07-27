-- Lua patterns: find/match/gmatch/gsub with classes, sets, captures,
-- quantifiers, %b balance, %f frontier, backreferences, position captures.
print(string.find("hello world", "wor"))
print(string.find("a.b", ".", 1, true))
print(string.match("key=value", "(%w+)=(%w+)"))
print(string.match("2026-07-27", "(%d+)-(%d+)-(%d+)"))
local acc = {}
for w in string.gmatch("one two three", "%a+") do acc[#acc + 1] = w end
print(#acc, acc[1], acc[3])
for k, v in string.gmatch("a=1, b=2", "(%w+)=(%w+)") do acc[#acc + 1] = k .. v end
print(acc[4], acc[5])
print(string.gsub("hello world", "o", "0"))
print(string.gsub("hello world", "(%w+)", "<%1>"))
print(string.gsub("hello", "l+", function(m) return "[" .. m .. "]" end))
print(string.gsub("ab cd", "%w+", { ab = "AB", cd = "CD" }))
print(string.match("<<x>>", "<(.-)>"))
print(string.find("th{is} b{alan}ced", "%b{}"))
print(string.gsub("the word is word", "%f[%w]%w+", "X"))
print(string.match("aa bb aa", "(%a+) %a+ %1"))
print(string.match("abc123", "()%d+()"))
print(string.find("hello", "o", -2))
print((", spaced , csv ,x"):gsub("%s*,%s*", ","))
print(pcall(string.match, "a", "(a"))
print(select('#', string.match("abc", "x")))
