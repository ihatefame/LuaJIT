-- string.format: numeric conversions, width/precision, strings, %c, %%.
print(string.format("%d %i %5d %-5d|", 42, -7, 42, 42))
print(string.format("%x %X %o %u", 255, 255, 8, 7))
print(string.format("%.2f %e %g", 3.14159, 12345.678, 0.5))
print(string.format("%s %10s %-10s| %c", "hi", "pad", "left", 65))
print(string.format("%5.2f%%", 12.345))
print(("x=%d y=%s"):format(1, true))
print(#string.format("%q", 'a"b\\c'))
