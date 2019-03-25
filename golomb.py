import math as m

def is_power_of_2(num):
	return num != 0 and ((num & (num - 1)) == 0)

def code(num, mod):
    unary = num // mod
    reminder = num % mod
    code = '1' * unary + '0'
    print(code)
    if is_power_of_2(mod):
        bits_needed = int(m.log(mod, 2))
        add_bits = bin(reminder)[2:]
        add_bits = '0' * (bits_needed - len(add_bits)) + add_bits
    else:
        bits_needed = int(m.ceil(m.log(mod, 2)))
        cutoff = (1 << bits_needed) - mod
        if reminder < cutoff:
            add_bits = bin(reminder)[2:]
            add_bits = '0' * (bits_needed - len(add_bits) - 1) + add_bits
        else:
            add_bits = bin(reminder + cutoff)[2:]
            add_bits = '0' * (bits_needed - len(add_bits)) + add_bits
    code = code + add_bits
    return code

def decode(code, mod):
    idx = code.find('0')
    num = mod * idx
    rem_part = code[idx+1:]
    if is_power_of_2(mod):
        num = num + int(rem_part, 2)
    else:
        bits_needed = int(m.ceil(m.log(mod, 2)))
        cutoff = (1 << bits_needed) - mod
        if len(rem_part) == bits_needed:
            num = num + (int(rem_part, 2) - cutoff)
        else:
            num = num + int(rem_part, 2)
    return num

N, N2, M1, M2 = 42, 47, 10, 16
c1 = code(N, M2)
c2 = code(N2, M2)
c3 = code(N, M1)
c4 = code(N2, M1)

dc1 = decode(c1, M2)
dc2 = decode(c2, M2)
dc3 = decode(c3, M1)
dc4 = decode(c4, M1)

print(N, c1, dc1)
print(N2, c2, dc2)
print(N, c3, dc3)
print(N2, c3, dc3)