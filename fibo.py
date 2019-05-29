N = 1100010101

def compute_fibo_less_or_equal(N):
    fibos, last, idx = [0, 1], 1, 2
    while last <= N:
        last = (fibos[idx-1] + fibos[idx-2])
        idx += 1
        fibos.append(last)
    return fibos[:idx-1], idx-2

smaller_fibo, less_or_equal_idx = compute_fibo_less_or_equal(N)
print(smaller_fibo)
code, idx = [0] * len(smaller_fibo), len(smaller_fibo)
temp = N - smaller_fibo[less_or_equal_idx]
code[less_or_equal_idx] = 1
less_or_equal_idx -= 1

while temp>0 and less_or_equal_idx >= 0:
    if smaller_fibo[less_or_equal_idx] <= temp:
        temp -= smaller_fibo[less_or_equal_idx]
        code[less_or_equal_idx] = 1
    less_or_equal_idx -= 1    

print(code)

decoded = 0
for i, v in enumerate(code):
    if v:
        decoded += smaller_fibo[i]

print(decoded)