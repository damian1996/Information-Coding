import numpy as np
import itertools as it

bits = [[0,0,0,1,1], [0,0,1,0,1], [0,0,1,1,0], [0,1,0,0,1], [0,1,0,1,0], 
    [0,1,1,0,0], [1,0,0,0,1], [1,0,0,1,0], [1,0,1,0,0], [1,1,0,0,0]] 

ones = np.count_nonzero(np.array(bits[0]))
N = len(bits[0])
C = np.zeros((N+1, ones+1))

def compute_comb(n, k):
    if k == 0 or k == n: return 1
    if C[n,k] != 0:
        return C[n,k]
    C[n-1, k-1] = compute_comb(n-1, k-1)
    C[n-1, k] = compute_comb(n-1, k)
    return C[n-1, k-1] + C[n-1, k]

def code(bits, n, k, idx):
    if n == 0: return 0
    if bits[idx] == 0:
        return code(bits, n-1, k, idx+1)
    return C[n-1, k] + code(bits, n-1, k-1, idx+1)

def decode(i, n, k):
    combs = it.combinations(range(n), k)
    for c in combs:
        res = 0
        for j, el in enumerate(c):
            res += int(C[n-el-1, k-j])
        if res == i:
            decoded_bits = np.zeros(n)
            for idx in c:
                decoded_bits[idx] = 1
            return decoded_bits
    return None

C[N, ones] = compute_comb(N, ones)

encodes = []
for i in range(N*ones):
    encoded = int(code(bits[i], len(bits[i]), ones, 0))
    print(bits[i], '->', encoded)
    encodes.append(encoded)

for i in range(N*ones):
    print(encodes[i], '->', decode(i, N, ones))