import numpy as np
import itertools as it
from copy import deepcopy

perm = [8,1,3,5,0,4,2,6,7]

def encode_to_number(enc_perm, idx, pos):
    if idx == len(enc_perm): return 0
    rec = encode_to_number(enc_perm, idx+1, pos-1)
    return enc_perm[idx] + pos*rec

def encode_perm(perm):
    enc_perm = deepcopy(perm)
    for i, el in enumerate(perm):
        smaller = np.count_nonzero(np.where(np.array(perm[:i]) < el, 1, 0))
        enc_perm[i] = max(0, el - smaller)
    
    return encode_to_number(enc_perm, 0, len(enc_perm))

def decode_perm(enc_perm, N):
    dec_perm = deepcopy(enc_perm)
    for n in range(N):
        i = N-n-1
        for j in range(i+1, N):
            if dec_perm[j] >= dec_perm[i]:
                dec_perm[j] += 1
    return dec_perm

def decode_recursive(x, N, perm, idx):
    if N == 0: return
    perm[idx] = x % N
    x = x // N
    decode_recursive(x, N-1, perm, idx+1)

def decode_number_to_array(encoding, N):
    perm = [0]*N
    decode_recursive(encoding, N, perm, 0)
    return decode_perm(perm, N)

# 8+9(1+8(2+7(3+6(0+5(1+4(0+3(0+2(0+0))))))))=16793

print("Start point ")
print(perm)

enc_perm = encode_perm(perm)
print(enc_perm)

decoded_perm = decode_number_to_array(enc_perm, len(perm))
print(decoded_perm)
