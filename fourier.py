import numpy as np
import matplotlib.pyplot as plt
bits = "10001001110010101000100111001010"

# omega = 2pi*f

mapping_table = {
    (0,0,0,0) : -3-3j,
    (0,0,0,1) : -3-1j,
    (0,0,1,0) : -3+3j,
    (0,0,1,1) : -3+1j,
    (0,1,0,0) : -1-3j,
    (0,1,0,1) : -1-1j,
    (0,1,1,0) : -1+3j,
    (0,1,1,1) : -1+1j,
    (1,0,0,0) :  3-3j,
    (1,0,0,1) :  3-1j,
    (1,0,1,0) :  3+3j,
    (1,0,1,1) :  3+1j,
    (1,1,0,0) :  1-3j,
    (1,1,0,1) :  1-1j,
    (1,1,1,0) :  1+3j,
    (1,1,1,1) :  1+1j
}

decode_mapping_table = {
    -3-3j: (0,0,0,0),
    -3-1j: (0,0,0,1),
    -3+3j: (0,0,1,0),
    -3+1j: (0,0,1,1),
    -1-3j: (0,1,0,0),
    -1-1j: (0,1,0,1),
    -1+3j: (0,1,1,0),
    -1+1j: (0,1,1,1),
    3-3j: (1,0,0,0),
    3-1j: (1,0,0,1),
    3+3j: (1,0,1,0),
    3+1j: (1,0,1,1),
    1-3j: (1,1,0,0),
    1-1j: (1,1,0,1),
    1+3j: (1,1,1,0),
    1+1j: (1,1,1,1)
}

def code(bits):
    grouped_bits = []
    for i in range(0,int(len(bits)/4)):
        t = tuple(list(map(lambda x: int(x), bits[i*4:(i+1)*4])))
        grouped_bits.append(mapping_table[t])
    return np.fft.ifft(grouped_bits)

def decode(ifft_res):
    fft_res = np.fft.fft(ifft_res)
    bits = []
    for i, r in enumerate(fft_res):
        t = decode_mapping_table[r]
        for bit in t:
            bits.append(bit)
    return ''.join([str(bit) for bit in bits])

print(bits)
res = code(bits)
print(decode(res))