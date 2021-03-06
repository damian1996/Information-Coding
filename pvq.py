import numpy as np
from math import sqrt
import itertools as it
from copy import deepcopy
import os

# Zmemoryzowac funkcje N
# Jak dokladnie enkodowac kule (y_i / K)

all_vectors = []

def get_next_vector(combi, vec, missing_value):
    if len(combi) == 1:
        nonneg, neg = deepcopy(vec), deepcopy(vec)
        nonneg[combi[0]] = missing_value
        neg[combi[0]] = -missing_value
        all_vectors.append(nonneg)
        all_vectors.append(neg)
    else:
        possible_values = missing_value - (len(combi) - 1)
        for i in range(1, possible_values+1):
            nonneg, neg = deepcopy(vec), deepcopy(vec)
            nonneg[combi[0]] = i
            neg[combi[0]] = -i
            get_next_vector(combi[1:], nonneg, missing_value - i)
            get_next_vector(combi[1:], neg, missing_value - i)

def find_all_vectors(L, K):
    # jednoelementowe
    for i in range(L):
        s1, s2 = np.zeros(L), np.zeros(L)        
        s1[i], s2[i] = K, -K
        all_vectors.append(s1)
        all_vectors.append(s2)
    # dwu i wiecej elementowe
    for i in range(2, L+1): # ile wektorow jest niezerowych
        all_combi = list(it.combinations((np.arange(L)), i))
        for combi in all_combi:
            #vec, neg_vec = np.zeros(L), np.zeros(L)
            possible_values = K - (i-1)
            for val in range(1, possible_values+1):
                vec, neg_vec = np.zeros(L), np.zeros(L)
                vec[combi[0]], neg_vec[combi[0]] = val, -val
                get_next_vector(combi[1:], vec, K-val)
                get_next_vector(combi[1:], neg_vec, K-val)

def smallest_distance_in_unit_sphere(projected_vec):
    min_dists = []
    for i, vec in enumerate(all_vectors):
        min_dists.append(np.linalg.norm(projected_vec - vec))
    return np.argmin(np.array(min_dists))

def multivariate_proj_into_S1(vec_in_S2, p=1.0): # shape -> batch_size x L
    #norm = np.zeros(vec_in_S2.shape[1])
    norm = np.sum(np.abs(vec_in_S2)) # po kolumnach, shape -> L
    projected = vec_in_S2 / norm
    # projectd = projected ** (1/p)
    return projected, norm
  
def multivariate_proj_into_S2(vec_in_S1, p=1.0):
    norm = np.sqrt(np.sum(vec_in_S1 ** 2)) # znow po kolumnach
    projected = vec_in_S1 / norm
    # projected = projected ** p
    return projected, norm

def multivariate_gauss_proj_into_S2(gauss_vec, p=1.0):
    norm = np.sqrt(np.sum(gauss_vec ** 2)) # znow po kolumnach
    projected = gauss_vec / norm
    # projected = projected ** p
    return projected, norm

def projection_into_S1(vec_in_S2): # P_1(x) = x / ||x||_1
    x, y = vec_in_S2[0], vec_in_S2[1]
    norm_1 = abs(x) + abs(y)
    return [x/norm_1, y/norm_1]

def projection_into_S2(vec_in_S1):
    x, y = vec_in_S1[0], vec_in_S1[1]
    norm_1 = sqrt(x*x + y*y)
    return [x/norm_1, y/norm_1]

def N(L, K):
    if K == 0: return 1
    if L == 0: return 0
    return N(L, K-1) + N(L-1, K-1) + N(L-1, K)

def sgn(x):
    if x>0: return 1
    elif x==0: return 0
    else: return -1

def code_to_number(L, K, i, vec, idx):
    while K > 0: 
        if vec[i] == 0:
            idx = idx + 0
        elif abs(vec[i]) == 1:
            idx = idx + N(L-1, K)
            idx = idx + ((1-sgn(vec[i]))/2)*N(L-1, K-abs(int(vec[i])))
        elif abs(vec[i]) > 1:
            idx = idx + N(L-1, K)
            idx = idx + 2*sum(N(L-1, K-j) for j in range(1, abs(int(vec[i]))))
            idx = idx + ((1-sgn(vec[i]))/2)*N(L-1, K-abs(int(vec[i])))
        K = K - abs(int(vec[i]))
        L = L - 1
        i = i + 1
    return int(idx)

def step1(xb, i, l, k, code, vec_to_reach, j, L):
    if xb == code:
        vec_to_reach[i] = 0
        step5(xb, i, l, k, code, vec_to_reach, j, L)
    return xb

def step2(xb, i, l, k, code, vec_to_reach, j, L):
    if code < (xb + N(l-1, k)):
        vec_to_reach[i] = 0
        k, l, i = step4(xb, i, l, k, code, vec_to_reach, j, L)
    else:
        xb, j = (xb + N(l-1, k)), 1
        xb, j = step3(xb, i, l, k, code, vec_to_reach, j, L)
        k, l, i = step4(xb, i, l, k, code, vec_to_reach, j, L)
    return k, l, i, xb, j

def step3(xb, i, l, k, code, vec_to_reach, j, L):
    while code >= (xb + 2*N(l- 1, k -j)):
        xb, j = xb + 2*N(l-1, k-j), j+1
    vec_to_reach[i] = (-j if code >= xb+N(l-1,k-j) else j)
    if code >= xb+N(l-1,k-j): # czy to powinno tutaj byc... mozliwe, ze nie, ale bez tegp nie dziala
        xb += N(l-1,k-j)
    return xb, j

def step4(xb, i, l, k, code, vec_to_reach, j, L):
    k = k - abs(int(vec_to_reach[i]))
    l, i = l-1, i+1
    return k, l, i

def step5(xb, i, l, k, code, vec_to_reach, j, L):
    if k > 0:
        vec_to_reach[L-1] = k - abs(int(vec_to_reach[i]))

def decode_to_vector(i, l, k, code):
    j, xb = 0, 0
    vec = np.zeros(l)
    cnt = 0
    L = deepcopy(l)
    while k > 0:
        cnt += 1
        xb = step1(xb, i, l, k, code, vec, j, L)
        if xb == code:
            break
        k, l, i, xb, j = step2(xb, i, l, k, code, vec, j, L)
    return vec

def all_operations(vec, all_possibilites, L, K):
    print(vec)
    vec_in_S2, norm_L2 = multivariate_gauss_proj_into_S2(vec)
    vec_in_S1, norm_L1 = multivariate_proj_into_S1(vec_in_S2)
    
    closest_vec = smallest_distance_in_unit_sphere(vec_in_S1)
    #print("Closest = ", all_vectors[closest_vec])
    code = code_to_number(L, K, 0, all_vectors[closest_vec], 0)
    #print("Code = ", code)
    dec = decode_to_vector(0, L, K, code)
    #print("Decoded = ", dec)
    vec_back_in_S2, _ = multivariate_gauss_proj_into_S2(dec)
    normal_points = vec_back_in_S2 * norm_L2
    print(normal_points)

if __name__ == '__main__':
    L, K = 6, 6
    vectors_in_S2 = [[-1, 0], [-1/2, sqrt(3)/2]]
    all_possibilites = N(L, K)
    print(all_possibilites)
    find_all_vectors(L, K)
    assert(len(all_vectors) == all_possibilites)

    results, uniques = [], []
    for ii, v in enumerate(all_vectors):
        id_for_vector = code_to_number(L, K, 0, v, 0)
        uniques.append(id_for_vector)
        results.append((id_for_vector, v))
    assert(all_possibilites == np.unique(np.array(uniques)).size)

    os.chdir('.')
    outfile = 'vectors.npy'
    np.save(outfile, np.array(uniques))
    
    loaded_uniques = np.load(outfile)
    assert(np.array_equal(np.array(loaded_uniques), np.array(uniques)))

    res = sorted(results, key=lambda x:x[0])
    # for v in res: print(f'{v[0]} -> {v[1]}')

    for v in res: 
        decoded_vector = decode_to_vector(0, L, K, v[0])
        assert(np.array_equal(np.array(v[1]), np.array(decoded_vector)))
        #print(f'{v[1]} -> {v[0]} -> {decoded_vector}')

    mu, sigma = 0, 0.5
    s1, s2 = np.random.normal(mu, sigma, L), np.random.normal(mu, sigma, L)
    all_operations(s1, all_possibilites, L, K)
    print()
    all_operations(s2, all_possibilites, L, K)

# - osobno kodujemy |z|, można bezpośrednio zapisać, lepiej użyć że |z|^2 
# jest z rozkładu chi-squared,

# - osobno z/|z| ze sfery używając PVQ: rzutujemy na S1, kwantyzacja, 
# enumarative coding. Decoder rzutuje na S2.
# D(Q(E(x)))