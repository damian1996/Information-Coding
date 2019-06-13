import numpy as np
from math import sqrt
import itertools as it
from copy import deepcopy
import os

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
    norm = np.sum(np.abs(vec_in_S2))
    projected = vec_in_S2 / norm
    # projectd = projected ** (1/p)
    return projected, norm
  
def multivariate_proj_into_S2(vec_in_S1, p=1.0):
    norm = np.sqrt(np.sum(vec_in_S1 ** 2))
    projected = vec_in_S1 / norm
    # projected = projected ** p
    return projected, norm

def multivariate_gauss_proj_into_S2(gauss_vec, p=1.0):
    norm = np.sqrt(np.sum(gauss_vec ** 2))
    projected = gauss_vec / norm
    # projected = projected ** p
    return projected, norm

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
    vec_in_S2, norm_L2 = multivariate_gauss_proj_into_S2(vec)
    vec_in_S1, norm_L1 = multivariate_proj_into_S1(vec_in_S2)
    closest_vec = smallest_distance_in_unit_sphere(vec_in_S1)
    code = code_to_number(L, K, 0, all_vectors[closest_vec], 0)
    dec = decode_to_vector(0, L, K, code)
    vec_back_in_S2, _ = multivariate_gauss_proj_into_S2(dec)
    normal_points = vec_back_in_S2 * norm_L2

if __name__ == '__main__':
    L, K = 8, 10
    all_possibilites = N(L, K)
    print(all_possibilites)
    find_all_vectors(L, K)
    assert(len(all_vectors) == all_possibilites)
    # results, uniques = [], []
    # for ii, v in enumerate(all_vectors):
    #     id_for_vector = code_to_number(L, K, 0, v, 0)
    #     uniques.append(id_for_vector)
    #     results.append((id_for_vector, v))
    #     if ii%1000==0: print(ii)

    # assert(all_possibilites == np.unique(np.array(uniques)).size)

    os.chdir('.')
    # serialize_indices = 'indices.npy'
    outfile = 'vectors.npy'

    stacked_vectors = np.stack(all_vectors)
    # np.save(serialize_indices, np.array(uniques))
    np.save(outfile, stacked_vectors)

    # loaded_uniques = np.load(serialize_indices)
    loaded_stacked = np.load(outfile)
    
    # assert(np.array_equal(np.array(loaded_uniques), np.array(uniques)))
    assert(np.array_equal(stacked_vectors, loaded_stacked))
    assert(loaded_stacked.shape == stacked_vectors.shape)
