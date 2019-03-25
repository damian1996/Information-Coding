import heapq as h, numpy as np
from collections import defaultdict
import operator

def create_heap_from_text(text):
    counter = defaultdict(int)
    for c in inp:
        counter[c] += 1
    maxi = max(counter.items(), key=operator.itemgetter(1))[0]
    pair_to_save = (maxi, counter[maxi])
    del counter[maxi]
    heap = []
    for c, v in counter.items():
        h.heappush(heap, (v, c, -1, -1))
    return heap, pair_to_save

def create_tree(heap, maxi_pair):
    tree = []
    while True:
        if not heap:
            break
        p1 = h.heappop(heap)
        tree.append(p1)
        if not heap:
            break
        p2 = h.heappop(heap)
        tree.append(p2)
        summed = p1[0] + p2[0]
        letters = p1[1] + p2[1]
        left, right = len(tree)-2, len(tree)-1
        h.heappush(heap, (summed, letters, left, right))
    tree.append((maxi_pair[1], maxi_pair[0], -1, -1))
    left, right = len(tree)-1, len(tree)-2
    tree.append((10000, 'jakistekst', left, right))
    return tree

def encode_to_string(text, codes):
    codecs = [codes[c] for c in text]
    return ''.join(codecs)

def is_valid_tree(codes):
    for o1, o2 in codes.items():
        for s1, s2 in codes.items():
            if s2 == o2 and s1 == o1:
                continue
            if len(s2) >= len(o2):
                if s2.startswith(o2):
                    return False
    return True

def encode(text, codes):
    # pomija leading zeros ;_;
    code = 0
    for c in text:
        length = 1 if (codes[c] == 0) else codes[c].bit_length()
        code = code * (1 << length) + codes[c]
    return code, bin(code)[2:]

def tree_walk(node, tree, codes, code):
    if node[2] == -1 and node[3] == -1:
        codes[node[1]] = code
    if node[2] != -1:
        tree_walk(tree[node[2]], tree, codes, code+'0')
    if node[3] != -1:
        tree_walk(tree[node[3]], tree, codes, code+'1')

def create_table_of_symbols(sorted_codes, table_symbol):
    vals = [i[1] for i in sorted_codes]
    int_vals = [int(cd, 2) for cd in vals]
    print(len(set(vals)) == len(set(int_vals)))
    
    for t in zip(sorted_codes):
        letter, code = t[0]
        int_code = int(code, 2)
        diff = D - len(code)
        if diff == 0:
            table_symbol[int_code] = letter
        else:
            nr_to_fill = (1 << diff)
            start = int_code*nr_to_fill
            end = start + nr_to_fill
            table_symbol[start:end] = letter * (end-start)

def bits_extracted(number, k, p):
    if p<0:
        nr = number << abs(p)
    else:
        nr = (number >> p) if p>0 else number
    return (((1 << k) - 1) & nr)

def fast_decoding(code, codes, D):
    table_symbol = [''] * (1 << D)
    sorted_codes = sorted(codes.items(), key=lambda x: x[1] ,reverse=True)
    create_table_of_symbols(sorted_codes, table_symbol)
    bit_len = len(bin(code)) - 2
    shift, msk = bit_len - D, ((1 << D) - 1)
    buffer = bits_extracted(code, D, shift)
    restore = []
    while shift > -D:
        s = table_symbol[buffer]
        restore.append(s)
        nb_bits = (1 if codes[s] == 0 else codes[s].bit_length())
        shift -= nb_bits
        buffer = bits_extracted(code, D, shift)
    return ''.join(restore)

def fast_decoding_from_bitstring(code, codes, D):
    table_symbol = [''] * (1 << D)
    sorted_codes = sorted(codes.items(), key=lambda x: len(x[1]), reverse=True)
    create_table_of_symbols(sorted_codes, table_symbol)
    first_to_read, last_decoded = D, -1
    bit_len, msk = len(code), ((1 << D) - 1)
    code = code + ''.join(['0'] * D)
    buffer, restore = code[:D], []
    while last_decoded < bit_len - 1:
        idx = int(buffer, 2)
        s = table_symbol[idx]
        restore.append(s)
        nb_bits = len(codes[s])
        #x = (x << nb_bits) & msk + read_bits(nb_bits)
        buffer = buffer[nb_bits:D] + code[first_to_read:first_to_read + nb_bits]
        first_to_read += nb_bits
        last_decoded += nb_bits
    return ''.join(restore)

#inp = 'aababdecaeaaeadadwwacde'
#inp = 'aabadbabdababbasbabdabdbabdbadb'
#inp = 'ababbc'
strings = ['Doubtful it stood;As two spent swimmers, that do cling together',
    'And choke their art. The merciless Macdonwald-- Worthy to be a rebel, for to that',
    'The multiplying villanies of nature Do swarm upon him--from the western isles Of kerns and gallowglasses is supplied;',
    'And fortune, on his damned quarrel smiling, Show like a rebel whore: but all too weak:For brave Macbeth--well he deserves that name--',
    'Disdaining fortune, with his brandishd steel,Which smoked with bloody execution,Like valour minion carved out his passage',
    'Till he faced the slave;Which ne shook hands, nor bade farewell to him,Till he unseamd him from the nave to the chaps,And fixd his head upon our battlements.'  
]
inp = ''.join(strings)

heap, maxi_pair = create_heap_from_text(inp)
tree = create_tree(heap, maxi_pair)
codes = {}
tree_walk(tree[len(tree) - 1], tree, codes, '')
print(is_valid_tree(codes))
code = encode_to_string(inp, codes)
print(len(inp)*8 > len(code))

arg = np.argmax(np.array([len(n) for n in codes.values()]))
D = len(list(codes.values())[arg])
print(D)
#print([(tup[0], bin(tup[1])) for tup in codes.items()])
text = fast_decoding_from_bitstring(code, codes, D)
print(text == inp)