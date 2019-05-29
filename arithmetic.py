import numpy as np

probs = {'A': [0,50000], 'B': [50000,70000], 'C': [70000,100000]}
low, rang = 0, 100000

# start/total, (start+size)/total
def encode(start, size, total):
    global rang
    global low
    rang /= total
    low = low + start * size 
    rang *= size

    while (low / 10000) == ((low + rang) / 10000):
        emit_digit()

    if rang < 1000:
        emit_digit()
        emit_digit()
        rang = 100000 - low

def emit_digit():
    global rang
    global low
    print(low / 10000)
    low = (low % 10000) * 10
    rang *= 10

def append_digit():
    global rang
    global low
    code = (code % 10000) * 10
    low = (low % 10000) * 10
    rang *= 10

def decode(start, size, total):
    global rang
    global low
    rang /= total
    low += start*rang
    rang *= size

    while (low / 10000) == ((low + rang) / 10000):
        append_digit()

    if rang < 1000:
        append_digit()
        append_digit()
        rang = 100000 - low

encode(0, 6, 10)
print(low, rang)
encode(0, 6, 10) #A
print(low, rang)
encode(6, 2, 10) #B
print(low, rang)
encode(0, 6, 10) #A
print(low, rang)
encode(8, 2, 10) #<EOM>
print(low, rang)

#emit final digits - see below
while rang < 10000:
    emit_digit()

low += 10000
emit_digit()
