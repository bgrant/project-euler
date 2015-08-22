import numpy as np


def problem1(unsigned int ulimit):
    cdef unsigned int i, total
    total = 0
    for i in range(3, ulimit):
        if i % 3 == 0 or i % 5 == 0:
            total += i
    return total

def problem2(unsigned int ulimit):
    cdef unsigned int i, j, total
    total, i, j = 0, 1, 2
    while j < ulimit:
        i, j = j, i + j
        if i % 2 == 0:
            total += i
    return total

def problem4():
    cdef:
        unsigned int i, j
        unsigned int prod = 0
        unsigned int maxprod = 0
    for i in range(999, 99, -1):
        for j in range(999, 99, -1):
            prod = i * j
            if ((prod == int(str(prod)[::-1])) and
                    (prod > maxprod)):
                maxprod = prod
    return maxprod

def problem5():
    cdef:
        unsigned int[::1] divisors = np.array((20, 19, 18, 17, 16,
                                               15, 14, 13, 12, 11),
                                               dtype=np.uint32)
        unsigned int inc = divisors[0]
        unsigned int ndivs = len(divisors)
        unsigned int i = 0
        unsigned int j = 0
        bint found = False
    while not found:
        i += inc
        for j in range(ndivs):
            if i % divisors[j] != 0:
                break
            if j == (ndivs - 1):
                found = True
    return i

def problem6(unsigned int n):
    cdef:
        unsigned int sum_of_squares = 0
        unsigned int sum_of_nums = 0
        unsigned int i
    for i in range(1, n+1):
        sum_of_nums += i
        sum_of_squares += i**2
    return sum_of_nums**2 - sum_of_squares

def problem8():
    cdef:
        char* data = p8_data
        unsigned int start = 13
        unsigned int winsize = start
        unsigned int end = len(p8_data)
        unsigned long product = 1
        unsigned long subproduct = 1
        unsigned int i, j,
    for i in range(start, end):
        subproduct = 1
        for j in range(winsize):
            subproduct *= <int>data[i-j] - 48
        product = subproduct if subproduct > product else product
    return product


p8_data_py = """
73167176531330624919225119674426574742355349194934
96983520312774506326239578318016984801869478851843
85861560789112949495459501737958331952853208805511
12540698747158523863050715693290963295227443043557
66896648950445244523161731856403098711121722383113
62229893423380308135336276614282806444486645238749
30358907296290491560440772390713810515859307960866
70172427121883998797908792274921901699720888093776
65727333001053367881220235421809751254540594752243
52584907711670556013604839586446706324415722155397
53697817977846174064955149290862569321978468622482
83972241375657056057490261407972968652414535100474
82166370484403199890008895243450658541227588666881
16427171479924442928230863465674813919123162824586
17866458359124566529476545682848912883142607690042
24219022671055626321111109370544217506941658960408
07198403850962455444362981230987879927244284909188
84580156166097919133875499200524063689912560717606
05886116467109405077541002256983155200055935729725
71636269561882670428252483600823257530420752963450
"""

p8_data = p8_data_py.strip().replace("\n", "")
