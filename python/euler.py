"""
`Project Euler <http://www.projecteuler.net>`_ solutions in Python.

Tested in Python 2.7.2.

:author: Robert David Grant <robert.david.grant@gmail.com>

:copyright:
    Copyright 2011 Robert David Grant.

    Licensed under the Apache License, Version 2.0 (the "License"); you
    may not use this file except in compliance with the License.  You
    may obtain a copy of the License at

      http://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
    implied.  See the License for the specific language governing
    permissions and limitations under the License.
"""
from __future__ import division

from itertools import takewhile, dropwhile, islice, imap,\
        count, permutations, chain, ifilter, groupby, cycle
from math import sqrt, factorial, log
from scipy import array, fliplr, arange, ones, nonzero
from scipy.linalg import toeplitz, circulant

import english_numbers
import itertools

__docformat__ = "restructuredtext en"

###############
### Classes ###
###############

class TreeNode():
    def __init__(self, value=None, left=None, right=None):
        self.value = value
        self.left = left
        self.right = right
    def __repr__(self):
        return str(self.value)


#################
### Functions ###
#################

def nth(iterable, n):
    return islice(iterable, n, n+1).next()

def takefirst(pred, iterable):
    return dropwhile(lambda x: not pred(x), iterable).next()

def take(n, iterable):
    return islice(iterable, 0, n)

def chars(n):
    """Return list of n's digits as chars."""
    return [x for x in str(n)]

def cton(ns):
    """Convert list of chars to a number."""
    return int("".join(ns))

def digits(n):
    """Return list of n's digits."""
    return [int(x) for x in str(n)]

def ndigits(integer):
    """Return number of digits in an int."""
    return len(chars(integer))

def even(n):
    return n%2 == 0

def odd(n):
    return not even(n)

def collatz_sequence(n):
    while n != 1:
        yield n
        if even(n):
            n = n/2
        else:
            n = 3*n + 1
    yield n

def collatz_lens():
    """Yield the lengths of collatz sequences starting with 1."""
    return (len(list(collatz_sequence(x))) for x in count(1))

def triangle_ns():
    total = 0
    ns = count(1)
    while True:
        total += ns.next()
        yield total

def divides(x, n):
    """Return if n%x == 0."""
    return n % x == 0

def divisors(n):
    """Yield all divisors by trial division."""
    return (x for x in xrange(1,n+1) if divides(x,n))

def proper_divisors(n):
    """Yield all proper divisors (doesn't include n) by trial
    division."""
    return chain((1,), (x for x in xrange(2,n+1//2) if divides(x,n)))

def ndivisor_impl0(n):
    """Compute the number of divisors of a number."""
    return len(list(proper_divisors(n))) + 1

def ndivisors_impl1(n):
    """Return the number of divisors of a number.

    Given a prime factorization of number
        n = p1^c1 + p2^c2 + ... + pm^cm.

    The number of divisors of a number is equal to
        (c1 + 1) + (c2 + 1) + ... + (cm + 1).

    From Wikipedia, `Highly Composite Number`.
    """
    cs = array([len(list(x[1])) for x in groupby(prime_factors(n))])
    cs += 1
    return cs.prod()

ndivisors = ndivisors_impl1

def rotations(n):
    """Yield all possible circular shifts of a string."""
    rot_mat = circulant(chars(n)).T
    for row in rot_mat:
        yield cton(row)

def reversed_n(n):
    """Return n with digits reversed."""
    return cton(reversed(chars(n)))

def left_truncations(n):
    s = chars(n)
    for offset in range(len(s)):
        yield cton(s[offset:])

def right_truncations(n):
    s = chars(n)
    for offset in range(len(s)):
        yield cton(s[:offset+1])

def truncations(n):
    """Yield all left and right truncations of n."""
    return chain(left_truncations(n), right_truncations(n))

known_primes = [2, 3, 5, 7]
def is_prime(n):
    """Check for primality by dividing by known primes."""
    def divisible_by_known_prime(n):
        for x in known_primes:
            if divides(x,n):
                return x
        return False

    # Quick filters
    if n in known_primes:
        return True
    elif divisible_by_known_prime(n):
        return False
    else:
        # Start searching again at the max known prime
        start = known_primes[-1] + 2
        for y in xrange(start, int(sqrt(n)), 2):
            if divisible_by_known_prime(y):
                continue
            else:
                if n%y == 0:
                    return False
                known_primes.append(y)
    return True

def prime_count(n):
    """Return approximate number of primes less than n."""
    return n / log(n)

def nth_prime_impl0(n):
    """Calculate nth prime by trial division."""
    return nth((x for x in count(2) if is_prime(x)), n-1)

def nth_prime_impl1(n):
    """Calculate nth prime by trial division.

    Start prime search close to the correct place by estimating with the prime
    counting function.
    """
    if n == 1:
        return 2
    elif n == 2:
        return 3
    else:
        ubound = takefirst(lambda x: prime_count(x) > 1.2*n, count(2))
        return nth(primes(ubound), n-1)

# Top level nth prime function.
nth_prime = nth_prime_impl1

def prime_factors(num):
    """Compute prime factors by trial division."""
    x = 2
    n = num
    while x < (n//2+1):
        if divides(x,n) and is_prime(x):
            yield x
            n = n//x
            x = 2
            continue
        x+=1
    yield n

def primes_impl0():
    """Yield primes by trial division."""
    return (x for x in count(2) if is_prime(x))

def primes_impl1(n):
    """Yield primes less than n using Sieve of Eratosthenes."""
    possibles = range(2, n)
    while len(possibles) > 0:
        newest_prime = possibles[0]
        yield newest_prime
        possibles = [x for x in possibles if x%newest_prime!=0]

def primes_impl2(n):
    """Yield primes less than n using Sieve of Eratosthenes."""
    yield 2
    possibles = arange(3, n, 2, dtype=int)
    while len(possibles) > 0:
        newest_prime = possibles[0]
        yield newest_prime
        possibles = possibles[possibles%newest_prime!=0]

def primes_impl3(n):
    """Return list of primes less than n using Sieve of Eratosthenes.

    Thought it would be faster since it marks non-primes in place, reusing the
    same numpy array.  Isn't.
    """
    # initial = arange(3, n, 2, dtype=int)
    # possibles = zeros(len(initial) + 1, dtype=int)
    # possibles[0] = 2
    # possibles[1:] = initial
    possibles = wheel_factorization(n)
    marks = ones(len(possibles), dtype=bool)
    index = 1
    while index < len(possibles):
        newest_prime = possibles[index]
        index += 1
        if index >= len(possibles):
            break
        remaining = possibles[index:]
        marks[index:][remaining%newest_prime==0] = False
        while index < len(possibles) and marks[index] != 1:
            index += 1
    return possibles[marks]

def wheel_factorization(n, nprimes=10):
    """Use wheel factorization to jump-start sieving.

    Inner circle is product(first nprimes primes) long.
    nprimes is reduced if inner_circle_len >= n/2
    """
    while True:
        initial_primes = array(list(take(nprimes, primes_impl0())), dtype=int)
        inner_circle_len = initial_primes.prod()
        if inner_circle_len < n/2: # heuristic for good nprimes
            break
        nprimes -= 1
    overshoot = inner_circle_len - (n % inner_circle_len) + 1
    wheel = arange(1, n+overshoot, dtype=int).reshape((-1, inner_circle_len))
    wheel[0,0] = 0 # strike off 1
    for x in initial_primes:
        # strike off spoke of prime
        wheel[1:, x-1] = 0
        # strike off spokes of multiples of prime
        doomed_spokes = nonzero(wheel[0] *
                               (wheel[0] != x) *
                               (wheel[0] % x == 0))
        wheel[:, doomed_spokes] = 0
    wheel[wheel >= n] = 0
    return wheel[nonzero(wheel)].flatten()

def primes_impl4(n):
    """Find primes less than n using Sieve of Eratosthenes and wheel
    factorization."""
    possibles = wheel_factorization(n)
    while len(possibles) > 0:
        newest_prime = possibles[0]
        yield newest_prime
        possibles = possibles[possibles%newest_prime!=0]

def primes(n=None):
    """Top level primes generator."""
    if n is None:
        return primes_impl0()
    else:
        return primes_impl4(n)

def fibonacci():
    """Yield Fibonacci sequence."""
    n0 = 0
    n1 = 1
    while True:
        last_sum = n0 + n1
        n0 = n1
        n1 = last_sum
        yield n1

def is_palindrome(n):
    return chars(n) == list(reversed(chars(n)))

def pythagorean_triples(n):
    """Yield triples less than n."""
    return ((a, b, sqrt(a**2 + b**2)) for a in xrange(1, n-1)
                                      for b in xrange(a, n))

def amicables():
    """Yield amicable numbers."""
    def d(n):
        return sum(proper_divisors(n))
    return (x for x in count(1) if d(d(x)) == x and x != d(x))

def ncr(n, r):
    """Compute n choose r."""
    return factorial(n) / (factorial(n-r) * factorial(r))

def tetration(a, b):
    "Return the tetration of a to the b."""
    if b == 1:
        return a
    else:
        return a**tetration(a, b-1)

def is_pandigital(n):
    """Determine if a number makes use of all the digits 1:u+1 exactly
    once, where u is ndigits(n).
    """
    num_digits = ndigits(n)
    if num_digits > 9:
        return False
    digit_set = set(digits(n))
    matches = digits(123456789)
    return len(digit_set) == num_digits and \
                digit_set == set(matches[:ndigits(n)])

def pandigitals(n):
    """Yield all pandigital numbers with n digits."""
    digits = '987654321'
    for x in permutations(digits[len(digits)-n:]):
        yield cton(x)

def is_abundant(n):
    return sum(proper_divisors(n)) > n

def abundants():
    """Yield abundant numbers.

    A number n is abundant if the sum of its proper divisors is greater
    than n.

    12 is the smallest abundant number:
        1 + 2 + 3 + 4 + 6 = 16.
    """
    return (x for x in count(12) if is_abundant(x))

################
### Problems ###
################

def problem1(n=1000):
    """Sum all multiples of 3 or 5 below n=1000."""
    return sum(x for x in range(1,n) if x%3==0 or x%5==0)

def problem2(n=4e6):
    """Sum all even Fibonacci numbers below n=4e6."""
    fs = takewhile(lambda x: x<n, fibonacci())
    return sum(x for x in fs if even(x))

def problem3(n=600851475143):
    """Compute greatest prime factor of n."""
    return max(x for x in prime_factors(n))

def problem4(ndigits=3):
    """Compute largest palindrome made from product of two 3-digit numbers."""
    llimit = int('9' + '0'*(ndigits-1))
    ulimit = int('9'*ndigits) + 1
    products = (x*y for x in xrange(llimit, ulimit)
                    for y in xrange(llimit, ulimit))
    return max(x for x in products if is_palindrome(x))

def problem5(ulimit=21):
    """Compute smallest number divisible by 1:`ulimit`."""
    divisors = range(ulimit-1, 0, -1)
    def all_divide(n):
        return all(divides(x,n) for x in divisors)
    possibles = count(ulimit-1, ulimit-1)
    return takefirst(all_divide, possibles)

def problem6(ulimit=101):
    """Compute square-of-sum - sum-of-squares of 1:`ulimit`."""
    nums = xrange(1,ulimit)
    return (sum(nums))**2 - sum(x**2 for x in nums)

def problem7(n=10001):
    """Compute nth prime number."""
    return nth_prime(n)

def problem8():
    """Compute greatest product of five consecutive digits in p8_data."""
    str_data = p8_data
    data = array([int(x) for x in chars(str_data)])
    return toeplitz([7,0,0,0,0], data).prod(0).max()

def problem9(n=1000):
    """Find pythagorean triple s.t. a + b + c = 1000."""
    triple = takefirst(lambda x: sum(x)==n, pythagorean_triples(n))
    return int(array(triple).prod())

def problem10(n=int(2e6)):
    """Compute sum of primes below n.

    TODO: Not yet fast enough.
    """
    return sum(primes(n))

def problem11():
    """Compute greatest product of four adjacent numbers in p11_data in any
    direction.

    Directions are up, down, left, right or diagonally.
    """
    data = p11_data
    def max_vertical_product(dta, left_diagonal=False):
        maxes = []
        rnge = 17
        for start_row in range(rnge):
            chunk = dta[start_row:start_row+4, :]
            if left_diagonal:
                chunk = array([chunk.diagonal(offset=x) for x in
                    range(rnge)]).T
            maxes.append(chunk.prod(0).max())
        return max(maxes)

    return max(max_vertical_product(data),
            max_vertical_product(data.T),
            max_vertical_product(data, left_diagonal=True),
            max_vertical_product(fliplr(data), left_diagonal=True))

def problem12(lim=500):
    """Compute the first triangle number with over 500 divisors.
    """
    return takefirst(lambda n: ndivisors(n) > lim, triangle_ns())

def problem13():
    """Compute first 10 digits of sum of p13_data."""
    data = p13_data
    return int(''.join(chars(sum(data))[:10]))

def problem14_impl0(n=int(1e6)):
    """Find longest Collatz sequence starting under 1e6.

    TODO: Not yet fast enough.
    """
    maxind = 0
    maxlen = 0
    for ind, chainlen in enumerate(len(list(collatz_sequence(x))) for x in
            xrange(1,n)):
        if chainlen > maxlen:
            maxlen = chainlen
            maxind = ind
            ind = ind + 1
    return maxind

def problem14_impl1(n=int(1e6)):
    """Find longest Collatz sequence starting under 1e6.

    By inspection, is vaguely monotonic.  Start near the end.
    """
    lowerbound = n-100000
    len_collatz = lambda x: len(list(collatz_sequence(x)))
    return array([len_collatz(x) for x in xrange(lowerbound,n)],
            dtype=int).argmax() + lowerbound

def problem14_impl2(n=int(1e6)):
    """Find longest Collatz sequence starting under 1e6.

    Use collatz_lens generator.
    """
    return array(list(take(n, collatz_lens()))).argmax() + 1

problem14 = problem14_impl2

problem15_npaths = 0
problem15_done = False
def problem15_impl0(side=20, progress=False):
    """Find the number of paths from the top left to bottom right in a
    20x20 grid.

    Brute force approach - enumerate all the paths, aside from the
    diagonal symmetry.  Ugly implementation with global variables.

    TODO: Not yet fast enough.

    Diagram:
        x x x x x x x x x x x x x x x x x x x x x
        x x x x x x x x x x x x x x x x x x x x x
        x x x x x x x x x x x x x x x x x x x x x
        x x x x x x x x x x x x x x x x x x x x x
        x x x x x x x x x x x x x x x x x x x x x
        x x x x x x x x x x x x x x x x x x x x x
        x x x x x x x x x x x x x x x x x x x x x
        x x x x x x x x x x x x x x x x x x x x x
        x x x x x x x x x x x x x x x x x x x x x
        x x x x x x x x x x x x x x x x x x x x x
        x x x x x x x x x x x x x x x x x x x x x
        x x x x x x x x x x x x x x x x x x x x x
        x x x x x x x x x x x x x x x x x x x x x
        x x x x x x x x x x x x x x x x x x x x x
        x x x x x x x x x x x x x x x x x x x x x
        x x x x x x x x x x x x x x x x x x x x x
        x x x x x x x x x x x x x x x x x x x x x
        x x x x x x x x x x x x x x x x x x x x x
        x x x x x x x x x x x x x x x x x x x x x
        x x x x x x x x x x x x x x x x x x x x x
        x x x x x x x x x x x x x x x x x x x x x

       side = 2  3  4   5   6    7     8     9     10     11      12       13       14        15
    lenpath = 4  6  8  10  12   14    16    18     20     22      24       26       28        30
     npaths = 6 20 70 252 924 3432 12870 48620 184756 705432 2704156 10400600 40116600 155117520
    time(s) =                       0.02  0.07   0.21   0.80    3.18    10.99    42.10    163.10
    """
    global problem15_npaths, problem15_done
    problem15_npaths = 0
    problem15_done = False
    ulimit = side
    def traverse_path(x=0, y=0):
        global problem15_npaths, problem15_done
        if x == ulimit and y == ulimit:
            problem15_npaths += 1
            if progress == True and divides(100000, problem15_npaths):
                print problem15_npaths
            return
        if not problem15_done:
            if x < ulimit:
                traverse_path(x+1, y)
            if x == 0 and y == 0:
                problem15_done = True
            if y < ulimit:
                traverse_path(x, y+1)

    traverse_path()
    return problem15_npaths * 2

def problem15_impl1(side_len=20, progress=False):
    """Find the number of paths from the top left (0,0) to bottom right
    (side_len, side_len) in a 20x20 grid.

    Dynamic programming approach.
    """
    nverts = side_len + 1
    cache = {}
    cache[(nverts-1, nverts-1)] = 0
    cache[(nverts-2, nverts-1)] = 1
    cache[(nverts-1, nverts-2)] = 1
    def count_paths(cache, x=0, y=0):
        try:
            return cache[(x, y)]
        except(KeyError):
            right_paths = count_paths(cache, x+1, y) if x < nverts else 0
            down_paths  = count_paths(cache, x, y+1) if y < nverts else 0
            cache[(x,y)] = right_paths + down_paths
            cache[(y,x)] = cache[(x,y)] # by symmetry
            return cache[(x, y)]

    return count_paths(cache, 0, 0)

problem15 = problem15_impl1

def problem16():
    """Find sum of digits in 2**1000."""
    return sum(int(c) for c in str(2**1000))

def problem17(ulimit=1001):
    """Compute how many letters used when writing out all the natural
    numbers in 1:`ulimit`.
    """
    return english_numbers.count_chars(ulimit)

def problem18():
    """Find the maximum total path top to bottom of the triangle in
    p18_data.
    """
    def make_tree(triangle):
        """Make a binary tree out of the p18_data."""
        levels = [[TreeNode(item) for item in level]
                for level in triangle]
        for lindex, level in enumerate(levels[:-1]):
            current_index = 0
            for item in level:
                item.left = levels[lindex+1][current_index]
                current_index += 1
                item.right = levels[lindex+1][current_index]
        return levels[0][0]

    sums = []
    def sum_paths(head=None, total=0):
        """Walk the tree and sum all the paths."""
        if head.value is not None:
            sum_paths(head.left, total + head.value)
        if head.value is not None:
            sum_paths(head.right, total + head.value)
        else:
            sums.append(total)

    tree = make_tree(p18_data)
    sum_paths(tree)
    return max(sums)

def is_leap_year(year):
    retval = False
    if not divides(100, year):
        if divides(4, year):
            retval = True
    else:
        if divides(400, year):
            retval = True
    return retval

def dates():
    """Generate a dictionary {weekday, day, month, year} starting
    Monday, 1 January 1900.
    """
    years = count(1900)
    months = cycle(['January', 'February', 'March', 'April', 'May', 'June',
        'July', 'August', 'September', 'October', 'November', 'December'])
    daysinmonth = cycle([lambda year: 31,
                   lambda year: 29 if is_leap_year(year) else 28,
                   lambda year: 31,
                   lambda year: 30,
                   lambda year: 31,
                   lambda year: 30,
                   lambda year: 31,
                   lambda year: 31,
                   lambda year: 30,
                   lambda year: 31,
                   lambda year: 30,
                   lambda year: 31])
    weekdays = cycle(['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday',
        'Saturday', 'Sunday'])

    year = years.next()
    month = months.next()
    weekday = weekdays.next()
    days = (x for x in xrange(1,daysinmonth.next()(year)+1))
    day = days.next()
    while True:
        yield {'weekday': weekday,
               'year': year,
               'month': month,
               'day': day}
        weekday = weekdays.next()
        try:
            day = days.next()
        except(StopIteration):
            days = (x for x in xrange(1,daysinmonth.next()(year)+1))
            day = days.next()
            month = months.next()
            if month == 'January':
                year = years.next()

def problem19():
    """Compute how many Sundays fell on the first of the month between
    (1 Jan 1901 and 31 Dec 2000).
    """
    all_dates = dates()
    all_dates = dropwhile(lambda date: date['year'] < 1901, all_dates)
    date_range = takewhile(lambda date: date['year'] < 2001, all_dates)
    return sum(1 for d in date_range if d['day'] == 1
                                    and d['weekday'] == 'Sunday')


def problem20():
    """Find sum of digits in 100!."""
    return sum(int(x) for x in chars(factorial(100)))

def problem21(ulimit=int(1e4)):
    """Find sum of all amicable numbers below 1e4."""
    return sum(takewhile(lambda x: x < ulimit, amicables()))

def problem22():
    """Total all `name scores` in names.txt.

    name_score = alphabetical_position * numeric_value_of_name,
        where numeric_value_of_name is defined by `value` nested function.
    """
    with open('names.txt', 'r') as f:
        names = eval(f.readline())
    names.sort()

    def value(name):
        return sum(ord(x)-64 for x in name)

    return sum((index+1) * value(name) for index, name in enumerate(names))

def problem23():
    """Find the sum of all positive integers that aren't the sum of two
    abundant numbers.

    A number n is abundant if the sum of its proper divisors in less
    than n.

    12 is the smallest abundant number:
        1 + 2 + 3 + 4 + 6 = 16.

    Then, 24 is the smallest number that is the sum of two abundant
    numbers.  All integers greater than 28123 are the sum of two
    abundant numbers.

    TODO: A little slow (143.10s).
    """
    ulimit = 28123
    eligible = list(takewhile(lambda x: x < ulimit, abundants()))
    sums_of_abundant = set(map(sum, itertools.product(eligible, eligible)))
    not_sums = set(range(1,ulimit)) - sums_of_abundant
    return sum(not_sums)

def problem24(n=int(1e6)):
    """Find nth lexicographic permutation of 0123456789."""
    ps = sorted(''.join(x) for x in permutations('0123456789'))
    return ps[n-1]

def problem25(digits=1000):
    """Find first Fibonacci term to contain `digits` number of digits."""
    fibs = enumerate(fibonacci())
    return dropwhile(lambda x: ndigits(x[1]) < digits, fibs).next()[0] + 2

def problem29(alimit=100, blimit=100):
    """Count all distinct terms in the sequence a**b s.t. 2 <= a <= 100
    and 2 <= b <= 100?
    """
    return len(set(a**b for a in xrange(2,alimit+1) for b in
                                 xrange(2,blimit+1)))

def problem34(ulimit=int(1e5)):
    """Find the sum of all numbers equal to the sum of the factorial of their
    digits."""
    def sum_of_fac_of_digits(n):
        return sum(factorial(x) for x in digits(n))
    return sum([x for x in xrange(3,ulimit) if x == sum_of_fac_of_digits(x)])

def problem35(ulimit=int(1e6)):
    """Count how many circular primes there are below `ulimit`."""
    pset = set(primes(ulimit))
    cprimes = []
    for p in pset:
        if set(rotations(p)).issubset(pset):
            cprimes.append(p)
    return len(cprimes)

def problem36(ulimit=int(1e6)):
    """Find all numbers less than `ulimit` which are palindromic in
    bases 10 and 2.
    """
    return sum(x for x in xrange(ulimit) if
                    is_palindrome(x) and
                    is_palindrome(bin(x)[2:]))

def problem37(ulimit=int(1e6)):
    """Find the sum of the only 11 primes that are both truncatable from
    left to right and right to left.
    """
    ps = set(primes(ulimit))
    truncatable_ps = (x for x in ps if set(truncations(x)).issubset(ps)
                             and x not in set([2,3,5,7]))
    return sum(take(11, truncatable_ps))

def problem40():
    """Find the product of several digits of an irrational number.

    The irrational number that created when concatenating the positive
    integers:

        0.123456789101112131415161718192021...

    Find d_1 * d_10 * d_100 * d_1000 * d_10000 * d_100000 * d_1000000.
    """
    number = []
    for x in xrange(1000000):
        number += chars(x)
    return array([int(number[int(1*10**x)]) for x in range(6)]).prod()

def problem41_impl0():
    """Find the largest pandigital prime.

    A pandigital number can't be more than 9 digits, so we'll start with
    the nine-digits and work down, narrowing the field by checking only
    odds.
    """
    ulimit = int(1e9)
    return takefirst(is_pandigital, count(ulimit-1, -2))

def problem41_impl1(num_digits=9):
    """Find the largest pandigital prime.

    A pandigital number can't be more than 9 digits, so we'll compute
    all the 9-digit pandigital numbers and take the max prime.
    """
    for n in xrange(num_digits, 0, -1):
        prime_pandigitals = ifilter(is_prime, sorted(pandigitals(n),
                                                        reverse=True))
        try:
            return prime_pandigitals.next()
        except(StopIteration):
            continue

problem41 = problem41_impl1

def problem48(ulimit=1000):
    """Compute the sum of 1**1, 2**2, ... `ulimit`**`ulimit`."""
    total = sum(x**x for x in xrange(1,ulimit+1))
    return cton(chars(total)[-10:])

def problem52():
    """Find the smallest positive integer x, s.t. 2x, 3x, 4x, 5x, and 6x
    contain the same digits.
    """
    def digits_match(x):
        return all(set(digits(x)) == set(digits(m*x))
                                             for m in xrange(2,7))
    return takefirst(digits_match, count(1))

def problem53(ulimit=101, bound=int(1e6)):
    """Compute how many values of ncr(x) for x in 1..`ulimit` are
    greater than `bound`.
    """
    overs = (ncr(n, r) for n in xrange(ulimit)
                       for r in xrange(n)
                       if ncr(n,r) > bound)
    return len(list(overs))

def problem56(ulimit=100):
    """Considering natural numbers of the form, a**b, where a, b <= 100,
    what is the maximum digital sum?
    """
    return max(sum(digits(x**y)) for x in xrange(1,ulimit)
                                 for y in xrange(1,ulimit))

def problem345_impl0(m):
    """Find the Matrix Sum of matrix `m`.

    The Matrix Sum of a matrix m is defined as the maximum sum of matrix
    elements in m such that each element is the only element is its own
    row and column.

    TODO: Not yet fast enough.
    """
    nrows, ncols = m.shape
    rows = array(range(nrows))
    pcols = (array(p) for p in permutations(xrange(ncols)))
    return max(m[rows, cols].sum() for cols in pcols)

def problem345_impl1(m):
    """Find the Matrix Sum of matrix `m`.

    The Matrix Sum of a matrix m is defined as the maximum sum of matrix
    elements in m such that each element is the only element is its own
    row and column.

    TODO: Not yet fast enough.
    """
    nrows, ncols = m.shape
    rows = array(range(nrows))
    pcols = (array(p) for p in permutations(xrange(ncols)))
    nsums = factorial(ncols)
    count = 0
    cached_max = 0
    skip = 100000
    for cols in pcols:
        current_sum = m[rows, cols].sum()
        if current_sum > cached_max:
            cached_max = current_sum
        if divides(skip, count):
            print count / nsums * 100, "% Done"
        count += 1
    return cached_max

def problem345_impl2(m):
    """Find the Matrix Sum of matrix `m` using memoization.

    The Matrix Sum of a matrix m is defined as the maximum sum of matrix
    elements in m such that each element is the only element is its own
    row and column.

    TODO: Not yet fast enough.  This solution is probably an incorrect
    approach--the limitation is not the n-number sum (15 for the full
    problem matrix), but how many sums are required (n!).  For this
    problem to be answerable in under a minute, there must be some way
    to reduce that space.
    """
    nrows, ncols = m.shape
    rows = array(range(nrows))
    pcols = (array(p) for p in permutations(xrange(ncols)))
    nsums = factorial(ncols)

    cache = {}
    def sums_cache(rows, cols):
        if len(cols) == 1:
            return m[rows[0], cols[0]]
        else:
            try:
                total = cache[tuple(cols)]
                return total
            except(KeyError):
                total = sums_cache(rows[:-1], cols[:-1]) + m[rows[-1], cols[-1]]
                cache[tuple(cols)] = total
                return total

    count = 0
    cached_max = 0
    skip = 100000
    for cols in pcols:
        current_sum = sums_cache(rows, cols)
        if current_sum > cached_max:
            cached_max = current_sum
        if divides(skip, count):
            print count / nsums * 100, "% Done"
        count += 1
    return cached_max

def problem345_impl3(m):
    """Find the Matrix Sum of matrix `m`.

    The Matrix Sum of a matrix m is defined as the maximum sum of matrix
    elements in m such that each element is the only element is its own
    row and column.

    This implementation attempted to select the maximum algorithmically.
    Unfortunately, it produces an incorrect solution.
    """
    maxvals = []
    nrows, ncols = m.shape
    while ncols > 0:
        print m
        col_max = m[:, 0].argmax()
        row_max = m[col_max, :].argmax()
        maxvals.append(m[col_max, row_max])
        m = m.take(range(col_max) + range(col_max+1, nrows), axis=0)
        m = m.take(range(row_max) + range(row_max+1, ncols), axis=1)
        nrows, ncols = m.shape
    return maxvals

def problem345_impl4(m):
    """Find the Matrix Sum of matrix `m`.

    The Matrix Sum of a matrix m is defined as the maximum sum of matrix
    elements in m such that each element is the only element is its own
    row and column.

    This implementation attempted to select the maximum algorithmically.
    Unfortunately, it produces an incorrect solution.
    """
    maxvals = []
    nrows, ncols = m.shape
    while ncols > 0:
        print m
        maxvals.append(m.max())
        row, col = (m == m.max()).nonzero()
        m = m.take(range(row) + range(row+1, nrows), axis=0)
        m = m.take(range(col) + range(col+1, ncols), axis=1)
        nrows, ncols = m.shape
    return maxvals


############
### Data ###
############

p8_data = \
    "73167176531330624919225119674426574742355349194934"\
    "96983520312774506326239578318016984801869478851843"\
    "85861560789112949495459501737958331952853208805511"\
    "12540698747158523863050715693290963295227443043557"\
    "66896648950445244523161731856403098711121722383113"\
    "62229893423380308135336276614282806444486645238749"\
    "30358907296290491560440772390713810515859307960866"\
    "70172427121883998797908792274921901699720888093776"\
    "65727333001053367881220235421809751254540594752243"\
    "52584907711670556013604839586446706324415722155397"\
    "53697817977846174064955149290862569321978468622482"\
    "83972241375657056057490261407972968652414535100474"\
    "82166370484403199890008895243450658541227588666881"\
    "16427171479924442928230863465674813919123162824586"\
    "17866458359124566529476545682848912883142607690042"\
    "24219022671055626321111109370544217506941658960408"\
    "07198403850962455444362981230987879927244284909188"\
    "84580156166097919133875499200524063689912560717606"\
    "05886116467109405077541002256983155200055935729725"\
    "71636269561882670428252483600823257530420752963450"

p11_data = array([
    [8,2,22,97,38,15,0,40,0,75,4,5,7,78,52,12,50,77,91,8],
    [49,49,99,40,17,81,18,57,60,87,17,40,98,43,69,48,4,56,62,0],
    [81,49,31,73,55,79,14,29,93,71,40,67,53,88,30,3,49,13,36,65],
    [52,70,95,23,4,60,11,42,69,24,68,56,1,32,56,71,37,2,36,91],
    [22,31,16,71,51,67,63,89,41,92,36,54,22,40,40,28,66,33,13,80],
    [24,47,32,60,99,3,45,2,44,75,33,53,78,36,84,20,35,17,12,50],
    [32,98,81,28,64,23,67,10,26,38,40,67,59,54,70,66,18,38,64,70],
    [67,26,20,68,2,62,12,20,95,63,94,39,63,8,40,91,66,49,94,21],
    [24,55,58,5,66,73,99,26,97,17,78,78,96,83,14,88,34,89,63,72],
    [21,36,23,9,75,0,76,44,20,45,35,14,0,61,33,97,34,31,33,95],
    [78,17,53,28,22,75,31,67,15,94,3,80,4,62,16,14,9,53,56,92],
    [16,39,5,42,96,35,31,47,55,58,88,24,0,17,54,24,36,29,85,57],
    [86,56,0,48,35,71,89,7,5,44,44,37,44,60,21,58,51,54,17,58],
    [19,80,81,68,5,94,47,69,28,73,92,13,86,52,17,77,4,89,55,40],
    [4,52,8,83,97,35,99,16,7,97,57,32,16,26,26,79,33,27,98,66],
    [88,36,68,87,57,62,20,72,3,46,33,67,46,55,12,32,63,93,53,69],
    [4,42,16,73,38,25,39,11,24,94,72,18,8,46,29,32,40,62,76,36],
    [20,69,36,41,72,30,23,88,34,62,99,69,82,67,59,85,74,4,36,16],
    [20,73,35,29,78,31,90,1,74,31,49,71,48,86,81,16,23,57,5,54],
    [1,70,54,71,83,51,54,69,16,92,33,48,61,43,52,1,89,19,67,48]])

p13_data = [37107287533902102798797998220837590246510135740250,
        46376937677490009712648124896970078050417018260538,
        74324986199524741059474233309513058123726617309629,
        91942213363574161572522430563301811072406154908250,
        23067588207539346171171980310421047513778063246676,
        89261670696623633820136378418383684178734361726757,
        28112879812849979408065481931592621691275889832738,
        44274228917432520321923589422876796487670272189318,
        47451445736001306439091167216856844588711603153276,
        70386486105843025439939619828917593665686757934951,
        62176457141856560629502157223196586755079324193331,
        64906352462741904929101432445813822663347944758178,
        92575867718337217661963751590579239728245598838407,
        58203565325359399008402633568948830189458628227828,
        80181199384826282014278194139940567587151170094390,
        35398664372827112653829987240784473053190104293586,
        86515506006295864861532075273371959191420517255829,
        71693888707715466499115593487603532921714970056938,
        54370070576826684624621495650076471787294438377604,
        53282654108756828443191190634694037855217779295145,
        36123272525000296071075082563815656710885258350721,
        45876576172410976447339110607218265236877223636045,
        17423706905851860660448207621209813287860733969412,
        81142660418086830619328460811191061556940512689692,
        51934325451728388641918047049293215058642563049483,
        62467221648435076201727918039944693004732956340691,
        15732444386908125794514089057706229429197107928209,
        55037687525678773091862540744969844508330393682126,
        18336384825330154686196124348767681297534375946515,
        80386287592878490201521685554828717201219257766954,
        78182833757993103614740356856449095527097864797581,
        16726320100436897842553539920931837441497806860984,
        48403098129077791799088218795327364475675590848030,
        87086987551392711854517078544161852424320693150332,
        59959406895756536782107074926966537676326235447210,
        69793950679652694742597709739166693763042633987085,
        41052684708299085211399427365734116182760315001271,
        65378607361501080857009149939512557028198746004375,
        35829035317434717326932123578154982629742552737307,
        94953759765105305946966067683156574377167401875275,
        88902802571733229619176668713819931811048770190271,
        25267680276078003013678680992525463401061632866526,
        36270218540497705585629946580636237993140746255962,
        24074486908231174977792365466257246923322810917141,
        91430288197103288597806669760892938638285025333403,
        34413065578016127815921815005561868836468420090470,
        23053081172816430487623791969842487255036638784583,
        11487696932154902810424020138335124462181441773470,
        63783299490636259666498587618221225225512486764533,
        67720186971698544312419572409913959008952310058822,
        95548255300263520781532296796249481641953868218774,
        76085327132285723110424803456124867697064507995236,
        37774242535411291684276865538926205024910326572967,
        23701913275725675285653248258265463092207058596522,
        29798860272258331913126375147341994889534765745501,
        18495701454879288984856827726077713721403798879715,
        38298203783031473527721580348144513491373226651381,
        34829543829199918180278916522431027392251122869539,
        40957953066405232632538044100059654939159879593635,
        29746152185502371307642255121183693803580388584903,
        41698116222072977186158236678424689157993532961922,
        62467957194401269043877107275048102390895523597457,
        23189706772547915061505504953922979530901129967519,
        86188088225875314529584099251203829009407770775672,
        11306739708304724483816533873502340845647058077308,
        82959174767140363198008187129011875491310547126581,
        97623331044818386269515456334926366572897563400500,
        42846280183517070527831839425882145521227251250327,
        55121603546981200581762165212827652751691296897789,
        32238195734329339946437501907836945765883352399886,
        75506164965184775180738168837861091527357929701337,
        62177842752192623401942399639168044983993173312731,
        32924185707147349566916674687634660915035914677504,
        99518671430235219628894890102423325116913619626622,
        73267460800591547471830798392868535206946944540724,
        76841822524674417161514036427982273348055556214818,
        97142617910342598647204516893989422179826088076852,
        87783646182799346313767754307809363333018982642090,
        10848802521674670883215120185883543223812876952786,
        71329612474782464538636993009049310363619763878039,
        62184073572399794223406235393808339651327408011116,
        66627891981488087797941876876144230030984490851411,
        60661826293682836764744779239180335110989069790714,
        85786944089552990653640447425576083659976645795096,
        66024396409905389607120198219976047599490197230297,
        64913982680032973156037120041377903785566085089252,
        16730939319872750275468906903707539413042652315011,
        94809377245048795150954100921645863754710598436791,
        78639167021187492431995700641917969777599028300699,
        15368713711936614952811305876380278410754449733078,
        40789923115535562561142322423255033685442488917353,
        44889911501440648020369068063960672322193204149535,
        41503128880339536053299340368006977710650566631954,
        81234880673210146739058568557934581403627822703280,
        82616570773948327592232845941706525094512325230608,
        22918802058777319719839450180888072429661980811197,
        77158542502016545090413245809786882778948721859617,
        72107838435069186155435662884062257473692284509516,
        20849603980134001723930671666823555245252804609722,
        53503534226472524250874054075591789781264330331690]

p18_data = (\
    (75,),
    (95,64),
    (17,47,82),
    (18,35,87,10),
    (20,4,82,47,65),
    (19,1,23,75,3,34),
    (88,2,77,73,7,63,67),
    (99,65,4,28,6,16,70,92),
    (41,41,26,56,83,40,80,70,33),
    (41,48,72,33,47,32,37,16,94,29),
    (53,71,44,65,25,43,91,52,97,51,14),
    (70,11,33,28,77,73,17,78,39,68,17,57),
    (91,71,52,38,17,14,91,43,58,50,27,29,48),
    (63,66,4,68,89,53,67,30,73,16,69,87,40,31),
    (4,62,98,27,23,9,70,98,73,93,38,53,60,4,23),
    (None,None,None,None,None,None,None,None,None,None,None,None,None,None,None,None))

p345_test_data = array([
    [  7, 53,183,439,863],
    [497,383,563, 79,973],
    [287, 63,343,169,583],
    [627,343,773,959,943],
    [767,473,103,699,303]])

p345_test_data_2 = array([
    [  6, 53,183,439,863,497,383],
    [627,343,773,959,943,767,473],
    [447,283,463, 29, 23,487,463],
    [217,623,  3,399,853,407,103],
    [960,376,682,962,300,780,486],
    [870,456,192,162,593,473,915],
    [973,965,905,919,133,673,665]])

p345_test_data_3 = array([
    [  6, 53,183,439,863,497,383,563, 79],
    [627,343,773,959,943,767,473,103,699],
    [447,283,463, 29, 23,487,463,993,119],
    [217,623,  3,399,853,407,103,983, 89],
    [960,376,682,962,300,780,486,502,912],
    [870,456,192,162,593,473,915, 45,989],
    [973,965,905,919,133,673,665,235,509],
    [322,148,972,962,286,255,941,541,265],
    [445,721, 11,525,473, 65,511,164,138]])

p345_test_data_4 = array([
    [  6, 53,183,439,863,497,383,563, 79,973],
    [627,343,773,959,943,767,473,103,699,303],
    [447,283,463, 29, 23,487,463,993,119,883],
    [217,623,  3,399,853,407,103,983, 89,463],
    [960,376,682,962,300,780,486,502,912,800],
    [870,456,192,162,593,473,915, 45,989,873],
    [973,965,905,919,133,673,665,235,509,613],
    [322,148,972,962,286,255,941,541,265,323],
    [445,721, 11,525,473, 65,511,164,138,672],
    [414,456,310,312,798,104,566,520,302,248]])

p345_data = array([
    [  6, 53,183,439,863,497,383,563, 79,973,287, 63,343,169,583],
    [627,343,773,959,943,767,473,103,699,303,957,703,583,639,913],
    [447,283,463, 29, 23,487,463,993,119,883,327,493,423,159,743],
    [217,623,  3,399,853,407,103,983, 89,463,290,516,212,462,350],
    [960,376,682,962,300,780,486,502,912,800,250,346,172,812,350],
    [870,456,192,162,593,473,915, 45,989,873,823,965,425,329,803],
    [973,965,905,919,133,673,665,235,509,613,673,815,165,992,326],
    [322,148,972,962,286,255,941,541,265,323,925,281,601, 95,973],
    [445,721, 11,525,473, 65,511,164,138,672, 18,428,154,448,848],
    [414,456,310,312,798,104,566,520,302,248,694,976,430,392,198],
    [184,829,373,181,631,101,969,613,840,740,778,458,284,760,390],
    [821,461,843,513, 17,901,711,993,293,157,274, 94,192,156,574],
    [ 34,124,  4,878,450,476,712,914,838,669,875,299,823,329,699],
    [815,559,813,459,522,788,168,586,966,232,308,833,251,631,107],
    [813,883,451,509,615, 77,281,613,459,205,380,274,302, 35,805]])
