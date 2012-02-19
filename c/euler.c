#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <math.h>
#include <assert.h>

bool is_prime(unsigned long composite);
void test_is_prime();

int max_nprimes_below(unsigned long n);

int prime_sieve(unsigned long n, unsigned long * primes);
void test_prime_sieve();

int problem1(void);
int problem2(void);
unsigned long problem3(void);
int problem5(void);
int problem6(void);

const bool DEBUG = false;


/*********************/
/* Utility functions */
/*********************/

// Test primality by trial division
bool is_prime(unsigned long n) {
    switch(n) {
        case 0:
        case 1:
            return false;
        case 2:
        case 3:
            return true;
        default:
            if (n % 2 == 0)
                return false;
            for (int i=3; i <= ceil(sqrt(n)); i+=2) {
                if (n % i == 0)
                    return false;
            }
            return true;
    }
}

void test_is_prime() {
    int primes[] = {2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 101,
        0};
    int composites[] = {4, 6, 8, 9, 10, 12, 14, 15, 16, 18, 20, 21, 22, 100,
        0};
    for (int i=0; primes[i]; ++i) {
//        printf("%d: %d\n", primes[i], is_prime(primes[i]));
        assert(is_prime(primes[i]));
    }
    for (int i=0; composites[i]; ++i) {
//        printf("%d: %d\n", composites[i], is_prime(composites[i]));
        assert(!is_prime(composites[i]));
    }
}

// Upper bound for prime counting function for n>1
// From Wikipedia: Prime-counting function
int max_nprimes_below(unsigned long n) {
    return ceil(1.25506 * (n / log(n)));
}

// Fills array 'primes' with of primes < n, terminates with a 0
// Returns nprimes
// Uses Sieve of Eratosthenes
int prime_sieve(unsigned long n, unsigned long * primes) {

    // indicates if pmask[i] is prime
    bool * pmask = malloc(sizeof(bool)*n);

    // initialize pmask
    for (unsigned long i=0; i < n; ++i)
        pmask[i] = true;

    // mark all non-primes false, using Sieve of Eratosthenes
    pmask[0] = pmask[1] = false;
    for (unsigned long i=2; i < n; ++i) {
        if (!pmask[i]) {
            continue;
        }
        if (DEBUG) { printf("%ld: ", i); }
        for (unsigned long j=i+i; j < n; j+=i) {
            if (DEBUG) { printf("%ld ", j); }
            pmask[j] = false;
        }
        if (DEBUG) { printf("\n"); }
    }

    // fill the primes array with known primes
    unsigned long pindex = 0;
    for (unsigned long i=2; i < n; ++i) {
        if (pmask[i]) {
            primes[pindex] = i;
            ++pindex;
        }
    }
    primes[pindex] = 0; // 0 terminate
    free(pmask);
    return(pindex);
}

void test_prime_sieve() {
    const int N = 44;
    unsigned long known_primes[] = {2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37,
        41, 43, 0}; // 0 terminated
    unsigned long * primes = malloc(sizeof(unsigned long) *
            (max_nprimes_below(N) + 1));
    unsigned long nprimes = 0;

    printf("Testing prime number sieve...\n");
    nprimes = prime_sieve(N, primes);
    assert(nprimes == 14);

    for (int i=0; primes[i]; ++i) {
        printf("%ld == %ld\n", primes[i], known_primes[i]);
        assert(primes[i] == known_primes[i]);
    }

    free(primes);
}


/************/
/* Problems */
/************/

// Find the sum of all the multiple of 3 or 5 below 1000.
int problem1(void) {
    const int ULIMIT = 1000;
    int i, sum;
    sum = 0;
    for (i = 3; i < ULIMIT; ++i) {
        if (i%3 == 0 || i%5 == 0)
            sum += i;
    }
    return(sum);
}

// Find the sum of the even-valued Fibonacci terms less than four
// million.
int problem2(void) {
    const int ULIMIT = 4000000;
    int a, b, sum;
    sum = 0;
    a = b = 1;
    while (b < ULIMIT) {
        if (b%2 == 0)
            sum += b;
        int cur = a + b;
        a = b;
        b = cur;
    }
    return(sum);
}

// What is the largest prime factor of the number 600851475143 ?
unsigned long problem3(void) {
    const unsigned long NUM = 600851475143;
    unsigned long ulimit = floor(sqrt(NUM)); // ulimit for prime factors
    unsigned long * primes = malloc(sizeof(unsigned long) *
            (max_nprimes_below(ulimit) + 1));
    unsigned long nprimes = 0;

    nprimes = prime_sieve(ulimit, primes);
    for (int i = nprimes-1; i >= 0; --i) {
        if (NUM % primes[i] == 0)
            return primes[i];
    }

    return -1; // no prime factor found
}

// Find the smallest number evenly divisible by all the numbers from 1
// to 20.
int problem5(void) {
    const int NDIVISORS = 10;
    // We don't need to try numbers that are multiples of other numbers
    const int divisors[] = {20, 19, 18, 17, 16, 15, 14, 13, 12, 11};
    bool success;
    for (int i = divisors[0];; i += divisors[0]) {
        success = true;
        for (int j = 1; j < NDIVISORS; ++j) {
            if (i % divisors[j] != 0) {
                success = false;
                break;
            }
        }
        if (success) return(i);
    }
}

// Find the difference between the sum of squares and the square of the
// sum of the first 100 natural numbers.
int problem6(void) {
    const int ULIMIT = 100;
    int i, sum, sum_of_squares;
    sum = (1 + ULIMIT) * (ULIMIT / 2);
    sum_of_squares = 0;
    for (i = 1; i <= ULIMIT; ++i) {
        sum_of_squares += i*i;
    }
    return(sum*sum - sum_of_squares);
}

int main() {
    printf("%ld\n", problem3());
    return(0);
}
