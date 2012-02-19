#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <math.h>
#include <assert.h>
#include <string.h>

bool is_prime(unsigned long composite);
void test_is_prime(void);

int max_nprimes_below(unsigned long n);
unsigned long nth_prime_upper_bound(int n);

unsigned long nth_prime(int n);
void test_nth_prime(void);

int prime_sieve(unsigned long n, unsigned long * primes);
void test_prime_sieve(void);

bool is_palindrome(const unsigned long num);
void test_is_palindrome(void);

int           problem1(void);
int           problem2(void);
unsigned long problem3(void);
unsigned long problem4(void);
int           problem5(void);
int           problem6(void);
unsigned long problem7(void);
unsigned long problem10(void);

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

// Upper bound for prime counting function
// From Wikipedia: Prime-counting function
int max_nprimes_below(unsigned long n) {
    if (n < 2) {
        return 0;
    } else {
        return ceil(1.25506 * (n / log(n)));
    }
}

// Upper bound for the nth prime, for n >= 6
// From Wikipedia: Prime-counting function
unsigned long nth_prime_upper_bound(int n) {
    n += 1; // this formula uses 1-based indexing
    return ceil(n*(log(n) + log(log(n))));
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
    unsigned long nprimes = 0;
    unsigned long * primes = malloc(sizeof(unsigned long) *
            (max_nprimes_below(N) + 1));

    printf("Testing prime number sieve...\n");
    nprimes = prime_sieve(N, primes);
    assert(nprimes == 14);

    for (int i=0; primes[i]; ++i) {
        printf("%ld == %ld\n", primes[i], known_primes[i]);
        assert(primes[i] == known_primes[i]);
    }

    free(primes);
}

unsigned long nth_prime(int n) {
    unsigned long answer = 0;
    unsigned long ulimit = -1;
    unsigned long * primes;

    if (n < 7) {
        ulimit = 20;
    } else {
        ulimit = nth_prime_upper_bound(n);
    }
    //printf("%ld, ", ulimit);
    primes = malloc(sizeof(unsigned long) * (max_nprimes_below(ulimit) + 1));

    prime_sieve(ulimit, primes);
    answer = primes[n];
    free(primes);

    return answer;
}

void test_nth_prime() {
    unsigned long known_primes[] = {2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37,
        41, 43, 0}; // 0 terminated
    for (int i=0; known_primes[i]; ++i) {
        //printf("%d: %ld\n", i, nth_prime(i));
        assert(nth_prime(i) == known_primes[i]);
    }
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
    unsigned long retval = 0;

    nprimes = prime_sieve(ulimit, primes);
    for (int i = nprimes-1; i >= 0; --i) {
        if (NUM % primes[i] == 0) {
            retval = primes[i];
            break;
        }
    }
    free(primes);
    return retval;
}

bool is_palindrome(const unsigned long num) {
    char num_str[10];
    int num_len = 0;
    snprintf(num_str, 10*sizeof(char), "%lu", num);
    num_len = strlen(num_str);
    for (int i=0; i < floor(num_len/2); ++i) {
        if (num_str[i] != num_str[num_len-i-1]) {
            return false;
        }
    }
    return true;
}

void test_is_palindrome() {
    assert(is_palindrome(90909));
    assert(is_palindrome(11111));
    assert(is_palindrome(19191));
    assert(is_palindrome(2222));
    assert(is_palindrome(363));
    assert(is_palindrome(44));
    assert(is_palindrome(5));
    assert(!is_palindrome(987654));
    assert(!is_palindrome(98765));
    assert(!is_palindrome(954));
    assert(!is_palindrome(21));
}

// Find the largest palindrome made from the product of two 3-digit numbers.
unsigned long problem4(void) {
    const int NPRODS = 99*(99+1)/2;
    unsigned long products[NPRODS];
    int index = 0;

    // Calculate products
    for (int i=900; i < 1000; ++i) {
        for (int j=i+1; j < 1000; ++j) {
            assert(index < NPRODS);
            products[index] = i*j;
            ++index;
        }
    }

    // Find max palindrome; linear scan
    unsigned long max_palindrome = 0;
    for (int i=0; i < NPRODS; ++i) {
        if (is_palindrome(products[i])) {
            if (products[i] > max_palindrome) {
                max_palindrome = products[i];
            }
            if (DEBUG) printf("%lu\n", products[i]);
        }
    }
    return max_palindrome;
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

// Find the 10001st prime.
unsigned long problem7(void) {
    const int PRIME_INDEX = 10000;
    return nth_prime(PRIME_INDEX);
}

// Find the sum of all primes below two million.
unsigned long problem10(void) {
    const unsigned long ULIMIT = 2000000;
    unsigned long * primes = malloc(sizeof(unsigned long) *
            (max_nprimes_below(ULIMIT) + 1));
    unsigned long sum = 0;
    prime_sieve(ULIMIT, primes);
    for (int i=0; primes[i]; ++i) {
        sum += primes[i];
    }
    return sum;
}

int main() {
    printf("%lu\n", problem4());
    return(0);
}
