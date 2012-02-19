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
int           problem8(void);
int           problem9(void);
unsigned long problem10(void);

int problem8_data[];
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

int problem8(void) {
    int *d = problem8_data;
    int max_product, product = 1;
    for (int i=4; problem8_data[i] != -99; ++i) {
        product = d[i-4] * d[i-3] * d[i-2] * d[i-1] * d[i];
        if (product > max_product) {
            max_product = product;
        }
    }
    return max_product;
}

// There exists exactly one Pythagorean triplet for which a + b + c =
// 1000.  Find the product abc.
int problem9(void) {
    int target = 1000;
    int c = 1;
    int cc = 1;
    for (int a=1; a < target; ++a) {
        for (int b=a+1; b < target; ++b) {
            cc = a*a + b*b;
            c = round(sqrt(cc));
            if ((c > b) && (a+b+c == target) && (c*c == cc)) {
                if (DEBUG) {
                    printf("%d + %d + %d = %d\n", a, b, c, a+b+c);
                }
                return a*b*c;
            }
        }
    }
    return 0;
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
    printf("%lu\n", problem10());
    return(0);
}


/********/
/* Data */
/********/

int problem8_data[] = {
    7,3,1,6,7,1,7,6,5,3,1,3,3,0,6,2,4,9,1,9,2,2,5,1,1,9,6,7,4,4,2,6,5,7,4,7,4,2,3,5,5,3,4,9,1,9,4,9,3,4,
    9,6,9,8,3,5,2,0,3,1,2,7,7,4,5,0,6,3,2,6,2,3,9,5,7,8,3,1,8,0,1,6,9,8,4,8,0,1,8,6,9,4,7,8,8,5,1,8,4,3,
    8,5,8,6,1,5,6,0,7,8,9,1,1,2,9,4,9,4,9,5,4,5,9,5,0,1,7,3,7,9,5,8,3,3,1,9,5,2,8,5,3,2,0,8,8,0,5,5,1,1,
    1,2,5,4,0,6,9,8,7,4,7,1,5,8,5,2,3,8,6,3,0,5,0,7,1,5,6,9,3,2,9,0,9,6,3,2,9,5,2,2,7,4,4,3,0,4,3,5,5,7,
    6,6,8,9,6,6,4,8,9,5,0,4,4,5,2,4,4,5,2,3,1,6,1,7,3,1,8,5,6,4,0,3,0,9,8,7,1,1,1,2,1,7,2,2,3,8,3,1,1,3,
    6,2,2,2,9,8,9,3,4,2,3,3,8,0,3,0,8,1,3,5,3,3,6,2,7,6,6,1,4,2,8,2,8,0,6,4,4,4,4,8,6,6,4,5,2,3,8,7,4,9,
    3,0,3,5,8,9,0,7,2,9,6,2,9,0,4,9,1,5,6,0,4,4,0,7,7,2,3,9,0,7,1,3,8,1,0,5,1,5,8,5,9,3,0,7,9,6,0,8,6,6,
    7,0,1,7,2,4,2,7,1,2,1,8,8,3,9,9,8,7,9,7,9,0,8,7,9,2,2,7,4,9,2,1,9,0,1,6,9,9,7,2,0,8,8,8,0,9,3,7,7,6,
    6,5,7,2,7,3,3,3,0,0,1,0,5,3,3,6,7,8,8,1,2,2,0,2,3,5,4,2,1,8,0,9,7,5,1,2,5,4,5,4,0,5,9,4,7,5,2,2,4,3,
    5,2,5,8,4,9,0,7,7,1,1,6,7,0,5,5,6,0,1,3,6,0,4,8,3,9,5,8,6,4,4,6,7,0,6,3,2,4,4,1,5,7,2,2,1,5,5,3,9,7,
    5,3,6,9,7,8,1,7,9,7,7,8,4,6,1,7,4,0,6,4,9,5,5,1,4,9,2,9,0,8,6,2,5,6,9,3,2,1,9,7,8,4,6,8,6,2,2,4,8,2,
    8,3,9,7,2,2,4,1,3,7,5,6,5,7,0,5,6,0,5,7,4,9,0,2,6,1,4,0,7,9,7,2,9,6,8,6,5,2,4,1,4,5,3,5,1,0,0,4,7,4,
    8,2,1,6,6,3,7,0,4,8,4,4,0,3,1,9,9,8,9,0,0,0,8,8,9,5,2,4,3,4,5,0,6,5,8,5,4,1,2,2,7,5,8,8,6,6,6,8,8,1,
    1,6,4,2,7,1,7,1,4,7,9,9,2,4,4,4,2,9,2,8,2,3,0,8,6,3,4,6,5,6,7,4,8,1,3,9,1,9,1,2,3,1,6,2,8,2,4,5,8,6,
    1,7,8,6,6,4,5,8,3,5,9,1,2,4,5,6,6,5,2,9,4,7,6,5,4,5,6,8,2,8,4,8,9,1,2,8,8,3,1,4,2,6,0,7,6,9,0,0,4,2,
    2,4,2,1,9,0,2,2,6,7,1,0,5,5,6,2,6,3,2,1,1,1,1,1,0,9,3,7,0,5,4,4,2,1,7,5,0,6,9,4,1,6,5,8,9,6,0,4,0,8,
    0,7,1,9,8,4,0,3,8,5,0,9,6,2,4,5,5,4,4,4,3,6,2,9,8,1,2,3,0,9,8,7,8,7,9,9,2,7,2,4,4,2,8,4,9,0,9,1,8,8,
    8,4,5,8,0,1,5,6,1,6,6,0,9,7,9,1,9,1,3,3,8,7,5,4,9,9,2,0,0,5,2,4,0,6,3,6,8,9,9,1,2,5,6,0,7,1,7,6,0,6,
    0,5,8,8,6,1,1,6,4,6,7,1,0,9,4,0,5,0,7,7,5,4,1,0,0,2,2,5,6,9,8,3,1,5,5,2,0,0,0,5,5,9,3,5,7,2,9,7,2,5,
    7,1,6,3,6,2,6,9,5,6,1,8,8,2,6,7,0,4,2,8,2,5,2,4,8,3,6,0,0,8,2,3,2,5,7,5,3,0,4,2,0,7,5,2,9,6,3,4,5,0,
-99};
