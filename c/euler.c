// Project Euler <http://www.projecteuler.net> solutions in C.
//
// :author: Robert David Grant <robert.david.grant@gmail.com>
//
// :copyright:
//   Copyright 2012 Robert David Grant
//
//   Licensed under the Apache License, Version 2.0 (the "License"); you
//   may not use this file except in compliance with the License.  You
//   may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
//   Unless required by applicable law or agreed to in writing, software
//   distributed under the License is distributed on an "AS IS" BASIS,
//   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
//   implied.  See the License for the specific language governing
//   permissions and limitations under the License.


#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <math.h>
#include <assert.h>
#include <string.h>


/*****************************************************************************/
/* Global                                                                    */
/*****************************************************************************/

const bool TEST = false;
const bool DEBUG = false;
int problem8_data[];


/*****************************************************************************/
/* Declarations                                                              */
/*****************************************************************************/

bool is_prime(unsigned long composite);
void test_is_prime();

int max_nprimes_below(unsigned long n);
unsigned long nth_prime_upper_bound(int n);

unsigned long nth_prime(int n);
void test_nth_prime();

int prime_sieve(unsigned long n, unsigned long * primes);
void test_prime_sieve();

bool is_palindrome(const unsigned long num);
void test_is_palindrome();

int word_value(const char * word);
void test_word_value();

int           problem1();
int           problem2();
unsigned long problem3();
unsigned long problem4();
int           problem5();
int           problem6();
unsigned long problem7();
int           problem8();
int           problem9();
unsigned long problem10();
unsigned long problem22();

void test_all();

/*****************************************************************************/
/* Utility functions                                                         */
/*****************************************************************************/

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
    printf("Testing is_prime...\n");
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
    printf("Testing completed.\n");
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
    if (n < 6) 
        assert(false);
    n += 1; // this formula uses 1-based indexing
    return ceil(n*(log(n) + log(log(n))));
}


// Fills array 'primes' with of primes < n, terminates with a 0
// Returns nprimes, or -1 on error
// Uses Sieve of Eratosthenes
int prime_sieve(unsigned long n, unsigned long * primes) {

    // check inputs
    if (primes == NULL || n <= 0)
        return -1;

    // indicates if pmask[i] is prime
    bool * pmask = malloc(n*sizeof(bool));

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
    printf("Testing prime number sieve...\n");
    const int N = 44;
    unsigned long known_primes[] = {2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37,
        41, 43, 0}; // 0 terminated
    unsigned long nprimes = 0;
    unsigned long * primes = malloc(sizeof(unsigned long) *
            (max_nprimes_below(N) + 1));

    nprimes = prime_sieve(N, primes);
    assert(nprimes == 14);

    for (int i=0; primes[i]; ++i) {
        if (DEBUG)
            printf("%ld == %ld\n", primes[i], known_primes[i]);
        assert(primes[i] == known_primes[i]);
    }

    free(primes);
    printf("Testing completed.\n");
}


unsigned long nth_prime(int n) {
    unsigned long answer = 0;
    unsigned long ulimit = -1;
    unsigned long * primes;

    assert(n >=0);

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
    printf("Testing nth_prime...\n");
    unsigned long known_primes[] = {2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37,
        41, 43, 0}; // 0 terminated
    for (int i=0; known_primes[i]; ++i) {
        //printf("%d: %ld\n", i, nth_prime(i));
        assert(nth_prime(i) == known_primes[i]);
    }
    printf("Testing completed.\n");
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
    printf("Testing is_palindrome...\n");
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
    printf("Testing completed.\n");
}


// Find value of a string (sum of values of its letters)
// Value of a letter is its 1-based position in the alphabet:
// A=1, B=2, etc.
int word_value(const char * word) {
    int sum = 0;
    if (word == NULL)
        return -1;

    for (int i=0; word[i]; ++i) {
        sum += (int)word[i] - 64;
    }
    return sum;
}

void test_word_value() {
    printf("Testing word_value...\n");
    char str[2];
    int i=1;
    int val=0;
    for (char c='A'; c <='Z'; ++c) {
        sprintf(str, "%c", c);
        val = word_value(str);
        if (DEBUG) printf("%d\n", val);
        assert(val == i++);
    }
    assert(word_value("COLIN") == 53);
    printf("Testing completed.\n");
}


/*****************************************************************************/
/* Problems                                                                  */
/*****************************************************************************/


// Find the sum of all the multiple of 3 or 5 below 1000.
int problem1() {
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
int problem2() {
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
unsigned long problem3() {
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


// Find the largest palindrome made from the product of two 3-digit numbers.
unsigned long problem4() {
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
int problem5() {
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
int problem6() {
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
unsigned long problem7() {
    const int PRIME_INDEX = 10000;
    return nth_prime(PRIME_INDEX);
}


// Find the greatest product of five consecutive digits in the
// 1000-digit number (problem8_data).
int problem8() {
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
int problem9() {
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
unsigned long problem10() {
    const unsigned long ULIMIT = 2000000;
    unsigned long * primes = malloc(sizeof(unsigned long) *
            (max_nprimes_below(ULIMIT) + 1));
    unsigned long sum = 0;
    prime_sieve(ULIMIT, primes);
    for (int i=0; primes[i]; ++i) {
        sum += primes[i];
    }
    free(primes);
    return sum;
}


// What is the total of all the name scores in the file?
// A name score is calculated as:
// (alphanumeric_word_value * position_in_sorted_wordlist)
unsigned long problem22() {
    const unsigned int NCHARS = 46448; // counted in vim
    const unsigned int NWORDS = 5163;  // counted in vim
    const unsigned int MAX_WORDLENGTH = 20;
    FILE *fh;
    char str[NCHARS], *word;
    char words[NWORDS][MAX_WORDLENGTH];
    int words_ix = 0;

    if ((fh = fopen("names.txt", "r"))) {

        // read the file
        fgets(str, NCHARS*sizeof(char), fh); // file is only one line
        fclose(fh);

        // tokenize, parse, and put words into an array
        word = strtok(str, ",");
        do {
            word += 1; // chop off leading "
            word[strlen(word)-1] = '\0'; //chop off tailing "
            strncpy(words[words_ix++], word, MAX_WORDLENGTH);
        } while ((word = strtok(NULL, ",")));

        // sort the list
        qsort(words, NWORDS, MAX_WORDLENGTH, (int(*)(const void*, const void*))strcmp);

        // calculate and sum the word scores
        unsigned long total = 0;
        for (int i=0; i < NWORDS; ++i) {
            total += (i+1) * word_value(words[i]);
        }

        return total;
    }
    return -1; // couldn't read the file
}

int main() {
    if (TEST)
        test_all();

    printf("Problem 1:  %d\n",  problem1());
    printf("Problem 2:  %d\n",  problem2());
    printf("Problem 3:  %lu\n", problem3());
    printf("Problem 4:  %lu\n", problem4());
    printf("Problem 5:  %d\n",  problem5());
    printf("Problem 6:  %d\n",  problem6());
    printf("Problem 7:  %lu\n", problem7());
    printf("Problem 8:  %d\n",  problem8());
    printf("Problem 9:  %d\n",  problem9());
    printf("Problem 10: %lu\n", problem10());
    printf("Problem 22: %lu\n", problem22());

    exit(EXIT_SUCCESS);
}

void test_all() {
    test_is_prime();
    test_nth_prime();
    test_prime_sieve();
    test_is_palindrome();
    test_word_value();
}

/*****************************************************************************/
/* Data                                                                      */
/*****************************************************************************/

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
