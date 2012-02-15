#include <stdio.h>
#include <stdbool.h>

int problem1(void);
int problem2(void);
int problem5(void);
int problem6(void);

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
    printf("%d\n", problem5());
    return(0);
}
