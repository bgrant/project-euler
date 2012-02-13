#include <stdio.h>

int problem1(void);
int problem2(void);

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
// million
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

int main(int argc, char** argv) {
    printf("%d\n", problem2());
    return(0);
}
