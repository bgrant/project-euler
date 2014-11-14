#include <iostream>

int problem1(int n){
    int sum = 0;
    for (int i=1; i<n; ++i){
        if ((i % 3 == 0) || (i % 5 == 0)) {
            sum += i;
        }
    }
    return sum;
}

int main(void){

    std::cout << "Problem 1: " << problem1(1000) << std::endl;

    return 0;
}
