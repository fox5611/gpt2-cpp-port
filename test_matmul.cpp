#include <iostream>
#include "ops.h"

int main() {
    // a = [[1,2],[3,4]]   b = [[5,6],[7,8]]
    Tensor a({2,2});
    a.at(0,0)=1; a.at(0,1)=2; a.at(1,0)=3; a.at(1,1)=4;
    Tensor b({2,2});
    b.at(0,0)=5; b.at(0,1)=6; b.at(1,0)=7; b.at(1,1)=8;

    Tensor c = matmul(a, b);
    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 2; j++) std::cout << c.at(i,j) << " ";
        std::cout << std::endl;
    }
    return 0;
}
