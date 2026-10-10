#include <iostream>
#include "ops.h"

int main() {
    Tensor x({2, 3});
    x.data = {1,2,3, 4,5,6};
    Tensor y = transpose(x);
    std::cout << "shape: " << y.shape[0] << " x " << y.shape[1] << std::endl;
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 2; j++) std::cout << y.at(i,j) << " ";
        std::cout << std::endl;
    }
    return 0;
}
