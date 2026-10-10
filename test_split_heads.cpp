#include <iostream>
#include "ops.h"

int main() {
    Tensor x({2, 768});
    for (int i = 0; i < x.size(); i++) x.data[i] = i;   // 0,1,2,...

    Tensor h = split_heads(x);
    std::cout << "shape: " << h.shape[0] << " x " << h.shape[1] << " x " << h.shape[2] << std::endl;

    // head 0, token 0, dim 0  should be x.at(0, 0) = 0
    std::cout << "h(0,0,0) = " << h.at(0,0,0) << "  expect 0" << std::endl;
    // head 1, token 0, dim 0  should be x.at(0, 64) = 64
    std::cout << "h(1,0,0) = " << h.at(1,0,0) << "  expect 64" << std::endl;
    // head 3, token 0, dim 5  should be x.at(0, 197) = 197
    std::cout << "h(3,0,5) = " << h.at(3,0,5) << "  expect 197" << std::endl;
    // head 0, token 1, dim 0  should be x.at(1, 0) = 768
    std::cout << "h(0,1,0) = " << h.at(0,1,0) << "  expect 768" << std::endl;
    return 0;
}
