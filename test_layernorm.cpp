#include <iostream>
#include <cmath>
#include "ops.h"

int main() {
    Tensor x({2, 4});
    x.data = {1.0f, 2.0f, 3.0f, 4.0f,
              10.0f, 20.0f, 30.0f, 40.0f};

    Tensor w({4});
    w.data = {1.0f, 1.0f, 1.0f, 1.0f};   // identity scale
    Tensor b({4});
    b.data = {0.0f, 0.0f, 0.0f, 0.0f};   // no shift

    Tensor y = layernorm(x, w, b);

    for (int i = 0; i < 2; i++) {
        float mean = 0.0f;
        for (int j = 0; j < 4; j++) mean += y.at(i, j);
        mean /= 4;

        float var = 0.0f;
        for (int j = 0; j < 4; j++) {
            float d = y.at(i, j) - mean;
            var += d * d;
        }
        var /= 4;

        std::cout << "row " << i << ": ";
        for (int j = 0; j < 4; j++) std::cout << y.at(i, j) << " ";
        std::cout << " | mean=" << mean << " var=" << var << std::endl;
    }
    return 0;
}
