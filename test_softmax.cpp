#include <iostream>
#include "ops.h"

int main() {
    Tensor x({2, 3});
    x.data = {1.0f, 1.0f, 1.0f,      // equal scores
              10.0f, 1.0f, 1.0f};     // one dominant

    Tensor y = softmax(x);
    for (int i = 0; i < 2; i++) {
        float sum = 0.0f;
        for (int j = 0; j < 3; j++) {
            std::cout << y.at(i, j) << " ";
            sum += y.at(i, j);
        }
        std::cout << "  (sum: " << sum << ")" << std::endl;
    }
    return 0;
}
