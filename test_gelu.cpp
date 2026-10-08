#include <iostream>
#include "ops.h"

int main() {
    Tensor x({1, 5});
    x.data = {-5.0f, -1.0f, 0.0f, 1.0f, 5.0f};
    Tensor y = gelu(x);
    for (int i = 0; i < 5; i++)
        std::cout << x.data[i] << " -> " << y.data[i] << std::endl;
    return 0;
}
