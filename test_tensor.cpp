#include <iostream>
#include "tensor.h"

int main() {
    Tensor t({10, 5, 8});

    std::cout << "size: " << t.size() << std::endl;

    t.at(1, 3, 2) = 42.0f;
    std::cout << "at(1,3,2): " << t.at(1, 3, 2) << std::endl;
    std::cout << "flat index should be 66, data[66] = " << t.data[66] << std::endl;

    return 0;
}

