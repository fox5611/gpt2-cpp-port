#include <iostream>
#include "loader.h"

int main() {
    auto w = load_weights("../gpt2-from-scratch/gpt2_weights.bin");
    std::cout << "tensors: " << w.size() << std::endl;
    Tensor& wte = w["transformer.wte.weight"];
    std::cout << "wte shape: " << wte.shape[0] << " x " << wte.shape[1] << std::endl;
    std::cout << "wte[0][0]: " << wte.at(0, 0) << std::endl;
    return 0;
}
