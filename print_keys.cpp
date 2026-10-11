#include <iostream>
#include "tensor.h"
#include "loader.h"

int main() {
    auto weights = load_weights("/home/rthacker/ai/gpt2-from-scratch/gpt2_weights.bin");

    for (auto& kv : weights) {
        if (kv.first.find("attn") != std::string::npos &&
            kv.first.find("h.0.") != std::string::npos) {
            std::cout << kv.first << "  shape: (";
            for (size_t d = 0; d < kv.second.shape.size(); d++)
                std::cout << kv.second.shape[d] << (d + 1 < kv.second.shape.size() ? ", " : "");
            std::cout << ")\n";
        }
    }
    return 0;
}
