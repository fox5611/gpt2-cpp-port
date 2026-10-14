#include <iostream>
#include "tensor.h"
#include "loader.h"
#include "paths.h"

int main() {
    auto weights = load_weights(default_weights_path());

    for (auto& kv : weights) {
        if (kv.first.find("ln_f") != std::string::npos &&
            kv.first.find("transformer") != std::string::npos) {
            std::cout << kv.first << "  shape: (";
            for (size_t d = 0; d < kv.second.shape.size(); d++)
                std::cout << kv.second.shape[d] << (d + 1 < kv.second.shape.size() ? ", " : "");
            std::cout << ")\n";
        }
    }
    return 0;
}
