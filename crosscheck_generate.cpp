#include <iostream>
#include <vector>
#include "tensor.h"
#include "loader.h"
#include "paths.h"
#include "ops.h"

int main() {
    auto W = load_weights(default_weights_path());
    std::vector<int> prompt = {464, 3139, 286, 4881, 318};

    std::vector<int> out = generate(prompt, W, 10);

    for (size_t i = 0; i < out.size(); i++)
        std::cout << out[i] << (i + 1 < out.size() ? " " : "\n");
    return 0;
}
