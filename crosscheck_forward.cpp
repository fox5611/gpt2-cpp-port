#include <iostream>
#include <iomanip>
#include <fstream>
#include <vector>
#include "tensor.h"
#include "loader.h"
#include "paths.h"
#include "ops.h"

int main() {
    auto W = load_weights(default_weights_path());
    std::vector<int> token_ids = {464, 3139, 286, 4881, 318};

    Tensor logits = gpt2_forward(token_ids, W);

    std::ofstream f("dbg_logits.txt");
    f << std::fixed << std::setprecision(6);
    for (int i = 0; i < logits.shape[0]; i++)
        for (int j = 0; j < logits.shape[1]; j++) f << logits.at(i, j) << "\n";

    int best = 0;
    for (int v = 1; v < logits.shape[1]; v++)
        if (logits.at(logits.shape[0] - 1, v) > logits.at(logits.shape[0] - 1, best)) best = v;

    std::cout << "shape (" << logits.shape[0] << ", " << logits.shape[1] << ")\n";
    std::cout << "argmax of last position: " << best << "\n";
    return 0;
}
