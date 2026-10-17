#include <iostream>
#include <chrono>
#include <vector>
#include "tensor.h"
#include "loader.h"
#include "paths.h"
#include "ops.h"

int main() {
    auto W = load_weights(default_weights_path());
    std::vector<int> prompt = {464, 3139, 286, 4881, 318};

    Tensor warm = gpt2_forward(prompt, W);   // discard: first pass pays page faults

    for (int run = 0; run < 3; run++) {
        auto t0 = std::chrono::steady_clock::now();
        Tensor logits = gpt2_forward(prompt, W);
        auto t1 = std::chrono::steady_clock::now();

        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

        int last = logits.shape[0] - 1, best = 0;
        for (int v = 1; v < logits.shape[1]; v++)
            if (logits.at(last, v) > logits.at(last, best)) best = v;

        std::cout << "run " << run << ": " << ms << " ms, argmax " << best << "\n";
    }
    return 0;
}
