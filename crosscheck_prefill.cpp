#include <iostream>
#include <algorithm>
#include <cmath>
#include <vector>
#include "tensor.h"
#include "loader.h"
#include "paths.h"
#include "ops.h"

static float max_abs_diff(Tensor& a, Tensor& b, int rows, int cols) {
    float worst = 0.0f;
    for (int i = 0; i < rows; i++)
        for (int j = 0; j < cols; j++)
            worst = std::max(worst, std::fabs(a.at(i, j) - b.at(i, j)));
    return worst;
}

int main() {
    auto W = load_weights(default_weights_path());
    int n_layers = 12;
    int d_model  = W.at("transformer.wte.weight").shape[1];

    std::vector<int> prompt = {464, 3139, 286, 4881, 318, 257, 1588, 1748};
    int n = (int)prompt.size();

    // A: the path being replaced, one token at a time
    KVCache ca(n_layers, n + 1, d_model);
    Tensor la;
    for (size_t t = 0; t < prompt.size(); t++)
        la = gpt2_forward_step(prompt[t], ca, W, n_layers, t + 1 == prompt.size());

    // B: the whole prompt in one pass
    KVCache cb(n_layers, n + 1, d_model);
    Tensor lb = gpt2_forward_prefill(prompt, cb, W, n_layers);

    std::cout << "n_used        a=" << ca.n_used << "  b=" << cb.n_used << "\n";

    float worst_k = 0.0f, worst_v = 0.0f;
    for (int i = 0; i < n_layers; i++) {
        worst_k = std::max(worst_k, max_abs_diff(ca.k[i], cb.k[i], n, d_model));
        worst_v = std::max(worst_v, max_abs_diff(ca.v[i], cb.v[i], n, d_model));
    }
    std::cout << "max |dK|      " << worst_k << "\n";
    std::cout << "max |dV|      " << worst_v << "\n";
    std::cout << "max |dlogit|  " << max_abs_diff(la, lb, 1, la.shape[1]) << "\n";

    int best_a = 0, best_b = 0;
    for (int v = 1; v < la.shape[1]; v++) {
        if (la.at(0, v) > la.at(0, best_a)) best_a = v;
        if (lb.at(0, v) > lb.at(0, best_b)) best_b = v;
    }
    std::cout << "argmax        a=" << best_a << "  b=" << best_b
              << (best_a == best_b ? "   MATCH\n" : "   MISMATCH\n");
    return 0;
}
