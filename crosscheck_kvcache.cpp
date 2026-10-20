#include <iostream>
#include <cmath>
#include <vector>
#include "tensor.h"
#include "loader.h"
#include "paths.h"
#include "ops.h"

// Two checks.
//
// 1. Position localization: step a prompt through the cache one token at a
//    time and compare each step's logits against gpt2_forward over the prefix
//    ending at that token. A wpe off-by-one shows up as a large diff at a
//    specific position, not as a vague global failure.
// 2. End to end: generate() vs generate_cached() must agree token for token.
int main() {
    auto W = load_weights(default_weights_path());

    std::vector<int> prompt = {464, 3139, 286, 4881, 318};
    const int n_new    = 10;
    const int n_layers = 12;

    Tensor& wte = W.at("transformer.wte.weight");
    int d_model = wte.shape[1];

    std::cout << "--- per-position logits, cached step vs full forward ---\n";

    KVCache cache(n_layers, (int)prompt.size() + n_new, d_model);
    bool ok = true;

    for (size_t t = 0; t < prompt.size(); t++) {
        Tensor step = gpt2_forward_step(prompt[t], cache, W, n_layers, true);

        std::vector<int> prefix(prompt.begin(), prompt.begin() + t + 1);
        Tensor full = gpt2_forward(prefix, W, n_layers, /*last_row_only=*/true);

        float md = 0.0f;
        for (int v = 0; v < full.shape[1]; v++) {
            float d = std::fabs(step.at(0, v) - full.at(0, v));
            if (d > md) md = d;
        }

        int bs = 0, bf = 0;
        for (int v = 1; v < full.shape[1]; v++) {
            if (step.at(0, v) > step.at(0, bs)) bs = v;
            if (full.at(0, v) > full.at(0, bf)) bf = v;
        }

        std::cout << "  position " << t << ": max abs diff " << md
                  << ", argmax cached " << bs << " vs full " << bf
                  << (md > 1e-2f || bs != bf ? "   <-- MISMATCH" : "") << "\n";

        if (md > 1e-2f || bs != bf) ok = false;
    }

    std::cout << "\n--- end to end ---\n";

    std::vector<int> a = generate(prompt, W, n_new);
    std::vector<int> b = generate_cached(prompt, W, n_new);

    std::cout << "  uncached:";
    for (int id : a) std::cout << " " << id;
    std::cout << "\n  cached  :";
    for (int id : b) std::cout << " " << id;
    std::cout << "\n";

    if (a.size() != b.size()) {
        std::cout << "  length mismatch: " << a.size() << " vs " << b.size() << "\n";
        ok = false;
    } else {
        int first = -1;
        for (size_t i = 0; i < a.size(); i++)
            if (a[i] != b[i]) { first = (int)i; break; }
        if (first >= 0) {
            std::cout << "  first divergence at index " << first
                      << ": " << a[first] << " vs " << b[first] << "\n";
            ok = false;
        } else {
            std::cout << "  identical\n";
        }
    }

    std::cout << (ok ? "\nPASS\n" : "\nFAILED\n");
    return ok ? 0 : 1;
}
