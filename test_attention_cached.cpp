#include <iostream>
#include <cmath>
#include <vector>
#include "tensor.h"
#include "loader.h"
#include "paths.h"
#include "ops.h"

// Replays a sequence one token at a time through attention_cached and compares
// each step against the corresponding row of the uncached attention() over the
// whole sequence. Reports a per-position diff, so an indexing bug says WHICH
// position is wrong instead of only that something is.
int main() {
    auto W = load_weights(default_weights_path());

    Tensor& c_attn_w = W.at("transformer.h.0.attn.c_attn.weight");
    Tensor& c_attn_b = W.at("transformer.h.0.attn.c_attn.bias");
    Tensor& c_proj_w = W.at("transformer.h.0.attn.c_proj.weight");
    Tensor& c_proj_b = W.at("transformer.h.0.attn.c_proj.bias");

    const int d_model    = c_attn_b.size() / 3;   // 2304 / 3
    const int n          = 8;
    const int max_tokens = 16;

    if (d_model != 768) {
        std::cout << "unexpected d_model " << d_model << "\n";
        return 1;
    }

    // deterministic pseudo-random input, identical on every run
    Tensor x({n, d_model});
    unsigned int s = 12345u;
    for (int i = 0; i < x.size(); i++) {
        s = s * 1664525u + 1013904223u;
        x.data[i] = ((float)(s >> 8) / (float)(1u << 24) - 0.5f) * 0.5f;
    }

    Tensor full = attention(x, c_attn_w, c_attn_b, c_proj_w, c_proj_b);

    Tensor ck({max_tokens, d_model});
    Tensor cv({max_tokens, d_model});

    bool  ok      = true;
    float worst   = 0.0f;
    int   worst_t = -1;

    for (int t = 0; t < n; t++) {
        Tensor x1({1, d_model});
        for (int j = 0; j < d_model; j++) x1.at(0, j) = x.at(t, j);

        Tensor out1 = attention_cached(x1, ck, cv, t,
                                       c_attn_w, c_attn_b, c_proj_w, c_proj_b);

        float md = 0.0f;
        for (int j = 0; j < d_model; j++) {
            float d = std::fabs(out1.at(0, j) - full.at(t, j));
            if (d > md) md = d;
        }

        std::cout << "  position " << t << ": max abs diff " << md
                  << (md > 1e-4f ? "   <-- MISMATCH" : "") << "\n";

        if (md > worst) { worst = md; worst_t = t; }
        if (md > 1e-4f) ok = false;
    }

    std::cout << "worst position " << worst_t << " at " << worst << "\n";
    std::cout << (ok ? "PASS\n" : "FAILED\n");
    return ok ? 0 : 1;
}
