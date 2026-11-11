#include <iostream>
#include <chrono>
#include <functional>
#include <vector>
#include <string>
#include "tensor.h"
#include "loader.h"
#include "paths.h"
#include "ops.h"

// Times each piece of one transformer block in isolation at the real prefill
// shapes (n=256), so the per-flop efficiency of the matmuls and of the n^2
// attention work can be compared directly. Caveat: isolated timing sees a
// different cache state than the same op inside a full pass, so treat these as
// a ratio between the pieces rather than an exact budget.
static double time_ms(int reps, const std::function<void()>& f) {
    auto t0 = std::chrono::steady_clock::now();
    for (int r = 0; r < reps; r++) f();
    auto t1 = std::chrono::steady_clock::now();
    return std::chrono::duration<double, std::milli>(t1 - t0).count() / reps;
}

int main() {
    auto W = load_weights(default_weights_path());
    const int n = 256, d_model = 768, n_heads = 12, head_dim = 64;

    std::string p = "transformer.h.0.";
    Tensor& c_attn_w = W.at(p + "attn.c_attn.weight");
    Tensor& c_proj_w = W.at(p + "attn.c_proj.weight");
    Tensor& c_fc_w   = W.at(p + "mlp.c_fc.weight");
    Tensor& mlp_cp_w = W.at(p + "mlp.c_proj.weight");
    Tensor& ln_w     = W.at(p + "ln_1.weight");
    Tensor& ln_b     = W.at(p + "ln_1.bias");

    // Values do not matter for timing, only shapes. Deterministic filler.
    Tensor x({n, d_model});
    for (int i = 0; i < x.size(); i++) x.data[i] = 0.01f * ((i % 97) - 48);

    Tensor x3072({n, 3072});
    for (int i = 0; i < x3072.size(); i++) x3072.data[i] = 0.01f * ((i % 89) - 44);

    Tensor qh({n_heads, n, head_dim}), kh({n_heads, n, head_dim}), vh({n_heads, n, head_dim});
    for (int i = 0; i < qh.size(); i++) {
        qh.data[i] = 0.01f * ((i % 61) - 30);
        kh.data[i] = 0.01f * ((i % 71) - 35);
        vh.data[i] = 0.01f * ((i % 53) - 26);
    }

    std::cout << "n=" << n << ", one transformer block, ms\n\n";
    std::cout.precision(2);
    std::cout << std::fixed;

    double t, mm = 0.0;

    t = time_ms(3, [&]{ Tensor r = matmul(x, c_attn_w);     (void)r; });
    std::cout << "  matmul qkv       768 -> 2304    " << t << "\n"; mm += t;

    t = time_ms(3, [&]{ Tensor r = matmul(x, c_proj_w);     (void)r; });
    std::cout << "  matmul attn_out  768 -> 768     " << t << "\n"; mm += t;

    t = time_ms(3, [&]{ Tensor r = matmul(x, c_fc_w);       (void)r; });
    std::cout << "  matmul mlp_fc    768 -> 3072    " << t << "\n"; mm += t;

    t = time_ms(3, [&]{ Tensor r = matmul(x3072, mlp_cp_w); (void)r; });
    std::cout << "  matmul mlp_proj  3072 -> 768    " << t << "\n"; mm += t;

    double attn = time_ms(3, [&]{
        for (int h = 0; h < n_heads; h++) {
            Tensor s = attention_scores(qh, kh, h);
            Tensor w = softmax(s);
            Tensor o = attention_gather(w, vh, h);
            (void)o;
        }
    });
    std::cout << "  attention n^2    12 heads       " << attn << "\n";

    double ln = time_ms(3, [&]{ Tensor r = layernorm(x, ln_w, ln_b); (void)r; });
    std::cout << "  layernorm                       " << ln << "\n\n";

    std::cout << "  matmuls    " << mm   << " ms/block, x12 = " << mm * 12   << " ms\n";
    std::cout << "  attention  " << attn << " ms/block, x12 = " << attn * 12 << " ms\n";
    std::cout << "  full prefill at n=256 measured  2292 ms\n\n";

    const double mm_flop   = 2.0 * n * (768.0*2304 + 768.0*768 + 768.0*3072 + 3072.0*768);
    const double attn_flop = 2.0 * n_heads * 2.0 * n * n * head_dim * 0.5;
    std::cout << "  matmul    " << mm_flop   / (mm   * 1e6) << " GFLOPS\n";
    std::cout << "  attention " << attn_flop / (attn * 1e6) << " GFLOPS\n";
    return 0;
}
