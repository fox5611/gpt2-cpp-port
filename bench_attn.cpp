#include <iostream>
#include <chrono>
#include <functional>
#include "tensor.h"
#include "ops.h"

static double time_ms(int reps, const std::function<void()>& f) {
    auto t0 = std::chrono::steady_clock::now();
    for (int r = 0; r < reps; r++) f();
    auto t1 = std::chrono::steady_clock::now();
    return std::chrono::duration<double, std::milli>(t1 - t0).count() / reps;
}

int main() {
    const int n = 256, n_heads = 12, head_dim = 64;

    Tensor qh({n_heads, n, head_dim}), kh({n_heads, n, head_dim}), vh({n_heads, n, head_dim});
    for (int i = 0; i < qh.size(); i++) {
        qh.data[i] = 0.01f * ((i % 61) - 30);
        kh.data[i] = 0.01f * ((i % 71) - 35);
        vh.data[i] = 0.01f * ((i % 53) - 26);
    }

    // built once, outside every timed region
    Tensor s0 = attention_scores(qh, kh, 0);
    Tensor w0 = softmax(s0);

    double t_scores = time_ms(5, [&]{
        for (int h = 0; h < n_heads; h++) { Tensor s = attention_scores(qh, kh, h); (void)s; }
    });
    double t_soft = time_ms(5, [&]{
        for (int h = 0; h < n_heads; h++) { Tensor w = softmax(s0); (void)w; }
    });
    double t_gather = time_ms(5, [&]{
        for (int h = 0; h < n_heads; h++) { Tensor o = attention_gather(w0, vh, h); (void)o; }
    });

    // both kernels now touch the lower triangle only
    const double pairs = n * (n + 1) / 2.0;
    const double flop  = 2.0 * n_heads * pairs * head_dim;

    std::cout << "n=" << n << ", one block, all 12 heads\n\n";
    std::cout << std::fixed;
    std::cout.precision(3);
    std::cout << "  attention_scores  " << t_scores << " ms   "
              << flop / (t_scores * 1e6) << " GFLOPS\n";
    std::cout << "  softmax           " << t_soft   << " ms\n";
    std::cout << "  attention_gather  " << t_gather << " ms   "
              << flop / (t_gather * 1e6) << " GFLOPS\n";
    double tot = t_scores + t_soft + t_gather;
    std::cout << "  total             " << tot << " ms   x12 blocks = " << tot * 12 << " ms\n";
    return 0;
}
