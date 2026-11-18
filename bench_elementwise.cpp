// bench_elementwise.cpp
#include "ops.h"
#include <chrono>
#include <cstdio>
#include <vector>

static double ms_now() {
    using namespace std::chrono;
    return duration<double, std::milli>(steady_clock::now().time_since_epoch()).count();
}

template <typename F>
static double time_best(int reps, F f) {
    double best = 1e18;
    for (int r = 0; r < reps; r++) {
        double t0 = ms_now();
        f();
        double t1 = ms_now();
        if (t1 - t0 < best) best = t1 - t0;
    }
    return best;
}

static Tensor filled(std::vector<int> shape, float base) {
    Tensor t(shape);
    for (int i = 0; i < t.size(); i++) {
        t.data[i] = base + 0.001f * (float)(i % 1000) - 0.5f;
    }
    return t;
}

int main() {
    const int n = 256, d = 768, dff = 3072, dqkv = 2304;
    const int n_heads = 12, n_blocks = 12;

    Tensor x768   = filled({n, d}, 0.0f);
    Tensor y768   = filled({n, d}, 0.1f);
    Tensor x3072  = filled({n, dff}, 0.0f);
    Tensor x2304  = filled({n, dqkv}, 0.0f);
    Tensor b768   = filled({1, d}, 0.0f);
    Tensor b3072  = filled({1, dff}, 0.0f);
    Tensor b2304  = filled({1, dqkv}, 0.0f);
    Tensor lnw    = filled({1, d}, 1.0f);
    Tensor lnb    = filled({1, d}, 0.0f);
    Tensor scores = filled({n, n}, 0.0f);

    volatile float sink = 0.0f;

    double t_gelu = time_best(5, [&]{ Tensor o = gelu(x3072);               sink += o.data[0]; });
    double t_ab23 = time_best(5, [&]{ Tensor o = add_bias(x2304, b2304);    sink += o.data[0]; });
    double t_ab30 = time_best(5, [&]{ Tensor o = add_bias(x3072, b3072);    sink += o.data[0]; });
    double t_ab07 = time_best(5, [&]{ Tensor o = add_bias(x768,  b768);     sink += o.data[0]; });
    double t_add  = time_best(5, [&]{ Tensor o = add(x768, y768);           sink += o.data[0]; });
    double t_ln   = time_best(5, [&]{ Tensor o = layernorm(x768, lnw, lnb); sink += o.data[0]; });
    double t_sm   = time_best(5, [&]{ Tensor o = softmax(scores);           sink += o.data[0]; });

    printf("one call each, best of 5, ms\n\n");
    printf("  gelu       (%d,%d)  %8.3f\n", n, dff,  t_gelu);
    printf("  add_bias   (%d,%d)  %8.3f\n", n, dqkv, t_ab23);
    printf("  add_bias   (%d,%d)  %8.3f\n", n, dff,  t_ab30);
    printf("  add_bias   (%d,%d)   %8.3f\n", n, d,   t_ab07);
    printf("  add        (%d,%d)   %8.3f\n", n, d,   t_add);
    printf("  layernorm  (%d,%d)   %8.3f\n", n, d,   t_ln);
    printf("  softmax    (%d,%d)   %8.3f\n", n, n,   t_sm);

    // counts per transformer_block_prefill, from ops.h:
    //   layernorm x2, add_bias(2304) x1, add_bias(768) x2, add_bias(3072) x1,
    //   gelu x1, add x2, softmax x n_heads  <- verify the softmax count
    double per_block = 2.0 * t_ln + t_ab23 + 2.0 * t_ab07 + t_ab30
                     + t_gelu + 2.0 * t_add + (double) n_heads * t_sm;

    printf("\n  per block %8.3f ms, x%d = %8.3f ms\n", per_block, n_blocks, per_block * n_blocks);
    printf("  measured prefill total 381 ms, matmuls 98, attention 112\n");
    printf("  sink %g\n", (float) sink);
    return 0;
}

