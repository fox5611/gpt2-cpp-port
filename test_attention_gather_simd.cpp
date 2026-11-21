// test_attention_gather_simd.cpp
#include "ops.h"
#include <cstdio>
#include <cmath>

int main() {
    const int cases[][3] = {{1, 64, 1}, {5, 64, 3}, {17, 64, 12}, {256, 64, 12}, {7, 24, 2}};
    double worst = 0.0;

    for (auto & c : cases) {
        int n = c[0], head_dim = c[1], n_heads = c[2];
        Tensor weights({n, n});
        Tensor v({n_heads, n, head_dim});
        for (int i = 0; i < weights.size(); i++) {
            weights.data[i] = 0.001f * (float) (i % 719);
        }
        for (int i = 0; i < v.size(); i++) {
            v.data[i] = 0.01f * (float) ((i * 13) % 251) - 1.0f;
        }

        for (int h = 0; h < n_heads; h++) {
            Tensor a = attention_gather_scalar(weights, v, h);
            Tensor b = attention_gather(weights, v, h);
            if (a.size() != b.size()) { printf("FAIL size\n"); return 1; }
            for (int k = 0; k < a.size(); k++) {
                double d = std::fabs((double) a.data[k] - (double) b.data[k]);
                if (d > worst) worst = d;
            }
        }
        printf("  n=%d head_dim=%d heads=%d ok\n", n, head_dim, n_heads);
    }

    printf("max |diff| %.3e\n", worst);
    printf(worst < 1e-5 ? "PASS\n" : "FAIL: difference too large\n");
    return worst < 1e-5 ? 0 : 1;
}
