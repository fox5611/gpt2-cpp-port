// test_logits_simd.cpp
#include "ops.h"
#include <cstdio>
#include <cmath>

int main() {
    const int cases[][3] = {{1, 8, 3}, {3, 13, 7}, {4, 768, 11}, {2, 768, 50257}};
    double worst = 0.0;

    for (auto & c : cases) {
        int n = c[0], d = c[1], vocab = c[2];
        Tensor xf({n, d});
        Tensor wte({vocab, d});
        for (int i = 0; i < xf.size(); i++)  xf.data[i]  = 0.01f * (float) (i % 197) - 1.0f;
        for (int i = 0; i < wte.size(); i++) wte.data[i] = 0.01f * (float) (i % 131) - 0.5f;

        Tensor a = logits_last_row_scalar(xf, wte, n, d, vocab);
        Tensor b = logits_last_row(xf, wte, n, d, vocab);
        if (a.size() != b.size()) { printf("FAIL size\n"); return 1; }
        for (int v = 0; v < vocab; v++) {
            double diff = std::fabs((double) a.data[v] - (double) b.data[v]);
            double rel  = diff / (std::fabs((double) a.data[v]) + 1.0);
            if (rel > worst) worst = rel;
        }
        printf("  n=%d d=%d vocab=%d ok\n", n, d, vocab);
    }

    printf("max rel %.3e\n", worst);
    printf(worst < 1e-4 ? "PASS\n" : "FAIL: difference too large\n");
    return worst < 1e-4 ? 0 : 1;
}
