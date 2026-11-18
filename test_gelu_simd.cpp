// test_gelu_simd.cpp
#include "ops.h"
#include <cstdio>
#include <cmath>

int main() {
    // include the awkward sizes: not a multiple of 8, and the saturating tails
    const int cases[][2] = {{1, 1}, {1, 7}, {1, 8}, {3, 9}, {256, 3072}};
    double worst_abs = 0.0, worst_rel = 0.0;
    int worst_i = -1;

    for (auto & c : cases) {
        Tensor x({c[0], c[1]});
        for (int i = 0; i < x.size(); i++) {
            // sweep -20..20 so the exp clamp and both saturating limbs get hit
            x.data[i] = -20.0f + 40.0f * (float) (i % 997) / 996.0f;
        }
        Tensor a = gelu_scalar(x);
        Tensor b = gelu(x);
        if (a.size() != b.size()) { printf("FAIL size %d vs %d\n", a.size(), b.size()); return 1; }
        for (int i = 0; i < a.size(); i++) {
            double d = std::fabs((double) a.data[i] - (double) b.data[i]);
            double r = d / (std::fabs((double) a.data[i]) + 1e-6);
            if (d > worst_abs) { worst_abs = d; worst_i = i; }
            if (r > worst_rel) worst_rel = r;
        }
        printf("  (%d,%d) ok\n", c[0], c[1]);
    }

    printf("max |diff| %.3e at %d, max rel %.3e\n", worst_abs, worst_i, worst_rel);
    printf(worst_abs < 1e-5 ? "PASS\n" : "FAIL: difference too large\n");
    return worst_abs < 1e-5 ? 0 : 1;
}
