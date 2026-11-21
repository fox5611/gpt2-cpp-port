// test_layernorm_simd.cpp
#include "ops.h"
#include <cstdio>
#include <cmath>

int main() {
    const int cases[][2] = {{1,1}, {1,7}, {1,8}, {3,9}, {5,17}, {256,768}};
    double worst = 0.0;

    for (auto & c : cases) {
        int rows = c[0], cols = c[1];
        Tensor x({rows, cols}), w({1, cols}), b({1, cols});
        for (int i = 0; i < x.size(); i++) {
            x.data[i] = 0.05f * (float) ((i * 31) % 211) - 5.0f;
        }
        for (int j = 0; j < cols; j++) {
            w.data[j] = 0.5f + 0.01f * (float) (j % 17);
            b.data[j] = -0.2f + 0.01f * (float) (j % 13);
        }

        Tensor a = layernorm_scalar(x, w, b);
        Tensor r = layernorm(x, w, b);
        if (a.size() != r.size()) { printf("FAIL size\n"); return 1; }
        for (int k = 0; k < a.size(); k++) {
            double d = std::fabs((double) a.data[k] - (double) r.data[k]);
            if (d > worst) worst = d;
        }
        printf("  (%d,%d) ok\n", rows, cols);
    }

    printf("max |diff| %.3e\n", worst);
    printf(worst < 1e-5 ? "PASS\n" : "FAIL: difference too large\n");
    return worst < 1e-5 ? 0 : 1;
}
