// test_softmax_simd.cpp
#include "ops.h"
#include <cstdio>
#include <cmath>

int main() {
    const int cases[][2] = {{1,1}, {1,7}, {1,8}, {3,9}, {5,17}, {256,256}};
    double worst = 0.0;
    int wr = -1, wc = -1;

    for (auto & c : cases) {
        int rows = c[0], cols = c[1];
        Tensor x({rows, cols});
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                float v = 0.01f * (float) ((i * 7 + j * 13) % 211) - 1.0f;
                if (rows == cols && j > i) {
                    v = -1e9f;            // the causal mask attention_scores writes
                }
                x.at(i, j) = v;
            }
        }

        Tensor a = softmax_scalar(x);
        Tensor b = softmax(x);
        if (a.size() != b.size()) { printf("FAIL size\n"); return 1; }

        for (int i = 0; i < rows; i++) {
            double rowsum = 0.0;
            for (int j = 0; j < cols; j++) {
                double d = std::fabs((double) a.at(i, j) - (double) b.at(i, j));
                if (d > worst) { worst = d; wr = i; wc = j; }
                rowsum += (double) b.at(i, j);
            }
            if (std::fabs(rowsum - 1.0) > 1e-4) {
                printf("FAIL (%d,%d) row %d sums to %.8f\n", rows, cols, i, rowsum);
                return 1;
            }
        }
        printf("  (%d,%d) ok\n", rows, cols);
    }

    printf("max |diff| %.3e at (%d,%d)\n", worst, wr, wc);
    printf(worst < 1e-6 ? "PASS\n" : "FAIL: difference too large\n");
    return worst < 1e-6 ? 0 : 1;
}
