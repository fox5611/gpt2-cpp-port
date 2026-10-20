#include <iostream>
#include <cmath>
#include <vector>
#include "tensor.h"
#include "loader.h"
#include "paths.h"
#include "ops.h"

// Verifies that gpt2_forward(..., last_row_only=true) returns exactly the final
// row of the full logits tensor. The arithmetic per row is untouched, so the
// values should be bitwise identical -- any drift means the shorter outer loop
// changed how the inner accumulation got vectorized.
int main() {
    auto W = load_weights(default_weights_path());
    std::vector<int> token_ids = {464, 3139, 286, 4881, 318};
    const int n = (int)token_ids.size();

    Tensor full = gpt2_forward(token_ids, W, 12, /*last_row_only=*/false);
    Tensor last = gpt2_forward(token_ids, W, 12, /*last_row_only=*/true);

    bool ok = true;

    if (full.shape[0] != n) {
        std::cout << "FAIL full rows: got " << full.shape[0]
                  << ", expected " << n << "\n";
        ok = false;
    }
    if (last.shape[0] != 1 || last.shape[1] != full.shape[1]) {
        std::cout << "FAIL last shape: got (" << last.shape[0] << ", " << last.shape[1]
                  << "), expected (1, " << full.shape[1] << ")\n";
        ok = false;
    }

    if (ok) {
        float max_diff = 0.0f;
        int   worst_v  = -1;
        for (int v = 0; v < full.shape[1]; v++) {
            float d = std::fabs(full.at(n - 1, v) - last.at(0, v));
            if (d > max_diff) { max_diff = d; worst_v = v; }
        }
        std::cout << "max abs diff vs final row: " << max_diff
                  << " (vocab index " << worst_v << ")\n";
        if (max_diff != 0.0f) {
            std::cout << "FAIL: expected bitwise-identical values\n";
            ok = false;
        }
    }

    // The property generate() actually depends on.
    int best_full = 0, best_last = 0;
    for (int v = 1; v < full.shape[1]; v++) {
        if (full.at(n - 1, v) > full.at(n - 1, best_full)) best_full = v;
        if (last.at(0,     v) > last.at(0,     best_last)) best_last = v;
    }
    std::cout << "argmax final row: " << best_full
              << ", argmax last_row_only: " << best_last << "\n";
    if (best_full != best_last) {
        std::cout << "FAIL: argmax disagrees\n";
        ok = false;
    }

    std::cout << (ok ? "PASS\n" : "FAILED\n");
    return ok ? 0 : 1;
}
