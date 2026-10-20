#include <iostream>
#include <chrono>
#include <vector>
#include "tensor.h"
#include "loader.h"
#include "paths.h"
#include "ops.h"

// Times ONE forward pass at increasing sequence length, with last_row_only on
// so the output head is a fixed cost (one row) at every n.
//
// If a step is bound by streaming the ~340 MB of layer weights from DRAM, that
// cost is paid once per pass regardless of n, so ms should stay nearly flat.
// If it is bound by arithmetic, ms should grow roughly linearly with n.
int main() {
    auto W = load_weights(default_weights_path());

    const std::vector<int> lengths = {5, 10, 25, 50, 100, 200, 400};

    // warm-up at the largest length: pay page faults once, up front
    {
        std::vector<int> ids(lengths.back(), 3139);
        Tensor warm = gpt2_forward(ids, W, 12, /*last_row_only=*/true);
        std::cout << "warm-up done, logits shape (" << warm.shape[0]
                  << ", " << warm.shape[1] << ")\n\n";
    }

    std::cout << "    n      ms     ms/token\n";
    for (int n : lengths) {
        std::vector<int> ids(n, 3139);

        double best = 1e30;
        for (int run = 0; run < 3; run++) {
            auto t0 = std::chrono::steady_clock::now();
            Tensor logits = gpt2_forward(ids, W, 12, /*last_row_only=*/true);
            auto t1 = std::chrono::steady_clock::now();
            double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
            if (ms < best) best = ms;
            if (logits.shape[0] != 1) {
                std::cout << "unexpected logits rows: " << logits.shape[0] << "\n";
                return 1;
            }
        }
        std::cout << std::fixed;
        std::cout.precision(2);
        std::cout << "  " << n << "\t" << best << "\t" << best / n << "\n";
    }
    return 0;
}
