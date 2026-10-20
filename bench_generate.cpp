#include <iostream>
#include <chrono>
#include <vector>
#include "tensor.h"
#include "loader.h"
#include "paths.h"
#include "ops.h"

// Times generate() only -- weights are loaded once, outside the timed region,
// and one warm pass is discarded. Any number printed here excludes the ~475 MB
// weight load; quote it that way.
int main() {
    auto W = load_weights(default_weights_path());
    std::vector<int> prompt = {464, 3139, 286, 4881, 318};
    const int n_new = 10;

    std::vector<int> warm = generate(prompt, W, n_new);   // discard
    std::cout << "warm-up produced " << warm.size() << " tokens\n";

    for (int run = 0; run < 3; run++) {
        auto t0 = std::chrono::steady_clock::now();
        std::vector<int> out = generate(prompt, W, n_new);
        auto t1 = std::chrono::steady_clock::now();

        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

        std::cout << "run " << run << ": " << ms << " ms for " << n_new
                  << " tokens from a " << prompt.size() << "-token prompt"
                  << " (" << ms / n_new << " ms/token), last id " << out.back() << "\n";
    }
    return 0;
}
