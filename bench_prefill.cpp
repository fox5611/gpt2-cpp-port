#include <iostream>
#include <functional>
#include <chrono>
#include <vector>
#include "tensor.h"
#include "loader.h"
#include "paths.h"
#include "ops.h"

// Isolates prefill. n_new is pinned at 1 so decode contributes a constant rather
// than a trend, and every number is prompt cost. Weights load once, outside the
// timed region.
static double time_ms(const std::function<void()>& f) {
    auto t0 = std::chrono::steady_clock::now();
    f();
    auto t1 = std::chrono::steady_clock::now();
    return std::chrono::duration<double, std::milli>(t1 - t0).count();
}

int main() {
    auto W = load_weights(default_weights_path());

    // Token content does not affect cost, only count.
    auto make_prompt = [](int n) {
        std::vector<int> p;
        p.reserve(n);
        for (int i = 0; i < n; i++) p.push_back(464 + (i % 97));
        return p;
    };

    { std::vector<int> warm = generate_cached(make_prompt(8), W, 1); (void)warm; }

    std::cout << "--- prefill cost vs prompt length, n_new=1, best of 3 ---\n";
    std::cout << "  prompt\tms\tms/prompt_token\n";
    std::cout << std::fixed;
    for (int n : {1, 16, 64, 128, 256}) {
        std::vector<int> prompt = make_prompt(n);
        double best = 1e30;
        for (int r = 0; r < 3; r++) {
            double t = time_ms([&]{ std::vector<int> o = generate_cached(prompt, W, 1); (void)o; });
            if (t < best) best = t;
        }
        std::cout.precision(1);
        std::cout << "  " << n << "\t" << best;
        std::cout.precision(3);
        std::cout << "\t" << best / n << "\n";
    }
    return 0;
}
