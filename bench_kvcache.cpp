#include <iostream>
#include <functional>
#include <chrono>
#include <vector>
#include "tensor.h"
#include "loader.h"
#include "paths.h"
#include "ops.h"

// A/B of generate() against generate_cached() in one process, then a check on
// whether the cached cost per token stays flat as the context grows. Weights
// are loaded once, outside every timed region.
static double time_ms(const std::function<void()>& f) {
    auto t0 = std::chrono::steady_clock::now();
    f();
    auto t1 = std::chrono::steady_clock::now();
    return std::chrono::duration<double, std::milli>(t1 - t0).count();
}

int main() {
    auto W = load_weights(default_weights_path());
    std::vector<int> prompt = {464, 3139, 286, 4881, 318};

    { std::vector<int> warm = generate_cached(prompt, W, 3); (void)warm; }

    std::cout << "--- prompt=5, n_new=10, best of 3 ---\n";

    double un = 1e30, ca = 1e30;
    for (int r = 0; r < 3; r++) {
        double t = time_ms([&]{ std::vector<int> o = generate(prompt, W, 10); (void)o; });
        if (t < un) un = t;
    }
    for (int r = 0; r < 3; r++) {
        double t = time_ms([&]{ std::vector<int> o = generate_cached(prompt, W, 10); (void)o; });
        if (t < ca) ca = t;
    }

    std::cout.precision(1);
    std::cout << std::fixed;
    std::cout << "  uncached : " << un << " ms\n";
    std::cout << "  cached   : " << ca << " ms\n";
    std::cout.precision(2);
    std::cout << "  speedup  : " << un / ca << "x\n\n";

    std::cout << "--- cached cost per generated token as context grows ---\n";
    std::cout << "  n_new      ms   ms/token\n";
    for (int n_new : {10, 50, 100}) {
        double t = time_ms([&]{ std::vector<int> o = generate_cached(prompt, W, n_new); (void)o; });
        std::cout.precision(1);
        std::cout << "  " << n_new << "\t" << t;
        std::cout.precision(2);
        std::cout << "\t" << t / n_new << "\n";
    }
    return 0;
}
