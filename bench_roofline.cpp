#include <iostream>
#include <chrono>
#include <functional>
#include "tensor.h"
#include "ops.h"

static double time_ms(int reps, const std::function<void()>& f) {
    auto t0 = std::chrono::steady_clock::now();
    for (int r = 0; r < reps; r++) f();
    auto t1 = std::chrono::steady_clock::now();
    return std::chrono::duration<double, std::milli>(t1 - t0).count() / reps;
}

// Same kernel, same M and K, only the size of b changes. If GFLOPS holds steady
// as b grows past L2 and then past L3, the kernel is compute-bound and no amount
// of cache work will help. If it falls, it is memory-bound and the fall says
// which level is the wall.
int main() {
    const int M = 256, K = 256;
    std::cout << "M=256 K=256, L1 32 KB, L2 512 KB, L3 32 MB\n\n";
    std::cout << "      N    b size        ms    GFLOPS\n";
    std::cout << std::fixed;

    for (int N : {64, 256, 1024, 4096, 16384, 65536}) {
        Tensor a({M, K}), b({K, N});
        for (int i = 0; i < a.size(); i++) a.data[i] = 0.01f * ((i % 97) - 48);
        for (int i = 0; i < b.size(); i++) b.data[i] = 0.01f * ((i % 89) - 44);

        double t    = time_ms(5, [&]{ Tensor r = matmul(a, b); (void)r; });
        double flop = 2.0 * M * K * N;
        double kb   = K * (double)N * 4.0 / 1024.0;

        std::cout.precision(0);
        std::cout << "  " << N << "\t  " << kb << " KB";
        std::cout.precision(2);
        std::cout << "\t  " << t << "\t  " << flop / (t * 1e6) << "\n";
    }
    return 0;
}
