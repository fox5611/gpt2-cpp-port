#include <iostream>
#include <chrono>

// Peak FMA throughput with no memory traffic at all. Eight independent
// accumulator vectors, because Zen 3 FMA latency is about 4 cycles and there
// are 2 FMA units, so you need at least 8 chains in flight to keep both busy.
// Everything should live in YMM registers; the sink at the end stops the whole
// loop being deleted as dead code.
int main() {
    constexpr int V = 8;   // independent accumulator vectors
    constexpr int W = 8;   // floats per AVX2 vector
    alignas(32) float acc[V][W];

    for (int v = 0; v < V; v++)
        for (int j = 0; j < W; j++)
            acc[v][j] = 1.0f + 0.01f * (v * W + j);

    const float x = 1.0000001f;
    const float y = 0.0000001f;
    const long iters = 500000000L;

    auto t0 = std::chrono::steady_clock::now();
    for (long i = 0; i < iters; i++)
        for (int v = 0; v < V; v++)
            for (int j = 0; j < W; j++)
                acc[v][j] = acc[v][j] * x + y;
    auto t1 = std::chrono::steady_clock::now();

    double s = std::chrono::duration<double>(t1 - t0).count();
    double flops = 2.0 * V * W * (double)iters;

    double sink = 0.0;
    for (int v = 0; v < V; v++)
        for (int j = 0; j < W; j++) sink += acc[v][j];

    std::cout << "time      " << s << " s\n";
    std::cout << "peak FMA  " << flops / (s * 1e9) << " GFLOPS\n";
    std::cout << "sink      " << sink << "\n";
    return 0;
}
