#include <iostream>
#include <chrono>
#include <functional>
#include <cmath>
#include <algorithm>
#include <string>
#include "tensor.h"
#include "loader.h"
#include "paths.h"
#include "ops.h"

static double time_ms(int reps, const std::function<void()>& f) {
    auto t0 = std::chrono::steady_clock::now();
    for (int r = 0; r < reps; r++) f();
    auto t1 = std::chrono::steady_clock::now();
    return std::chrono::duration<double, std::milli>(t1 - t0).count() / reps;
}

int main() {
    auto W = load_weights(default_weights_path());
    const int n = 256;
    std::string p = "transformer.h.0.";

    Tensor& c_attn_w = W.at(p + "attn.c_attn.weight");
    Tensor& c_fc_w   = W.at(p + "mlp.c_fc.weight");
    Tensor& mlp_cp_w = W.at(p + "mlp.c_proj.weight");

    Tensor x768({n, 768});
    for (int i = 0; i < x768.size(); i++) x768.data[i] = 0.01f * ((i % 97) - 48);
    Tensor x3072({n, 3072});
    for (int i = 0; i < x3072.size(); i++) x3072.data[i] = 0.01f * ((i % 89) - 44);

    std::cout << std::fixed;
    std::cout.precision(2);

    auto check = [&](const char* name, Tensor& A, Tensor& B) {
        Tensor s = matmul_simple(A, B);
        Tensor t = matmul(A, B);

        float worst = 0.0f, mag = 0.0f;
        for (int i = 0; i < s.size(); i++) {
            worst = std::max(worst, std::fabs(s.data[i] - t.data[i]));
            mag   = std::max(mag,   std::fabs(s.data[i]));
        }

        double ts = time_ms(3, [&]{ Tensor r = matmul_simple(A, B); (void)r; });
        double tb = time_ms(3, [&]{ Tensor r = matmul(A, B);        (void)r; });
        double flop = 2.0 * A.shape[0] * A.shape[1] * B.shape[1];

        std::cout << "  " << name << "\n"
                  << "    simple  " << ts << " ms  " << flop / (ts * 1e6) << " GFLOPS\n"
                  << "    blocked " << tb << " ms  " << flop / (tb * 1e6) << " GFLOPS\n"
                  << "    speedup " << ts / tb << "   max|d| " << worst
                  << "  against max|value| " << mag << "\n\n";
    };

    std::cout << "n=" << n << "\n\n";
    check("qkv       768 -> 2304", x768,  c_attn_w);
    check("mlp_fc    768 -> 3072", x768,  c_fc_w);
    check("mlp_proj  3072 -> 768", x3072, mlp_cp_w);
    return 0;
}
