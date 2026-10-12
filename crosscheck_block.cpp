#include <iostream>
#include <iomanip>
#include <fstream>
#include "tensor.h"
#include "loader.h"
#include "ops.h"

int main() {
    auto W = load_weights("/home/rthacker/ai/gpt2-from-scratch/gpt2_weights.bin");
    int n = 4, d_model = 768;

    Tensor x({n, d_model});
    for (int i = 0; i < n; i++)
        for (int j = 0; j < d_model; j++)
            x.at(i, j) = ((i * d_model + j) % 100) / 100.0f;

    std::string P = "transformer.h.0.";
    Tensor out = transformer_block(x,
        W.at(P + "ln_1.weight"),        W.at(P + "ln_1.bias"),
        W.at(P + "attn.c_attn.weight"), W.at(P + "attn.c_attn.bias"),
        W.at(P + "attn.c_proj.weight"), W.at(P + "attn.c_proj.bias"),
        W.at(P + "ln_2.weight"),        W.at(P + "ln_2.bias"),
        W.at(P + "mlp.c_fc.weight"),    W.at(P + "mlp.c_fc.bias"),
        W.at(P + "mlp.c_proj.weight"),  W.at(P + "mlp.c_proj.bias"));

    std::ofstream f("/home/rthacker/ai/gpt2-cpp-port/dbg_block.txt");
    f << std::fixed << std::setprecision(8);
    for (int i = 0; i < n; i++)
        for (int j = 0; j < d_model; j++) f << out.at(i, j) << "\n";

    std::cout << "block dumped, shape (" << out.shape[0] << ", " << out.shape[1] << ")\n";
    return 0;
}
