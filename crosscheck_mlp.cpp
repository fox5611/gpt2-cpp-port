#include <iostream>
#include <iomanip>
#include <fstream>
#include "tensor.h"
#include "loader.h"
#include "paths.h"
#include "ops.h"

int main() {
    auto W = load_weights(default_weights_path());
    int n = 4, d_model = 768;

    Tensor x({n, d_model});
    for (int i = 0; i < n; i++)
        for (int j = 0; j < d_model; j++)
            x.at(i, j) = ((i * d_model + j) % 100) / 100.0f;

    Tensor out = mlp(x,
        W.at("transformer.h.0.mlp.c_fc.weight"),
        W.at("transformer.h.0.mlp.c_fc.bias"),
        W.at("transformer.h.0.mlp.c_proj.weight"),
        W.at("transformer.h.0.mlp.c_proj.bias"));

    std::ofstream f("dbg_mlp.txt");
    f << std::fixed << std::setprecision(8);
    for (int i = 0; i < n; i++)
        for (int j = 0; j < d_model; j++) f << out.at(i, j) << "\n";

    std::cout << "mlp dumped, shape (" << out.shape[0] << ", " << out.shape[1] << ")\n";
    return 0;
}
