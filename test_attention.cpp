#include <iostream>
#include "tensor.h"
#include "ops.h"

int main() {
    int n = 3, d_model = 768;

    // x = all ones
    Tensor x({n, d_model});
    for (int f = 0; f < x.size(); f++) x.data[f] = 1.0f;

    // c_attn_w = three stacked identities, so Q = K = V = x
    Tensor c_attn_w({d_model, 3 * d_model});
    for (int f = 0; f < c_attn_w.size(); f++) c_attn_w.data[f] = 0.0f;
    for (int i = 0; i < d_model; i++)
        for (int blk = 0; blk < 3; blk++)
            c_attn_w.at(i, blk * d_model + i) = 1.0f;

    Tensor c_attn_b({3 * d_model});
    for (int f = 0; f < c_attn_b.size(); f++) c_attn_b.data[f] = 0.0f;

    // c_proj_w = identity
    Tensor c_proj_w({d_model, d_model});
    for (int f = 0; f < c_proj_w.size(); f++) c_proj_w.data[f] = 0.0f;
    for (int i = 0; i < d_model; i++) c_proj_w.at(i, i) = 1.0f;

    Tensor c_proj_b({d_model});
    for (int f = 0; f < c_proj_b.size(); f++) c_proj_b.data[f] = 0.0f;

    Tensor out = attention(x, c_attn_w, c_attn_b, c_proj_w, c_proj_b);

    std::cout << "shape: (" << out.shape[0] << ", " << out.shape[1] << ")\n";
    for (int i = 0; i < n; i++)
        std::cout << "row " << i << ":  " << out.at(i, 0)
                  << "  " << out.at(i, 767) << "\n";
    return 0;
}
