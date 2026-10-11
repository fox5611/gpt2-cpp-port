#include <iostream>
#include <iomanip>
#include <fstream>
#include "tensor.h"
#include "loader.h"
#include "ops.h"

void dump2(const std::string& path, Tensor& t) {
    std::ofstream f(path);
    f << std::fixed << std::setprecision(8);
    for (int i = 0; i < t.shape[0]; i++)
        for (int j = 0; j < t.shape[1]; j++) f << t.at(i, j) << "\n";
}

void dump3head(const std::string& path, Tensor& t, int h) {
    std::ofstream f(path);
    f << std::fixed << std::setprecision(8);
    for (int i = 0; i < t.shape[1]; i++)
        for (int d = 0; d < t.shape[2]; d++) f << t.at(h, i, d) << "\n";
}

int main() {
    auto W = load_weights("/home/rthacker/ai/gpt2-from-scratch/gpt2_weights.bin");
    int n = 4, d_model = 768, n_heads = 12, head_dim = 64;

    Tensor x({n, d_model});
    for (int i = 0; i < n; i++)
        for (int j = 0; j < d_model; j++)
            x.at(i, j) = ((i * d_model + j) % 100) / 100.0f;

    Tensor& c_attn_w = W.at("transformer.h.0.attn.c_attn.weight");
    Tensor& c_attn_b = W.at("transformer.h.0.attn.c_attn.bias");
    Tensor& c_proj_w = W.at("transformer.h.0.attn.c_proj.weight");
    Tensor& c_proj_b = W.at("transformer.h.0.attn.c_proj.bias");

    std::string D = "/home/rthacker/ai/gpt2-cpp-port/dbg_";

    Tensor qkv_raw = matmul(x, c_attn_w);
    Tensor qkv     = add_bias(qkv_raw, c_attn_b);
    dump2(D + "qkv.txt", qkv);

    Tensor q({n, d_model}), k({n, d_model}), v({n, d_model});
    split_qkv(qkv, q, k, v);
    dump2(D + "q.txt", q);

    Tensor qh = split_heads(q, n_heads);
    Tensor kh = split_heads(k, n_heads);
    Tensor vh = split_heads(v, n_heads);
    dump3head(D + "qh0.txt", qh, 0);

    Tensor scores = attention_scores(qh, kh, 0);
    dump2(D + "scores0.txt", scores);

    Tensor weights = softmax(scores);
    dump2(D + "weights0.txt", weights);

    Tensor outh = attention_gather(weights, vh, 0);
    dump2(D + "gather0.txt", outh);

    Tensor heads({n_heads, n, head_dim});
    for (int h = 0; h < n_heads; h++) {
        Tensor s = attention_scores(qh, kh, h);
        Tensor w = softmax(s);
        Tensor o = attention_gather(w, vh, h);
        for (int i = 0; i < n; i++)
            for (int d = 0; d < head_dim; d++) heads.at(h, i, d) = o.at(i, d);
    }
    Tensor combined = combine_heads(heads);
    dump2(D + "combined.txt", combined);

    Tensor proj = matmul(combined, c_proj_w);
    Tensor out  = add_bias(proj, c_proj_b);
    dump2(D + "out.txt", out);

    // the real function, end to end
    Tensor outfn = attention(x, c_attn_w, c_attn_b, c_proj_w, c_proj_b);
    dump2(D + "outfn.txt", outfn);

    std::cout << "dumped\n";
    return 0;
}
