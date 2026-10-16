#pragma once
#include "tensor.h"

[[nodiscard]] Tensor matmul(Tensor& a, Tensor& b) {
    int M = a.shape[0], K = a.shape[1], N = b.shape[1];
    Tensor out({M, N});
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
            for (int k = 0; k < K; k++) {
                sum += a.at(i, k) * b.at(k, j);
            }
            out.at(i, j) = sum;
        }
    }
    return out;
}
// add a bias vector to every row
[[nodiscard]] Tensor add_bias(Tensor& x, Tensor& bias) {
    Tensor out({x.shape[0], x.shape[1]});
    for (int i = 0; i < x.shape[0]; i++)
        for (int j = 0; j < x.shape[1]; j++)
            out.at(i, j) = x.at(i, j) + bias.data[j];
    return out;
}

// element-wise add (for residuals)
[[nodiscard]] Tensor add(Tensor& a, Tensor& b) {
    Tensor out({a.shape[0], a.shape[1]});
    for (int i = 0; i < a.size(); i++)
        out.data[i] = a.data[i] + b.data[i];
    return out;
}
#include <cmath>

[[nodiscard]] Tensor gelu(Tensor& x) {
    Tensor out = x;
    for (int i = 0; i < out.size(); i++) {
        float v = out.data[i];
        out.data[i] = 0.5f * v * (1.0f + std::tanh(0.7978845608f * (v + 0.044715f * v * v * v)));
    }
    return out;
}
[[nodiscard]] Tensor softmax(Tensor& x) {
    int rows = x.shape[0], cols = x.shape[1];
    Tensor out({rows, cols});
    for (int i = 0; i < rows; i++) {
        // pass 1: find the max
        float max_val = x.at(i, 0);
        for (int j = 1; j < cols; j++)
            if (x.at(i, j) > max_val) max_val = x.at(i, j);

        // pass 2: exponentiate (shifted) and accumulate the sum
        float sum = 0.0f;
        for (int j = 0; j < cols; j++) {
            float e = std::exp(x.at(i, j) - max_val);
            out.at(i, j) = e;
            sum += e;
        }

        // pass 3: divide
        for (int j = 0; j < cols; j++)
            out.at(i, j) = out.at(i, j) / sum;
    }
    return out;
}
[[nodiscard]] Tensor layernorm(Tensor& x, Tensor& weight, Tensor& bias, float eps = 1e-5f) {
    int rows = x.shape[0], cols = x.shape[1];
    Tensor out({rows, cols});
    for (int i = 0; i < rows; i++) {
        float mean = 0.0f;
        for (int j = 0; j < cols; j++) mean += x.at(i, j);
        mean /= cols;

        float var = 0.0f;
        for (int j = 0; j < cols; j++) {
            float d = x.at(i, j) - mean;
            var += d * d;
        }
        var /= cols;

        float inv = 1.0f / std::sqrt(var + eps);
        for (int j = 0; j < cols; j++)
            out.at(i, j) = (x.at(i, j) - mean) * inv * weight.data[j] + bias.data[j];
    }
    return out;
}
[[nodiscard]] Tensor transpose(Tensor& x) {
    int rows = x.shape[0], cols = x.shape[1];
    Tensor out({cols, rows});
    for (int i = 0; i < rows; i++)
        for (int j = 0; j < cols; j++)
            out.at(j, i) = x.at(i, j);
    return out;
}

/*
[[nodiscard]] Tensor split_heads(Tensor& x, int n_heads = 12) {
    int n = x.shape[0];
    int d_model = x.shape[1];
    int head_dim = d_model / n_heads;   // 768 / 12 = 64

    Tensor out({n_heads, n, head_dim});
    for (int t = 0; t < n; t++) {
        for (int h = 0; h < n_heads; h++) {
            for (int d = 0; d < head_dim; d++) {
                out.at(h, t, d) = x.at(t, h * head_dim + d);
            }
        }
    }
    return out;

}
*/

[[nodiscard]] Tensor split_heads(Tensor& x, int n_heads = 12) {
    int n = x.shape[0];
    int d_model = x.shape[1];
    int head_dim = d_model / n_heads;

    Tensor out({n_heads, n, head_dim});
    for (int t = 0; t < n; t++) {
        for (int h = 0; h < n_heads; h++) {
            for (int d = 0; d < head_dim; d++) {
                out.at(h, t, d) = x.at(t, h * head_dim + d);
            }
        }
    }
    return out;
}
void split_qkv(Tensor& qkv, Tensor& q, Tensor& k, Tensor& v) {
    int n = qkv.shape[0];      // sequence length
    int d = q.shape[1];        // 768 — width of one wing

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < d; j++) {
            q.at(i, j) = qkv.at(i,         j);
            k.at(i, j) = qkv.at(i,     d + j);
            v.at(i, j) = qkv.at(i, 2 * d + j);
        }
    }
}
[[nodiscard]] Tensor attention_scores(Tensor& q, Tensor& k, int h) {
    int n        = q.shape[1];   // sequence length
    int head_dim = q.shape[2];   // 64

    Tensor scores({n, n});
    float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            float dot = 0.0f;
            for (int d = 0; d < head_dim; d++) {
                dot += q.at(h, i, d) * k.at(h, j, d);
            }
            scores.at(i, j) = (j > i) ? -1e9f : dot * scale;
        }
    }

    return scores;
}
[[nodiscard]] Tensor attention_gather(Tensor& weights, Tensor& v, int h) {
    int n        = weights.shape[0];
    int head_dim = v.shape[2];

    Tensor out({n, head_dim});

    for (int i = 0; i < n; i++) {
        for (int d = 0; d < head_dim; d++) {
            float sum = 0.0f;
            for (int j = 0; j < n; j++) {
                sum += weights.at(i, j) * v.at(h, j, d);
            }
            out.at(i, d) = sum;
        }
    }

    return out;
}
[[nodiscard]] Tensor combine_heads(Tensor& x) {
    int n_heads  = x.shape[0];
    int n        = x.shape[1];
    int head_dim = x.shape[2];

    Tensor out({n, n_heads * head_dim});

    for (int h = 0; h < n_heads; h++)
        for (int t = 0; t < n; t++)
            for (int d = 0; d < head_dim; d++)
                out.at(t, h * head_dim + d) = x.at(h, t, d);

    return out;
}
[[nodiscard]] Tensor attention(Tensor& x,
                 Tensor& c_attn_w, Tensor& c_attn_b,
                 Tensor& c_proj_w, Tensor& c_proj_b,
                 int n_heads = 12) {
    int n        = x.shape[0];
    int d_model  = x.shape[1];          // 768
    int head_dim = d_model / n_heads;   // 64

    // 1. QKV projection
    Tensor qkv_raw = matmul(x, c_attn_w);
    Tensor qkv     = add_bias(qkv_raw, c_attn_b);
    // 2. slice into Q, K, V
    Tensor q({n, d_model}), k({n, d_model}), v({n, d_model});
    split_qkv(qkv, q, k, v);

    // 3. split each into heads
    Tensor qh = split_heads(q, n_heads);   // (12, n, 64)
    Tensor kh = split_heads(k, n_heads);
    Tensor vh = split_heads(v, n_heads);

    // 4. one head at a time
    Tensor heads({n_heads, n, head_dim});
    for (int h = 0; h < n_heads; h++) {
        Tensor scores  = attention_scores(qh, kh, h);
        Tensor weights = softmax(scores);
        Tensor out_h   = attention_gather(weights, vh, h);

        for (int i = 0; i < n; i++)
            for (int d = 0; d < head_dim; d++)
                heads.at(h, i, d) = out_h.at(i, d);
    }

    // 5. recombine
    Tensor combined = combine_heads(heads);   // (n, 768)

    // 6. output projection
    Tensor proj = matmul(combined, c_proj_w);
    Tensor out  = add_bias(proj, c_proj_b);
    return out;
}

[[nodiscard]] Tensor mlp(Tensor& x,
                         Tensor& c_fc_w, Tensor& c_fc_b,
                         Tensor& c_proj_w, Tensor& c_proj_b) {
    Tensor h_raw = matmul(x, c_fc_w);        // (n, 768) @ (768, 3072) -> (n, 3072)
    Tensor h     = add_bias(h_raw, c_fc_b);
    Tensor a     = gelu(h);
    Tensor p     = matmul(a, c_proj_w);      // (n, 3072) @ (3072, 768) -> (n, 768)
    Tensor out   = add_bias(p, c_proj_b);
    return out;
}

[[nodiscard]] Tensor transformer_block(
        Tensor& x,
        Tensor& ln_1_w, Tensor& ln_1_b,
        Tensor& c_attn_w, Tensor& c_attn_b,
        Tensor& c_proj_w, Tensor& c_proj_b,
        Tensor& ln_2_w, Tensor& ln_2_b,
        Tensor& c_fc_w, Tensor& c_fc_b,
        Tensor& mlp_c_proj_w, Tensor& mlp_c_proj_b) {

    Tensor n1 = layernorm(x, ln_1_w, ln_1_b);
    Tensor a  = attention(n1, c_attn_w, c_attn_b, c_proj_w, c_proj_b);
    Tensor x1 = add(x, a);                      // residual 1

    Tensor n2 = layernorm(x1, ln_2_w, ln_2_b);
    Tensor m  = mlp(n2, c_fc_w, c_fc_b, mlp_c_proj_w, mlp_c_proj_b);
    Tensor x2 = add(x1, m);                     // residual 2

    return x2;
}
[[nodiscard]] Tensor gpt2_forward(std::vector<int>& token_ids,
                                  std::map<std::string, Tensor>& W,
                                  int n_layers = 12) {
    Tensor& wte = W.at("transformer.wte.weight");   // (50257, 768)
    Tensor& wpe = W.at("transformer.wpe.weight");   // (1024, 768)

    int n       = (int)token_ids.size();
    int d_model = wte.shape[1];
    int vocab   = wte.shape[0];

    // embeddings: token meaning + position
    Tensor x({n, d_model});
    for (int t = 0; t < n; t++)
        for (int j = 0; j < d_model; j++)
            x.at(t, j) = wte.at(token_ids[t], j) + wpe.at(t, j);

    for (int i = 0; i < n_layers; i++) {
        std::string p = "transformer.h." + std::to_string(i) + ".";
        x = transformer_block(x,
            W.at(p + "ln_1.weight"),        W.at(p + "ln_1.bias"),
            W.at(p + "attn.c_attn.weight"), W.at(p + "attn.c_attn.bias"),
            W.at(p + "attn.c_proj.weight"), W.at(p + "attn.c_proj.bias"),
            W.at(p + "ln_2.weight"),        W.at(p + "ln_2.bias"),
            W.at(p + "mlp.c_fc.weight"),    W.at(p + "mlp.c_fc.bias"),
            W.at(p + "mlp.c_proj.weight"),  W.at(p + "mlp.c_proj.bias"));
    }

    Tensor xf = layernorm(x, W.at("transformer.ln_f.weight"),
                             W.at("transformer.ln_f.bias"));

    // tied output head: logits = xf @ wte.T, no transpose materialized
    Tensor logits({n, vocab});
    for (int t = 0; t < n; t++)
        for (int vq = 0; vq < vocab; vq++) {
            float s = 0.0f;
            for (int j = 0; j < d_model; j++)
                s += xf.at(t, j) * wte.at(vq, j);
            logits.at(t, vq) = s;
        }

    return logits;
}
[[nodiscard]] std::vector<int> generate(std::vector<int> token_ids,
                                        std::map<std::string, Tensor>& W,
                                        int n_new) {
    for (int step = 0; step < n_new; step++) {
        Tensor logits = gpt2_forward(token_ids, W);

        int last = (int)token_ids.size() - 1;
        int best = 0;
        for (int v = 1; v < logits.shape[1]; v++)
            if (logits.at(last, v) > logits.at(last, best)) best = v;

        token_ids.push_back(best);
    }
    return token_ids;
}
