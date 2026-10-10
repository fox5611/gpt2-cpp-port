#pragma once
#include "tensor.h"

Tensor matmul(Tensor& a, Tensor& b) {
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
Tensor add_bias(Tensor& x, Tensor& bias) {
    Tensor out({x.shape[0], x.shape[1]});
    for (int i = 0; i < x.shape[0]; i++)
        for (int j = 0; j < x.shape[1]; j++)
            out.at(i, j) = x.at(i, j) + bias.data[j];
    return out;
}

// element-wise add (for residuals)
Tensor add(Tensor& a, Tensor& b) {
    Tensor out({a.shape[0], a.shape[1]});
    for (int i = 0; i < a.size(); i++)
        out.data[i] = a.data[i] + b.data[i];
    return out;
}
#include <cmath>

Tensor gelu(Tensor& x) {
    Tensor out = x;
    for (int i = 0; i < out.size(); i++) {
        float v = out.data[i];
        out.data[i] = 0.5f * v * (1.0f + std::tanh(0.7978845608f * (v + 0.044715f * v * v * v)));
    }
    return out;
}
Tensor softmax(Tensor& x) {
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
Tensor layernorm(Tensor& x, Tensor& weight, Tensor& bias, float eps = 1e-5f) {
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
Tensor transpose(Tensor& x) {
    int rows = x.shape[0], cols = x.shape[1];
    Tensor out({cols, rows});
    for (int i = 0; i < rows; i++)
        for (int j = 0; j < cols; j++)
            out.at(j, i) = x.at(i, j);
    return out;
}

/*
Tensor split_heads(Tensor& x, int n_heads = 12) {
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

Tensor split_heads(Tensor& x, int n_heads = 12) {
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
