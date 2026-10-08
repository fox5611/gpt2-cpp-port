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
