#include <iostream>
#include "tensor.h"
#include "ops.h"

int main() {
    int n_heads = 12, n = 3, head_dim = 64;

    Tensor q({n_heads, n, head_dim});
    Tensor k({n_heads, n, head_dim});
    for (int f = 0; f < q.size(); f++) { q.data[f] = 1.0f; k.data[f] = 1.0f; }

    // token j's note is just the number j, repeated across all 64 dims
    Tensor v({n_heads, n, head_dim});
    for (int h = 0; h < n_heads; h++)
        for (int j = 0; j < n; j++)
            for (int d = 0; d < head_dim; d++)
                v.at(h, j, d) = static_cast<float>(j);

    Tensor scores  = attention_scores(q, k, 0);
    Tensor weights = softmax(scores);
    Tensor out     = attention_gather(weights, v, 0);

    for (int i = 0; i < n; i++)
        std::cout << "row " << i << ":  " << out.at(i, 0)
                  << "  (col 63: " << out.at(i, 63) << ")\n";

    return 0;
}
