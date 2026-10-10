#include <iostream>
#include "tensor.h"
#include "ops.h"

int main() {
    int n_heads = 12, n = 3, head_dim = 64;

    // every value 1.0 -> every dot product is identical
    Tensor q({n_heads, n, head_dim});
    Tensor k({n_heads, n, head_dim});
    for (int f = 0; f < q.size(); f++) { q.data[f] = 1.0f; k.data[f] = 1.0f; }

    Tensor scores = attention_scores(q, k, 0);
    Tensor weights = softmax(scores);

    for (int i = 0; i < n; i++) {
        std::cout << "row " << i << ":  ";
        for (int j = 0; j < n; j++) std::cout << weights.at(i, j) << "  ";
        std::cout << "\n";
    }
    return 0;
}
