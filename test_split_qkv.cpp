#include <iostream>
#include "tensor.h"
#include "ops.h"

int main() {
    int n = 2;
    int d = 768;

    // qkv filled 0, 1, 2, ... so every value announces its own flat index
    Tensor qkv({n, 3 * d});
    for (int f = 0; f < qkv.size(); f++) {
        qkv.data[f] = static_cast<float>(f);
    }

    Tensor q({n, d});
    Tensor k({n, d});
    Tensor v({n, d});

    split_qkv(qkv, q, k, v);

    std::cout << "row 0:  q=" << q.at(0, 0)
              << "  k=" << k.at(0, 0)
              << "  v=" << v.at(0, 0) << "\n";
    std::cout << "row 1:  q=" << q.at(1, 0)
              << "  k=" << k.at(1, 0)
              << "  v=" << v.at(1, 0) << "\n";

    return 0;
}
