#include <iostream>
#include <cmath>
#include "tensor.h"
#include "ops.h"

int main() {
    int n = 3, d_model = 768;

    Tensor x({n, d_model});
    for (int f = 0; f < x.size(); f++) x.data[f] = static_cast<float>(f);

    Tensor heads = split_heads(x);
    Tensor back  = combine_heads(heads);

    int mismatches = 0;
    float max_diff = 0.0f;
    for (int f = 0; f < x.size(); f++) {
        float diff = std::fabs(back.data[f] - x.data[f]);
        if (diff > 0.0f) mismatches++;
        if (diff > max_diff) max_diff = diff;
    }

    std::cout << "shape: (" << back.shape[0] << ", " << back.shape[1] << ")\n";
    std::cout << "mismatches: " << mismatches << " / " << x.size() << "\n";
    std::cout << "max diff: " << max_diff << "\n";
    return 0;
}
