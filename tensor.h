#pragma once
#include <vector>

struct Tensor {
    std::vector<float> data;
    std::vector<int> shape;

    Tensor() {}

    Tensor(std::vector<int> s) {
        shape = s;
        int total = 1;
        for (int d : s) {
            total = total * d;
        }
        data.resize(total);
    }

    int size() const {
        return data.size();
    }

    // 2-D access
    float& at(int i, int j) {
        return data[i * shape[1] + j];
    }

    // 3-D access
    float& at(int i, int j, int k) {
        return data[i * shape[1] * shape[2] + j * shape[2] + k];
    }
};
