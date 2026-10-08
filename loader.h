#pragma once
#include <fstream>
#include <string>
#include <map>
#include <sstream>
#include "tensor.h"

std::map<std::string, Tensor> load_weights(const std::string& path) {
    std::map<std::string, Tensor> weights;
    std::ifstream f(path, std::ios::binary);

    std::string line;
    std::getline(f, line);
    int n_tensors = std::stoi(line);

    std::vector<std::string> names;
    std::vector<std::vector<int>> shapes;

    for (int i = 0; i < n_tensors; i++) {
        std::getline(f, line);
        std::istringstream iss(line);
        std::string name;
        int ndim;
        iss >> name >> ndim;
        std::vector<int> shape;
        for (int d = 0; d < ndim; d++) {
            int dim;
            iss >> dim;
            shape.push_back(dim);
        }
        names.push_back(name);
        shapes.push_back(shape);
    }

    std::getline(f, line);   // END_HEADER

    for (int i = 0; i < n_tensors; i++) {
        Tensor t(shapes[i]);
        f.read(reinterpret_cast<char*>(t.data.data()), t.size() * sizeof(float));
        weights[names[i]] = t;
    }

    return weights;
}

