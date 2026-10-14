#pragma once
#include <cstdlib>
#include <string>

inline std::string default_weights_path() {
    const char* env = std::getenv("GPT2_WEIGHTS");
    return env ? env : "../gpt2-from-scratch/gpt2_weights.bin";
}
