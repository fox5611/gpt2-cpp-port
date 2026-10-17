# gpt2-cpp-port

An implementation of GPT-2 (124M) inference in C++17, with no ML frameworks, no BLAS, and no external math libraries. Every tensor operation is written by hand. It loads the real OpenAI GPT-2 weights (148 tensors, 475 MB, from a binary format I exported myself) and generates text.

It's a port of my Python/numpy implementation ([gpt2-from-scratch](https://github.com/fox5611/gpt2-from-scratch)), which I built and verified against PyTorch. Because I already understood the architecture, the difficulty was the language rather than the model. Two lines of Python (`qkv = x @ c_attn_w + c_attn_b`, then a slice) became three functions and seven nested loops: numpy's `@` is a triple loop here, its broadcasting is an explicit loop adding the bias to every row, and its free slice is a real copy of 9,216 floats, because a tensor in this project is flat memory with no strides. I verified each operation before moving on, but four unit tests on `attention` passed with a real bug in it, because each used zero or uniform data. The cross-check against the verified Python is what caught it.

## Verification

The model's forward pass outputs 50257 floats per token. Nothing about `5.399480` tells you whether it's correct. There's no value a human can check by looking. Reading the code doesn't catch it either: `add_bias(qkv, c_attn_b);` compiled clean, was valid C++, and every function it touched was individually verified. The bug was invisible to inspection.

Instead, I generated identical input in both languages from integer arithmetic (`((i * 768 + j) % 100) / 100.0`) so there's no file to parse and no rounding mismatch, ran both, and compared every element. When the comparison failed, I dumped eight intermediate stages from both sides and found the first divergence in one round trip, instead of guessing.

The Python implementation was verified component by component against PyTorch (max difference 1.1e-04 on a full forward pass). Each C++ component was then compared against it on identical inputs, using real GPT-2 weights:

| Component | Relative error vs Python |
|---|---|
| `attention` | 4.1e-07 |
| `mlp` | 1.7e-06 |
| `transformer_block` | 9.8e-07 |
| `gpt2_forward` | 2.7e-06 |
| `generate` | identical token IDs |

float32 carries about 7 decimal digits. Both implementations multiply the same numbers, but they add them in different orders: numpy dispatches to BLAS, which blocks and reorders accumulation and may fuse multiply-add, while mine sums left to right. Across components I observed 4e-07 to 3e-06 relative. `mlp`'s absolute difference was 1.75e-04, which sounds alarming until you notice its values reach 104, so relative that's 1.7e-06. An absolute threshold would have failed a correct implementation.

The decisive check isn't a float comparison at all: it's identical token IDs. `argmax` is immune to noise smaller than the gap between the top two logits. Given the prompt `The capital of France is` and greedy decoding, both implementations emit the same 10 tokens:

```
464 3139 286 4881 318 262 3139 286 262 4141 2066 11 290 262 3139
```

Decoded:

```
The capital of France is the capital of the French Republic, and the capital
```

GPT-2 124M loops under greedy decoding. That is a property of the model and the sampling strategy, not of this port.

## Layout

| File | Contents |
|---|---|
| `tensor.h` | `Tensor`: a flat `std::vector<float>` plus a shape vector, with 2-D and 3-D indexing |
| `loader.h` | `load_weights()`: parses the exported binary into `std::map<std::string, Tensor>` |
| `paths.h` | resolves the weights path from `GPT2_WEIGHTS`, or a default |
| `ops.h` | every operation, from `matmul` up to `gpt2_forward` and `generate` |
| `test_*.cpp` | one unit test per operation |
| `crosscheck_*.cpp` | dumps intermediates for comparison against Python |
| `verify/*.py` | the Python side of each comparison |

## Requirements

- a C++17 compiler
- `gpt2_weights.bin` (475 MB), produced by `export_weights.py` in the Python repo. Not included here.
- the Python repo cloned alongside this one, if you want to run the verification

## Build

Header-only, so each test and cross-check compiles standalone:

```
g++ -std=c++17 -O2 -o crosscheck_forward crosscheck_forward.cpp
./crosscheck_forward
```

Weights are looked up at `../gpt2-from-scratch/gpt2_weights.bin` by default. Override with `GPT2_WEIGHTS=/path/to/gpt2_weights.bin`.

To reproduce a verification, run the C++ side first (it writes `dbg_*.txt`), then the Python side from the Python repo:

```
cd gpt2-cpp-port      && ./crosscheck_forward
cd gpt2-from-scratch  && python3 ../gpt2-cpp-port/verify/crosscheck_forward.py
```

The Python scripts import from the Python repo's `model.py`, so they must run from that directory. `GPT2_CPP_DIR` overrides where they look for the dump files.

## Design decisions

**Flat storage, no strides.** A tensor is one contiguous buffer plus a shape vector, with no stride vector. `transpose` is therefore a real data copy rather than a view. The cost is extra copies; the benefit is that every index is computed explicitly, so indexing errors surface as wrong numbers in small tests. I considered adding strides after learning the concept, and didn't.

**Fused output head.** Logits are `x @ wte.T`, where `wte` is 50257 x 768. The transpose is never materialized: `logits[t][v]` is a dot product over `wte` row `v`, which reads the matrix contiguously and avoids duplicating 154 MB.

**Tokenization stays in Python.** Token IDs go in, token IDs come out. A BPE tokenizer in C++ is its own project and teaches nothing about transformers.

**Greedy sampling only.** It's deterministic, and determinism is what makes the integer-level comparison possible. Temperature and top-k live in the Python repo.

## Performance

The 5-token forward pass takes 1.675 seconds, including the 475 MB weight load. Ten greedy tokens take 23.5 seconds, about 2.35 s/token. That's `-O2`, single-threaded, on a Ryzen 5 5600.

Per generated token the twelve blocks are roughly 85M multiply-adds and the output head another 38M. At a sequence length of 1 each weight is read once and used once, which gives about 0.25 flops per byte: memory-bound. There is also no KV cache, so every step recomputes the whole sequence.

## Not implemented

Deliberately out of scope: BPE tokenization, KV caching, temperature and top-k sampling, multithreading, SIMD, batching, and anything beyond the 124M configuration.

The natural next step is a KV cache, which removes the quadratic recomputation. A verified reference implementation now exists to check it against, which is usually the hard part of optimization work.

## Writeup

WRITEUP.md covers what the port cost rather than what it is: the index arithmetic, how correctness was established, what each test was blind to, and why it runs at the speed it does.
