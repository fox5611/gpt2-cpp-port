# Porting GPT-2 to C++

This is a record of what the translation from numpy to C++ cost. The README says what the thing is; this says what it took to get there.

## Flat storage and index arithmetic

Flat storage with no strides turns every shape operation into explicit arithmetic. That arithmetic is where all of my bugs lived, and keeping it in the open is why they were findable.

Tensors here are a flat `std::vector<float>` plus a shape vector, with no stride vector. `at(i, j)` computes `i * cols + j`, so the stride lives inside the accessor. The rule is that each index is multiplied by the size of everything below it: for a shape `(D0, D1, D2)`, element `(i, j, k)` sits at `i*(D1*D2) + j*D2 + k`, and the last index is never multiplied. It took four rounds of drilling before I could derive those mappings cold.

Because there are no strides, `transpose` is a real data copy rather than a view, and one side of it is always strided: contiguous reads, scattered writes. It also means shape and index are different kinds of thing, and easy to confuse. `(12, n, 64)` says how many; `(h, t, d)` says which one. A tensor with 12 heads has indices 0 through 11, so `at(12, t, d)` walks off the end.

`split_qkv` is where this bit. Q, K and V are columns of a single `(n, 2304)` tensor, so Q's columns are not contiguous across rows, and they are interrupted by K and V every row. `memcpy(q.data(), qkv.data(), n*768*sizeof(float))` gets row 0 right and then fills row 1 with row 0's K data, and a test with `n = 1` cannot detect it. The correct offsets are `i*2304 + {0, 768, 1536} + j`: the stride comes from the source, the bounds from the destination.

The bug produced a wrong number rather than a crash, and because the test filled the tensor sequentially, each stored value equalled its own flat index, so a wrong output named the exact location in memory it had been read from. There was no layer between the bug and the source: the index expression is written out in the line, so there is one place to look, and it can be checked by hand. With a stride vector, the same mistake would have been a wrong value in a stride field, and I would have been debugging a layout instead of a line.

`split_heads` and `combine_heads` show the other side of this. Each mapping's inverse is obtained by swapping the two sides of the equation, not by reasoning about layout: `out.at(h,t,d) = x.at(t, h*64+d)` reverses to `out.at(t, h*64+d) = x.at(h,t,d)`. Every index on one side appears on the other, with nothing summed and nothing dropped.

That also hands you a free exact test: split, combine, and compare to the original byte for byte. A round trip only proves the two are mutual inverses, though, not that either is right, which is why `split_heads` needed independent verification.

## Verification

Every test I wrote was blind to something, and the blindness was always a property of the data, not the code.

A forward pass produces 50257 floats per token, and no single value tells you whether it's right. The bug that cost me the most time was syntactically valid C++, and every function it called had already been verified individually; reading the code cannot detect a discarded return value. So correctness has to come from comparison instead.

What you compare against matters. The Python implementation was verified component by component against PyTorch, to 1.1e-04 on a full forward pass, and the C++ was then verified against the Python. Nothing here is correct on its own; it is correct relative to something else that was checked, and each link in that chain was verified independently.

Both sides generate the same input from integer arithmetic, `((i*768+j)%100)/100.0`, so there is no file to parse and no rounding mismatch, and both run on real GPT-2 weights rather than synthetic ones.

Then each test's blind spot:

- **Sequential fill** catches stride errors, because every stored value equals its own flat index. Blind to scale.
- **Uniform fill** catches the causal mask. Blind to the `÷8`, because softmax is unchanged when you scale a set of equal values.
- **All-ones with zero biases** is blind to a dropped bias, since adding zero changes nothing. This is the one that bit.
- **A round trip** proves two functions are mutual inverses, not that either is correct.
- **`n = 1`** is blind to every row-stride error, because there is no second row to run into.

The pattern is that whatever the data holds constant, the test cannot see.

When the output didn't match, dumping eight intermediates from both sides (`qkv`, `q`, `qh0`, `scores0`, `weights0`, `gather0`, `combined`, `out`) found the first divergence in one round trip instead of six guesses.

Absolute thresholds are the wrong instrument. `mlp`'s absolute difference of 1.75e-04 would have failed a correct implementation; measured against values reaching 104, it was 1.7e-06. And the decisive check isn't a float comparison at all: identical token IDs.

## What the language did

In Python, most mistakes stop the program. Two bugs account for nearly all the time I lost, and both did the opposite: they compiled cleanly and produced plausible output, because nothing was wrong with the syntax. Meanwhile, the language objected loudly to things that cost me minutes.

Python's `x = x + a` rebinds a local name. In C++, `Tensor& x` is a reference to the caller's tensor, so `x = add(x, a)` overwrites their data instead. `transformer_block` uses `x1` and `x2` to avoid exactly that, and `generate` takes `token_ids` by value on purpose so it can append to its own copy.

The second bug was worse: `add_bias` returns a new `Tensor`, and `add_bias(qkv, c_attn_b);` is a legal expression statement that computes the result and discards it. It compiled without a warning, cost 0.25 absolute error against Python, and four unit tests passed anyway. `[[nodiscard]]` turns it into a warning; all 17 operations carry it now.

The loud failures were all trivial by comparison. `add_bias(matmul(x, w), b)` doesn't compile, because a non-const `Tensor&` can't bind to a temporary, which forced the named intermediates `qkv_raw` and `qkv`.

The headers are order-dependent: with no forward declarations, a function must appear after everything it calls, so `transformer_block` sits after `attention` and `mlp`.

And when I pasted a function twice and its closing brace fused to `#pragma once`, I got six errors, five of which were echoes of the first. The rule I took from that is to fix the first error and recompile instead of reading the rest.

The split is the point. C++ complained instantly about every mistake that was merely malformed, and said nothing about the two that quietly produced wrong numbers.

## Performance

> Superseded. The figures in this section are from October and describe the
> single-threaded implementation before the KV cache, batched prefill and the
> vectorised matmul. Current numbers are in "Performance work, 11 to 12 November
> 2026" at the end of this file. This section is kept because that later section
> corrects and builds on the reasoning here.

A 5-token forward pass takes 199 ms, and generating 10 tokens takes 4.1 s, compiled with `-O3 -march=native` and run single-threaded on a Ryzen 5 5600. Both figures are after the change described below; the first working version took 1156 ms and 23.5 s.

At a sequence length of one, each of the roughly 7 million weights in a transformer block is read from memory and used in exactly one multiply-add, which is why the implementation is memory-bound rather than compute-bound. The multiply-add itself is nearly free; the expensive part is moving the weights from memory to the processor, which cannot be done fast enough to keep it busy.

Imagine you had 7 million numbers and your job was to add 1 to each one and throw it away. That is what this implementation does, so nearly all of the time goes into fetching the next number, and almost none into the addition.

Per-token cost grows as generation continues because of how `generate` calls `gpt2_forward`. The `for (int step = 0; step < n_new; step++)` loop appends each new token to `token_ids` and passes the whole vector back in, and `gpt2_forward` processes that entire sequence from the embeddings up through all 12 blocks. No state is kept between steps, so every previous token's keys and values are recomputed every time.

From a 5-token prompt, step 1 processes 5 tokens and step 10 processes 14, so step 10 does about 2.8× the work of step 1, and the total cost is quadratic in the number of tokens generated. That also means a per-token average is the average of a rising curve rather than a steady rate, and it only holds at these sequence lengths; it says nothing about generating 500 tokens.

### Measuring one change

With a 5-token prompt on a single-threaded Ryzen 5 5600, the forward pass originally took 1156 ms. Reordering `matmul`'s loops from `ijk` to `ikj` brought it to 314.5 ms at `-O2`, and compiling with `-O3 -march=native` brought it to 199.3 ms. Ten-token generation went from 23.5 s to 4.142 s. Correctness was unchanged at every step: 2.73e-06 relative against the Python implementation, and identical token IDs.

The reorder is worth 3.67× on its own, from the access pattern alone. The second win is not what I expected. `-march=native` by itself does nothing, at 324.7 ms, marginally worse than plain `-O2`, while `-O3` alone is worth 1.45×, because `-O3` is what lets the cost model vectorize the inner loop at all. AVX2's wider registers then add 1.08× on top of that. Under `-O3` the loop vectorizes with 32-byte vectors; under `-O2` the cost model declines it. `__restrict__` and raw row pointers changed nothing, so aliasing was never the blocker, and switching from `at()` to raw indexing changed nothing either: the accessor was already being inlined away.

That forces a correction to what I wrote above. At a sequence length of one, each weight is read once and used once, about 0.25 flops per byte, and the memory-bound argument holds. But every measurement here is at n = 5, where each weight is used five times, about 1.25 flops per byte, and `-O3` bought 1.45× purely by improving arithmetic throughput, which cannot happen if the processor is starved for data. The n=1 claim stands; these measurements don't test it, and at the sequence lengths I actually benchmarked, compute mattered. Two other things above are also wrong now: transposing the weights at load time is unnecessary, since the reorder achieves the same access pattern with no extra copy, and I was wrong to expect layout to matter before SIMD.

The larger win was still untouched at this point. A KV cache would make step n process one token instead of n, removing the quadratic term entirely, which is bigger than anything measured here. I did the layout fix first because it doesn't change what gets computed, only the order in which memory is traversed, so the existing cross-check validated it unchanged. The KV cache landed afterwards, in `generate_cached`.

## Performance work, 11 to 12 November 2026

The main mistake was choosing optimizations before measuring the constraint. Three attempts preserved the results but made the kernel slower. `bench_roofline` and `bench_peak` narrowed down the problem, and the assembly showed where the work was going.

### Batched prefill

`generate_cached` walked the prompt one token at a time, calling `gpt2_forward_step` once per token. Each call hauled all 85M transformer parameters, 340 MB, across the bus to perform 170 MFLOP of work, about 0.5 flops per byte. I added `gpt2_forward_prefill` to do the whole prompt in one multi-row pass, filling the KV cache for every position. `attention_scores` already applied the causal mask, so `attention()` was already correct for n rows and needed no change. A 256-token prompt went from 4157 to 2292 ms.

The limit moved. Before: 340 MB in 15.0 ms is about 23 GB/s, memory-limited. After: 170 MFLOP in 7.4 ms is about 23 GFLOPS, close to the throughput of the existing matmul. Batching removed repeated weight traffic, but the compute kernel then limited the gain to around 2x.

### Attention access pattern

`attention_gather`'s inner loop ran over `j`, so each access to `v` advanced by `head_dim * 4` = 256 bytes. This was the same `ijk` pattern that cost `matmul` 3.67x before the original reorder. Before the change, both attention kernels also processed the full n by n range and discarded the upper triangle. Reordering the gather and skipping `j > i` took attention from 61.1 to 21.4 ms per block, 2.85x faster, of which about 2.0x came from work no longer done and about 1.4x from the access pattern. Prefill went from 2292 to 1820 ms. The per-token rise at n=256 largely disappeared, with 7.37 to 8.95 becoming 6.97 to 7.11, supporting the n-squared term as the cause.

### Roofline

`bench_roofline.cpp` fixes M=256 and K=256 and sweeps the size of `b`.

| size of b | GFLOPS |
| --- | --- |
| 64 KB | 25.54 |
| 256 KB | 25.98 |
| 1 MB | 28.86 |
| 4 MB | 27.75 |
| 16 MB | 22.21 |
| 64 MB | 11.24 |

Throughput was roughly flat across the 64x range from 64 KB to 4 MB, and stayed relatively close through 16 MB. The model's weights are 7.1 MB for qkv and 9.4 MB for both MLP matmuls. At those sizes there was no large throughput drop suggesting DRAM bandwidth was the main constraint. That pointed back to the generated code.

The 64 MB row streamed 16.8 GB of `b` in 764 ms, about 22 GB/s of single-core DRAM bandwidth. This refined the earlier 5 GB/s figure, which came from dependent loads and was primarily a latency measurement.

### Peak FMA throughput

`bench_peak.cpp` uses 8 independent `v8f` accumulators, zero memory traffic, and 500M iterations. It measured 110.3 GFLOPS, equivalent to about 3.45 GHz at 32 flops per cycle. The original matmul was at 28 to 32 GFLOPS, around 27 percent of peak. Together with the roofline sweep, this showed there was substantial compute throughput available that the matmul was not using.

### Three failures

| attempt | speedup across qkv, mlp_fc, mlp_proj |
| --- | --- |
| cache blocking, MC=64 NC=256 KC=128 | 0.74 / 0.74 / 0.88x |
| row-pair blocking, MR=2, accumulators as full rows in memory | 0.84 / 0.72 / 0.76x |
| 4x16 tile with `acc[4][16]` as an array | 0.98 / 0.81 / 0.95x |

The reported `max|d|` was 0.00 in all three cases. Each output element still accumulated `k` in ascending order, and the correctness checks passed. The attempts made the kernel slower.

The damage from cache blocking tracked how much the inner loop was shortened: N=768 chopped into 3 pieces lost the least at 0.88x, while N=3072 chopped into 12 lost the most at 0.74x. In the packed version, `mlp_fc` and `mlp_proj` both came out at 39.99 ms:

| matmul | tiles and depth | k-steps |
| --- | --- | --- |
| mlp_fc | 64 i-tiles x 192 j-tiles x K=768 | 9,437,184 |
| mlp_proj | 64 i-tiles x 48 j-tiles x K=3072 | 9,437,184 |

Identical k-step counts, identical measured times, despite the different layouts. This supported instruction overhead as the explanation for the result.

`-fopt-info-vec` reported `matmul_simple`'s inner loop as "vectorized using 32 byte vectors", so GCC was not failing to vectorize. The next step was checking what that vectorization actually produced.

### The assembly

GCC kept 4 of the 8 accumulators in `ymm2` to `ymm5`, but stored them to the stack every iteration. The other 4 were memory operands in `vfmadd213ps`, at `288(%rsp)`, `352(%rsp)`, `416(%rsp)` and `480(%rsp)`. The loop body had 8 FMAs, 2 `vmovups` loads of `b`, 4 `vbroadcastss`, and 8 `vmovaps` stores to `(%rsp)`. Register thrashing was visible too, with `vmovaps %ymm6, %ymm7` doing no work.

Zen 3 retires 1 store per cycle, so the 8 stores needed at least 8 cycles, while the 8 FMAs only needed 4. That imposes a ceiling of 1 vector FMA per cycle, or 16 flops per cycle. At 3.45 GHz this is 55.2 GFLOPS before accounting for the remaining instructions. It does not predict the measured 30 GFLOPS exactly, but it exposes stack traffic that limits the loop well below the measured compute peak.

The remaining gap between the 55 GFLOPS ceiling and the measured 30 may come from the memory-operand FMAs. `vfmadd213ps 288(%rsp)` reads a stack slot that the previous iteration wrote, so each of those four accumulators carries a dependency chain through store-to-load forwarding rather than through a register, which costs several cycles per link on top of the store-port limit. Untested: confirming it would need a variant that keeps all eight accumulators in registers but deliberately spills only four.

### The fix

I used `typedef float v8f __attribute__((vector_size(32)))` and replaced the accumulator array with 8 named accumulator variables, which kept the accumulators in registers. `b` is packed into a contiguous K by 16 panel first, and the inner loop became 2 loads of `b`, 4 broadcasts of `a`, 8 FMAs and zero stores.

| matmul | GFLOPS | speedup |
| --- | --- | --- |
| qkv | 100.11 | 3.46x |
| mlp_fc | 96.49 | 3.42x |
| mlp_proj | 102.61 | 3.24x |

`max|d|` stayed at 0.00. The best result was 93 percent of the 110.3 GFLOPS measured peak, with qkv at about 91 percent. Prefill went from 1820 to 833 ms.

### The regression it caused

M=1 fell into the leftover-rows loop but still paid for K by N packing across the column blocks. Decode went from 45.2 to 101.0 ms per token, 2.2x slower. `bench_kvcache`'s cached run took 1017 ms, and the speedup against the uncached path fell to 1.35x. I added the guard `if (M < MR) return matmul_simple(a, b);`. Decode returned to 45.0 ms in the direct comparison, with subsequent runs at 46.4 to 51.5 ms per token, and the cache speedup returned to 2.67x. This was only caught because `bench_kvcache` was in the list.

### attention_scores as a matmul

`attention_scores` took 13.955 ms at 3.62 GFLOPS. Its inner loop was a 64-term scalar reduction with about 4 cycles of FMA latency per link, roughly 256 cycles for 64 multiply-accumulates. But `scores[i][j] = sum_d q[i][d] * k[j][d]` is Q times K-transposed, so I transposed `k` to `(head_dim, n)` and passed it to the tiled kernel. Scores went from 13.955 to 1.417 ms, 9.85x faster. The kernel computes the full n by n matrix, so it does about 2x the arithmetic at about 20x the rate, around 71 GFLOPS on the full-matrix work. The triangle skip stays only in the gather.

Attention per block went from 21.207 to 8.616 ms.

| kernel | before | after |
| --- | --- | --- |
| attention_scores | 13.955 ms | 1.417 ms |
| softmax | 5.199 ms | 5.144 ms |
| attention_gather | 2.053 ms | 2.055 ms |

The gather was already at 24.6 GFLOPS, so I left it alone. Prefill went from 833 to 681 ms.

### Threading

The `j0` column-block loop was independent, with each worker owning its `bpack` and its output columns, so no locks were needed. Each output element was still computed by one thread with `k` ascending, and the reported `max|d|` remained 0.00.

| matmul | GFLOPS | against the original ikj loop |
| --- | --- | --- |
| qkv | 483.62 | 16.66x |
| mlp_fc | 500.16 | 17.69x |
| mlp_proj | 490.18 | 16.65x |

The `mlp_fc` thread sweep:

| threads | ms |
| --- | --- |
| 1 | 12.48 |
| 2 | 6.54 |
| 3 | 4.49 |
| 4 | 3.58 |
| 6 | 2.42 |
| 8 | 3.32 |
| 12 | 2.43 |

Six threads gave 5.15x, or 86 percent scaling efficiency. Eight threads were slower than six or twelve. Uneven sharing of the 6 physical cores is a plausible explanation: some workers share execution resources while others do not, so equal chunks finish at different times and the join waits for the slower workers. At 12, sharing can become more uniform. The sweep supports 6 threads as the default, though it does not establish the scheduling mechanism.

I set the default to `hardware_concurrency/2`. Matmuls below 16e6 multiply-accumulates stay single-threaded, which covers the 144 `attention_scores` calls per pass. Prefill went from 681 to 483 ms. The matmuls were about 412 of the 681 ms, so applying the measured 5.15x gives `269 + 412/5.15` = about 349 ms. The actual 483 ms is about 134 ms above that estimate. Threading overhead and different scaling across the matmul shapes can both contribute to the gap.

The 110.3 GFLOPS single-core peak was measured with the other five cores idle. Six times that is about 662 GFLOPS, and all-core clocks are lower. The 500 GFLOPS result is a large fraction of that reference, though a measured all-core peak would be needed to say how close it is to the whole-chip limit.

### Where it stands

| | before | after |
| --- | --- | --- |
| prefill, 256-token prompt | 4157 ms | 483 ms |
| per prompt token | 16.24 ms | 1.888 ms |
| matmul | about 29 GFLOPS | 500 GFLOPS |
| decode | about 47 ms per token | unchanged |

That is 8.6x on prefill and about 17x on the matmul. The profile is now roughly 80 ms of weight matmuls, 165 ms of elementwise work and copies, and 103 ms of attention. Attention is now about 60 percent softmax. Elementwise work is the largest single category, and none of it is threaded.

### Tooling notes

`test-backend-ops perf` uses a different case list from `test`. A `-p` filter matched nothing and printed `OK` with zero cases run. The benchmark GFLOPS figures went stale twice when a kernel's work changed. `nano` damaged `ops.h` three times, while anchored Python edits asserting the match count went clean every time. `-Wall` would have caught a missing `return` in `attention_scores`. Thread scheduling also made best-of-3 too noisy, with the per-token column becoming non-monotonic.

The useful lesson was to measure the constraint before choosing the optimization. The roofline sweep showed where memory bandwidth mattered, the peak benchmark showed how much compute was available, and the assembly exposed the stack traffic. Those measurements came after three failed attempts. Keeping decode in the benchmark list also caught a prefill improvement that had made another workload twice as slow.
