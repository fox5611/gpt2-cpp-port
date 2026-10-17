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

A 5-token forward pass takes 199 ms, and generating 10 tokens takes 4.1 s, compiled with `-O3 -march=native` and run single-threaded on a Ryzen 5 5600. Both figures are after the change described below; the first working version took 1156 ms and 23.5 s.

At a sequence length of one, each of the roughly 7 million weights in a transformer block is read from memory and used in exactly one multiply-add, which is why the implementation is memory-bound rather than compute-bound. The multiply-add itself is nearly free; the expensive part is moving the weights from memory to the processor, which cannot be done fast enough to keep it busy.

Imagine you had 7 million numbers and your job was to add 1 to each one and throw it away. That is what this implementation does, so nearly all of the time goes into fetching the next number, and almost none into the addition.

Per-token cost grows as generation continues because of how `generate` calls `gpt2_forward`. The `for (int step = 0; step < n_new; step++)` loop appends each new token to `token_ids` and passes the whole vector back in, and `gpt2_forward` processes that entire sequence from the embeddings up through all 12 blocks. No state is kept between steps, so every previous token's keys and values are recomputed every time.

From a 5-token prompt, step 1 processes 5 tokens and step 10 processes 14, so step 10 does about 2.8× the work of step 1, and the total cost is quadratic in the number of tokens generated. That also means a per-token average is the average of a rising curve rather than a steady rate, and it only holds at these sequence lengths; it says nothing about generating 500 tokens.

### Measuring one change

With a 5-token prompt on a single-threaded Ryzen 5 5600, the forward pass originally took 1156 ms. Reordering `matmul`'s loops from `ijk` to `ikj` brought it to 314.5 ms at `-O2`, and compiling with `-O3 -march=native` brought it to 199.3 ms. Ten-token generation went from 23.5 s to 4.142 s. Correctness was unchanged at every step: 2.73e-06 relative against the Python implementation, and identical token IDs.

The reorder is worth 3.67× on its own, from the access pattern alone. The second win is not what I expected. `-march=native` by itself does nothing, at 324.7 ms, marginally worse than plain `-O2`, while `-O3` alone is worth 1.45×, because `-O3` is what lets the cost model vectorize the inner loop at all. AVX2's wider registers then add 1.08× on top of that. Under `-O3` the loop vectorizes with 32-byte vectors; under `-O2` the cost model declines it. `__restrict__` and raw row pointers changed nothing, so aliasing was never the blocker, and switching from `at()` to raw indexing changed nothing either: the accessor was already being inlined away.

That forces a correction to what I wrote above. At a sequence length of one, each weight is read once and used once, about 0.25 flops per byte, and the memory-bound argument holds. But every measurement here is at n = 5, where each weight is used five times, about 1.25 flops per byte, and `-O3` bought 1.45× purely by improving arithmetic throughput, which cannot happen if the processor is starved for data. The n=1 claim stands; these measurements don't test it, and at the sequence lengths I actually benchmarked, compute mattered. Two other things above are also wrong now: transposing the weights at load time is unnecessary, since the reorder achieves the same access pattern with no extra copy, and I was wrong to expect layout to matter before SIMD.

The larger win is still untouched. A KV cache would make step n process one token instead of n, removing the quadratic term entirely, which is bigger than anything measured here. I did the layout fix first because it doesn't change what gets computed, only the order in which memory is traversed, so the existing cross-check validated it unchanged.
