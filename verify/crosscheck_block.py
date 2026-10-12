import numpy as np
from load_weights import load_weights
from model import transformer_block

W = load_weights('gpt2_weights.bin')
n, d_model = 4, 768
P = 'transformer.h.0.'

x = np.array([[((i * d_model + j) % 100) / 100.0 for j in range(d_model)]
              for i in range(n)], dtype=np.float32)

py = transformer_block(x,
        W[P + 'ln_1.weight'],        W[P + 'ln_1.bias'],
        W[P + 'attn.c_attn.weight'], W[P + 'attn.c_attn.bias'],
        W[P + 'attn.c_proj.weight'], W[P + 'attn.c_proj.bias'],
        W[P + 'ln_2.weight'],        W[P + 'ln_2.bias'],
        W[P + 'mlp.c_fc.weight'],    W[P + 'mlp.c_fc.bias'],
        W[P + 'mlp.c_proj.weight'],  W[P + 'mlp.c_proj.bias'])

cpp = np.loadtxt('/home/rthacker/ai/gpt2-cpp-port/dbg_block.txt',
                 dtype=np.float32).reshape(py.shape)
d = np.abs(cpp - np.asarray(py, dtype=np.float32))
print(f"block max diff {d.max():.8f}")
print(f"py abs max     {np.abs(py).max():.6f}")
print(f"relative       {d.max() / np.abs(py).max():.2e}")
