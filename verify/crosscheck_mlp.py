import os
BASE = os.environ.get('GPT2_CPP_DIR', '../gpt2-cpp-port/')
import numpy as np
from load_weights import load_weights
from model import mlp

W = load_weights('gpt2_weights.bin')
n, d_model = 4, 768

x = np.array([[((i * d_model + j) % 100) / 100.0 for j in range(d_model)]
              for i in range(n)], dtype=np.float32)

py = mlp(x,
         W['transformer.h.0.mlp.c_fc.weight'],
         W['transformer.h.0.mlp.c_fc.bias'],
         W['transformer.h.0.mlp.c_proj.weight'],
         W['transformer.h.0.mlp.c_proj.bias'])

cpp = np.loadtxt(BASE + 'dbg_mlp.txt',
                 dtype=np.float32).reshape(py.shape)
d = np.abs(cpp - np.asarray(py, dtype=np.float32))
print(f"mlp max diff {d.max():.8f}")
print(f"py abs max   {np.abs(py).max():.6f}")
print(f"mean diff    {d.mean():.8f}")
print(f"relative     {d.max() / np.abs(py).max():.2e}")
