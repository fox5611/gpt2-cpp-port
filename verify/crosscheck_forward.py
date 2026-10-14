import os
BASE = os.environ.get('GPT2_CPP_DIR', '../gpt2-cpp-port/')
import numpy as np
from load_weights import load_weights
from model import gpt2_forward

W = load_weights('gpt2_weights.bin')
token_ids = [464, 3139, 286, 4881, 318]

py = gpt2_forward(token_ids, W)
cpp = np.loadtxt(BASE + 'dbg_logits.txt',
                 dtype=np.float32).reshape(py.shape)

d = np.abs(cpp - np.asarray(py, dtype=np.float32))
print(f"shape        {py.shape}")
print(f"max diff     {d.max():.8f}")
print(f"py abs max   {np.abs(py).max():.6f}")
print(f"relative     {d.max() / np.abs(py).max():.2e}")
print(f"argmax py    {int(py[-1].argmax())}")
print(f"argmax cpp   {int(cpp[-1].argmax())}")
