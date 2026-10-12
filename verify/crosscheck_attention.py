import numpy as np
from load_weights import load_weights
from model import qkv_projection, split_heads, attention_scores, causal_mask, softmax
from model import attention as py_attention
outfn = py_attention(x, caw, cab, cpw, cpb)
cmp('out_fn', outfn, 'outfn.txt')
W = load_weights('gpt2_weights.bin')
n, d_model, n_heads, head_dim = 4, 768, 12, 64

x = np.array([[((i * d_model + j) % 100) / 100.0 for j in range(d_model)]
              for i in range(n)], dtype=np.float32)

caw = W['transformer.h.0.attn.c_attn.weight']
cab = W['transformer.h.0.attn.c_attn.bias']
cpw = W['transformer.h.0.attn.c_proj.weight']
cpb = W['transformer.h.0.attn.c_proj.bias']

D = '/home/rthacker/ai/gpt2-cpp-port/dbg_'

def cmp(name, py, fname):
    cpp = np.loadtxt(D + fname, dtype=np.float32).reshape(py.shape)
    d = np.abs(cpp - np.asarray(py, dtype=np.float32))
    print(f"{name:10s} max diff {d.max():.8f}")

qkv = x @ caw + cab
cmp('qkv', qkv, 'qkv.txt')

q, k, v = qkv_projection(x, caw, cab)
cmp('q', q, 'q.txt')

qh, kh, vh = split_heads(q), split_heads(k), split_heads(v)
cmp('qh0', qh[0], 'qh0.txt')

scores = causal_mask(attention_scores(qh, kh))
cmp('scores0', scores[0], 'scores0.txt')

weights = softmax(scores)
cmp('weights0', weights[0], 'weights0.txt')

gathered = weights @ vh
cmp('gather0', gathered[0], 'gather0.txt')

combined = gathered.transpose(1, 0, 2).reshape(n, -1)
cmp('combined', combined, 'combined.txt')

out = combined @ cpw + cpb
cmp('out', out, 'out.txt')
