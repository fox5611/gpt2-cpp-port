import os
BASE = os.environ.get('GPT2_CPP_DIR', '../gpt2-cpp-port/')
from load_weights import load_weights
from model import gpt2_forward

W = load_weights('gpt2_weights.bin')
ids = [464, 3139, 286, 4881, 318]

for _ in range(10):
    logits = gpt2_forward(ids, W)
    ids.append(int(logits[-1].argmax()))

print(' '.join(str(i) for i in ids))
