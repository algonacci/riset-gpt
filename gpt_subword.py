import json
import random

import torch
import torch.nn as nn
from torch.nn import functional as F
import tiktoken

# Small GPT experiment with GPT-4-era BPE subword tokens.
batch_size = 4
block_size = 16
max_iters = 1000
eval_interval = 100
eval_iters = 20
learning_rate = 3e-4
n_embd = 128
n_head = 4
n_layer = 4
dropout = 0.2
device = 'cuda' if torch.cuda.is_available() else 'cpu'

torch.manual_seed(1337)
# enc = tiktoken.get_encoding('cl100k_base')
enc = tiktoken.get_encoding('gpt2')
with open('indo_input.jsonl', 'r', encoding='utf-8') as f:
    records = [json.loads(line)['text'] for line in f if line.strip()]
random.Random(1337).shuffle(records)
record_split = int(0.9 * len(records))
train_text = '\n'.join(records[:record_split])
val_text = '\n'.join(records[record_split:])
text = train_text + '\n' + val_text
vocab_size = enc.n_vocab
train_data = torch.tensor(enc.encode(train_text), dtype=torch.long)
val_data = torch.tensor(enc.encode(val_text), dtype=torch.long)

def decode(ids):
    return enc.decode(ids)

def get_batch(split):
    source = train_data if split == 'train' else val_data
    if len(source) <= block_size:
        raise ValueError(f'{split} dataset terlalu pendek: {len(source)} token')
    ix = torch.randint(len(source) - block_size, (batch_size,))
    x = torch.stack([source[i:i + block_size] for i in ix]).to(device)
    y = torch.stack([source[i + 1:i + block_size + 1] for i in ix]).to(device)
    return x, y

class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        k, q, v = self.key(x), self.query(x), self.value(x)
        wei = q @ k.transpose(-2, -1) * k.shape[-1] ** -0.5
        wei = wei.masked_fill(self.tril[:x.size(1), :x.size(1)] == 0, float('-inf'))
        return self.dropout(F.softmax(wei, dim=-1)) @ v

class MultiHeadAttention(nn.Module):
    def __init__(self):
        super().__init__()
        hs = n_embd // n_head
        self.heads = nn.ModuleList([Head(hs) for _ in range(n_head)])
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(self.proj(torch.cat([h(x) for h in self.heads], dim=-1)))

class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.sa = MultiHeadAttention()
        self.ffwd = nn.Sequential(nn.Linear(n_embd, 4 * n_embd), nn.ReLU(),
                                  nn.Linear(4 * n_embd, n_embd), nn.Dropout(dropout))
        self.ln1, self.ln2 = nn.LayerNorm(n_embd), nn.LayerNorm(n_embd)

    def forward(self, x):
        return x + self.sa(self.ln1(x)) + self.ffwd(self.ln2(x))

class GPTLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block() for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, idx, targets=None):
        _, T = idx.shape
        x = self.token_embedding_table(idx) + self.position_embedding_table(torch.arange(T, device=device))
        logits = self.lm_head(self.ln_f(self.blocks(x)))
        loss = None if targets is None else F.cross_entropy(logits.view(-1, vocab_size), targets.view(-1))
        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            logits, _ = self(idx[:, -block_size:])
            idx = torch.cat((idx, torch.multinomial(F.softmax(logits[:, -1], dim=-1), 1)), dim=1)
        return idx

model = GPTLanguageModel().to(device)
print(sum(p.numel() for p in model.parameters()) / 1e6, 'M parameters')
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

@torch.no_grad()
def estimate_loss():
    model.eval()
    result = {}
    for split in ('train', 'val'):
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            _, loss = model(*get_batch(split))
            losses[k] = loss.item()
        result[split] = losses.mean()
    model.train()
    return result

for step in range(max_iters):
    if step % eval_interval == 0 or step == max_iters - 1:
        losses = estimate_loss()
        print(f"step {step}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
    xb, yb = get_batch('train')
    _, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

prompt = input('Masukkan prompt (Enter untuk generate dari awal): ')
if prompt:
    context = torch.tensor([enc.encode(prompt)], dtype=torch.long, device=device)
else:
    context = torch.zeros((1, 1), dtype=torch.long, device=device)

print(decode(model.generate(context, 200)[0].tolist()))
