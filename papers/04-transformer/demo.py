"""Step 04: the Transformer replaces recurrence with attention.

Paper: Vaswani, Shazeer, Parmar, Uszkoreit, Jones, Gomez, Kaiser, Polosukhin,
"Attention Is All You Need" (2017), https://arxiv.org/abs/1706.03762

The demo trains two character models of about the same size on Tiny
Shakespeare with the same text, seed and steps: a recurrent network (GRU) that
reads one character at a time, and a 2-block Transformer that looks at every
earlier character at once through attention. It reports validation loss and
training time for both.

Run it from the repo root:
    python papers/04-transformer/demo.py            # full run, laptop GPU
    python papers/04-transformer/demo.py --quick    # fewer steps, for a CPU
    python papers/04-transformer/demo.py --smoke    # a few steps, used by tests
"""

import argparse
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

DATA = Path(__file__).resolve().parents[2] / "data" / "tinyshakespeare.txt"
CONTEXT, BATCH, LR = 64, 64, 1e-3
WIDTH, HEADS, LAYERS, GRU_HIDDEN = 128, 4, 2, 256  # about 0.4M parameters each
STEPS = {"full": 1500, "quick": 300, "smoke": 2}

GPUS = [("mps", torch.backends.mps.is_available()), ("cuda", torch.cuda.is_available())]


class CharGRU(nn.Module):
    def __init__(self, vocab):
        super().__init__()
        self.tok = nn.Embedding(vocab, GRU_HIDDEN)
        self.gru = nn.GRU(GRU_HIDDEN, GRU_HIDDEN, batch_first=True)
        self.head = nn.Linear(GRU_HIDDEN, vocab)

    def forward(self, x):
        out, _ = self.gru(self.tok(x))  # one character after another
        return self.head(out)


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(WIDTH), nn.LayerNorm(WIDTH)
        self.qkv, self.proj = nn.Linear(WIDTH, 3 * WIDTH), nn.Linear(WIDTH, WIDTH)
        self.mlp = nn.Sequential(
            nn.Linear(WIDTH, 4 * WIDTH), nn.GELU(), nn.Linear(4 * WIDTH, WIDTH)
        )

    def forward(self, x):
        b, t, _ = x.shape
        q, k, v = self.qkv(self.ln1(x)).view(b, t, 3, HEADS, WIDTH // HEADS).unbind(2)
        # Transformer: every position attends to all earlier positions at once
        out = F.scaled_dot_product_attention(
            q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), is_causal=True
        )
        x = x + self.proj(out.transpose(1, 2).reshape(b, t, WIDTH))
        return x + self.mlp(self.ln2(x))


class CharTransformer(nn.Module):
    def __init__(self, vocab):
        super().__init__()
        self.tok, self.pos = nn.Embedding(vocab, WIDTH), nn.Embedding(CONTEXT, WIDTH)
        self.blocks = nn.Sequential(*[Block() for _ in range(LAYERS)])
        self.ln, self.head = nn.LayerNorm(WIDTH), nn.Linear(WIDTH, vocab)

    def forward(self, x):
        h = self.tok(x) + self.pos(torch.arange(x.size(1), device=x.device))
        return self.head(self.ln(self.blocks(h)))


def get_batch(data, device):
    ix = torch.randint(len(data) - CONTEXT - 1, (BATCH,))
    x = torch.stack([data[i : i + CONTEXT] for i in ix])
    y = torch.stack([data[i + 1 : i + CONTEXT + 1] for i in ix])
    return x.to(device), y.to(device)


def train(name, model, splits, steps, device, eval_batches):
    train_data, val_data = splits
    opt = torch.optim.AdamW(model.parameters(), lr=LR)
    params = sum(p.numel() for p in model.parameters())
    t0 = time.time()
    for step in range(steps):
        x, y = get_batch(train_data, device)
        loss = F.cross_entropy(model(x).flatten(0, 1), y.flatten())
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if step % 500 == 0:
            print(f"  {name:11} step {step:4d}  loss {loss.item():.3f}", flush=True)
    if device == "mps":
        torch.mps.synchronize()
    elif device == "cuda":
        torch.cuda.synchronize()
    seconds = time.time() - t0
    model.eval()
    with torch.no_grad():
        losses = [
            F.cross_entropy(model(x).flatten(0, 1), y.flatten()).item()
            for x, y in (get_batch(val_data, device) for _ in range(eval_batches))
        ]
    val = sum(losses) / len(losses)
    print(
        f"{name:11} {params:,} parameters, val loss {val:.3f}, trained in {seconds:.0f} s"
    )
    return val, seconds


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    speed = p.add_mutually_exclusive_group()
    speed.add_argument(
        "--quick", action="store_true", help="300 steps per model, for a CPU"
    )
    speed.add_argument("--smoke", action="store_true", help="2 steps each, for tests")
    devices = [d for d, ok in GPUS if ok] + ["cpu"]  # the ones this machine has
    p.add_argument(
        "--device",
        default=devices[0],
        choices=devices,
        help="where to run (default: a GPU if found, else cpu)",
    )
    args = p.parse_args(argv)
    steps = STEPS["smoke" if args.smoke else "quick" if args.quick else "full"]
    eval_batches = 1 if args.smoke else 20
    t0 = time.time()

    text = DATA.read_text(encoding="utf-8")
    chars = sorted(set(text))
    stoi = {c: i for i, c in enumerate(chars)}
    data = torch.tensor([stoi[c] for c in text])
    split = int(0.9 * len(data))
    splits = (data[:split], data[split:])

    results = {}
    for name, cls in [("GRU", CharGRU), ("Transformer", CharTransformer)]:
        torch.manual_seed(0)  # same seed; batches differ, init uses the RNG differently
        model = cls(len(chars)).to(args.device)
        results[name] = train(name, model, splits, steps, args.device, eval_batches)

    (gru, gru_s), (tf, tf_s) = results["GRU"], results["Transformer"]
    print(
        f"RESULT before={gru:.2f} after={tf:.2f} seconds={time.time() - t0:.0f} "
        f"device={args.device} gru_seconds={gru_s:.0f} transformer_seconds={tf_s:.0f}"
    )


if __name__ == "__main__":
    main()
