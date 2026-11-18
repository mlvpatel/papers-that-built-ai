"""Step 05: ResNet's skip connections let a deep model keep learning.

Paper: He, Zhang, Ren, Sun, "Deep Residual Learning for Image Recognition" (2015),
https://arxiv.org/abs/1512.03385

The demo trains one small character-level Transformer on Tiny Shakespeare three
times: 2 blocks without skip connections, 24 blocks without them, and 24 blocks
with them. Same seed, data and optimizer each time. The deep model without skip
connections stops learning; adding x back at every block fixes it.

Run it from the repo root:
    python papers/05-resnet/demo.py            # full run, laptop GPU
    python papers/05-resnet/demo.py --quick    # fewer steps, for a CPU
    python papers/05-resnet/demo.py --smoke    # a few steps, used by the tests
"""

import argparse
import math
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

DATA = Path(__file__).resolve().parents[2] / "data" / "tinyshakespeare.txt"
WIDTH, HEADS, CONTEXT, BATCH, LR = 128, 4, 64, 64, 1e-3
STEPS = {"full": 1000, "quick": 250, "smoke": 2}
RUNS = [("shallow", 2, False), ("before", 24, False), ("after", 24, True)]

GPUS = [("mps", torch.backends.mps.is_available()), ("cuda", torch.cuda.is_available())]


class Block(nn.Module):
    def __init__(self, skip):
        super().__init__()
        self.skip = skip
        self.ln1, self.ln2 = nn.LayerNorm(WIDTH), nn.LayerNorm(WIDTH)
        self.qkv, self.proj = nn.Linear(WIDTH, 3 * WIDTH), nn.Linear(WIDTH, WIDTH)
        self.mlp = nn.Sequential(
            nn.Linear(WIDTH, 4 * WIDTH), nn.GELU(), nn.Linear(4 * WIDTH, WIDTH)
        )

    def attn(self, x):
        b, t, _ = x.shape
        qkv = self.qkv(x).view(b, t, 3, HEADS, WIDTH // HEADS).permute(2, 0, 3, 1, 4)
        out = F.scaled_dot_product_attention(*qkv, is_causal=True)
        return self.proj(out.transpose(1, 2).reshape(b, t, WIDTH))

    def forward(self, x):
        if self.skip:  # ResNet: add to x instead of replacing it
            x = x + self.attn(self.ln1(x))
            return x + self.mlp(self.ln2(x))
        x = self.attn(self.ln1(x))
        return self.mlp(self.ln2(x))


class CharModel(nn.Module):
    def __init__(self, vocab, layers, skip):
        super().__init__()
        self.tok, self.pos = nn.Embedding(vocab, WIDTH), nn.Embedding(CONTEXT, WIDTH)
        self.blocks = nn.Sequential(*[Block(skip) for _ in range(layers)])
        self.ln, self.head = nn.LayerNorm(WIDTH), nn.Linear(WIDTH, vocab)

    def forward(self, x):
        h = self.tok(x) + self.pos(torch.arange(x.size(1), device=x.device))
        return self.head(self.ln(self.blocks(h)))


def get_batch(data, device):
    ix = torch.randint(len(data) - CONTEXT - 1, (BATCH,))
    x = torch.stack([data[i : i + CONTEXT] for i in ix])
    y = torch.stack([data[i + 1 : i + CONTEXT + 1] for i in ix])
    return x.to(device), y.to(device)


def loss_on(model, x, y):
    return F.cross_entropy(model(x).flatten(0, 1), y.flatten())


@torch.no_grad()
def val_loss(model, val, device, batches):
    model.eval()
    total = sum(loss_on(model, *get_batch(val, device)).item() for _ in range(batches))
    model.train()
    return total / batches


def train(name, layers, skip, splits, vocab, steps, device, eval_batches):
    train_data, val_data = splits
    torch.manual_seed(0)  # same start for every run; only depth and skip differ
    model = CharModel(vocab, layers, skip).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=LR)
    label = f"{name:7} {layers:2d} blocks, skip={skip!s:5}"
    for step in range(steps):
        if step % 250 == 0:
            loss = val_loss(model, val_data, device, eval_batches)
            print(f"{label}  step {step:4d}  val loss {loss:.3f}", flush=True)
        loss = loss_on(model, *get_batch(train_data, device))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    final = val_loss(model, val_data, device, eval_batches)
    print(f"{label}  final      val loss {final:.3f}", flush=True)
    return final


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    speed = p.add_mutually_exclusive_group()
    speed.add_argument(
        "--quick", action="store_true", help="250 steps per run, for a CPU"
    )
    speed.add_argument(
        "--smoke", action="store_true", help="2 steps per run, for tests"
    )
    devices = [d for d, ok in GPUS if ok] + ["cpu"]  # the ones this machine has
    p.add_argument(
        "--device",
        default=devices[0],
        choices=devices,
        help="where to run (default: a GPU if found, else cpu)",
    )
    args = p.parse_args(argv)
    mode = "smoke" if args.smoke else "quick" if args.quick else "full"
    eval_batches = 1 if args.smoke else 20

    text = DATA.read_text(encoding="utf-8")
    chars = sorted(set(text))
    stoi = {c: i for i, c in enumerate(chars)}
    data = torch.tensor([stoi[c] for c in text])
    split = int(0.9 * len(data))
    splits = (data[:split], data[split:])
    print(
        f"{len(data):,} characters, {len(chars)} distinct; "
        f"random guessing scores {math.log(len(chars)):.2f}"
    )

    t0 = time.time()
    loss = {
        name: train(
            name,
            layers,
            skip,
            splits,
            len(chars),
            STEPS[mode],
            args.device,
            eval_batches,
        )
        for name, layers, skip in RUNS
    }
    print(
        f"RESULT before={loss['before']:.2f} after={loss['after']:.2f} "
        f"seconds={time.time() - t0:.0f} device={args.device} "
        f"shallow={loss['shallow']:.2f}"
    )


if __name__ == "__main__":
    main()
