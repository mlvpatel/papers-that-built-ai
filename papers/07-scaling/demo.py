"""Step 07: loss falls predictably with size, and a fixed budget has a best size.

Papers: Kaplan et al., "Scaling Laws for Neural Language Models" (2020),
https://arxiv.org/abs/2001.08361; Hoffmann et al., "Training Compute-Optimal
Large Language Models" (Chinchilla, 2022), https://arxiv.org/abs/2203.15556

Part 1 (Kaplan): train five sizes of the same character-level Transformer on
Tiny Shakespeare for the same number of steps and compare validation loss.
Part 2 (Chinchilla): give every size the same compute, counted as parameters
times characters seen, so small models train longer and big ones shorter, and
see which size ends up best. Both curves are saved to scaling.png, and to
scaling-dark.png for pages shown in dark mode.

Run it from the repo root:
    python papers/07-scaling/demo.py            # full run, laptop GPU
    python papers/07-scaling/demo.py --quick    # fewer steps, for a CPU
    python papers/07-scaling/demo.py --smoke    # a few steps, used by the tests
"""

import argparse
import io
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from matplotlib import style
from matplotlib.figure import Figure
from torch import nn

DATA = Path(__file__).resolve().parents[2] / "data" / "tinyshakespeare.txt"
HERE = Path(__file__).resolve().parent
# matplotlib style -> file; the page shows the dark one in dark mode
PLOTS = {"default": HERE / "scaling.png", "dark_background": HERE / "scaling-dark.png"}
WIDTHS, LAYERS, CONTEXT, BATCH, LR = [32, 64, 96, 128, 192], 4, 64, 64, 1e-3
STEPS = {"full": 1000, "quick": 250, "smoke": 2}
BUDGET_WIDTH = 64  # the fixed compute budget is this width's run in part 1

GPUS = [("mps", torch.backends.mps.is_available()), ("cuda", torch.cuda.is_available())]


class Block(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.width, self.heads = width, width // 32
        self.ln1, self.ln2 = nn.LayerNorm(width), nn.LayerNorm(width)
        self.qkv, self.proj = nn.Linear(width, 3 * width), nn.Linear(width, width)
        self.mlp = nn.Sequential(
            nn.Linear(width, 4 * width), nn.GELU(), nn.Linear(4 * width, width)
        )

    def forward(self, x):
        b, t, _ = x.shape
        qkv = self.qkv(self.ln1(x)).view(b, t, 3, self.heads, self.width // self.heads)
        out = F.scaled_dot_product_attention(
            *qkv.permute(2, 0, 3, 1, 4), is_causal=True
        )
        x = x + self.proj(out.transpose(1, 2).reshape(b, t, self.width))
        return x + self.mlp(self.ln2(x))


class CharModel(nn.Module):
    def __init__(self, vocab, width):
        super().__init__()
        self.tok, self.pos = nn.Embedding(vocab, width), nn.Embedding(CONTEXT, width)
        self.blocks = nn.Sequential(*[Block(width) for _ in range(LAYERS)])
        self.ln, self.head = nn.LayerNorm(width), nn.Linear(width, vocab)

    def forward(self, x):
        h = self.tok(x) + self.pos(torch.arange(x.size(1), device=x.device))
        return self.head(self.ln(self.blocks(h)))


def size(vocab, width):
    """Parameters outside the embeddings, the count Kaplan's laws use."""
    return sum(p.numel() for p in CharModel(vocab, width).blocks.parameters())


def get_batch(data, device):
    ix = torch.randint(len(data) - CONTEXT - 1, (BATCH,))
    x = torch.stack([data[i : i + CONTEXT] for i in ix])
    y = torch.stack([data[i + 1 : i + CONTEXT + 1] for i in ix])
    return x.to(device), y.to(device)


def val_loss_after(width, steps, splits, vocab, device, eval_batches):
    train, val = splits
    torch.manual_seed(0)
    model = CharModel(vocab, width).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=LR)
    for _ in range(steps):
        x, y = get_batch(train, device)
        loss = F.cross_entropy(model(x).flatten(0, 1), y.flatten())
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    model.eval()
    with torch.no_grad():
        losses = [
            F.cross_entropy(model(x).flatten(0, 1), y.flatten()).item()
            for x, y in (get_batch(val, device) for _ in range(eval_batches))
        ]
    return sum(losses) / len(losses)


def plot(sizes, fixed_steps, fixed_compute, steps, targets):
    for theme, target in targets.items():
        with style.context(theme):
            fig = Figure(figsize=(10, 4))
            left, right = fig.subplots(1, 2, sharey=True)
            for ax, losses, title in [
                (left, fixed_steps, f"Same steps ({steps} each)"),
                (right, fixed_compute, "Same compute (parameters x characters)"),
            ]:
                ax.plot(sizes, losses, "o-")
                ax.set(
                    xscale="log", xlabel="parameters (without embeddings)", title=title
                )
            left.set_ylabel("validation loss")
            fig.tight_layout()
            # the dark copy keeps the page background showing through
            dark = theme != "default"
            fig.savefig(target, format="png", dpi=120, transparent=dark)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    speed = p.add_mutually_exclusive_group()
    speed.add_argument(
        "--quick", action="store_true", help="250 steps per size, for a CPU"
    )
    speed.add_argument(
        "--smoke", action="store_true", help="2 steps per size, for the tests"
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
    steps, eval_batches = STEPS[mode], 1 if args.smoke else 20
    t0 = time.time()

    text = DATA.read_text(encoding="utf-8")
    chars = sorted(set(text))
    stoi = {c: i for i, c in enumerate(chars)}
    data = torch.tensor([stoi[c] for c in text])
    split = int(0.9 * len(data))
    splits, vocab = (data[:split], data[split:]), len(chars)
    sizes = [size(vocab, w) for w in WIDTHS]
    budget = size(vocab, BUDGET_WIDTH) * steps  # parameters x steps

    runs = {}  # (width, steps) -> validation loss; equal runs are trained once

    def loss(width, n_steps):
        if (width, n_steps) not in runs:
            runs[width, n_steps] = val_loss_after(
                width, n_steps, splits, vocab, args.device, eval_batches
            )
        return runs[width, n_steps]

    print(
        f"{'width':>5} {'params':>9} | same steps: {'steps':>5} {'loss':>6}"
        f" | same compute: {'steps':>5} {'loss':>6}"
    )
    fixed_steps, fixed_compute = [], []
    for width, n in zip(WIDTHS, sizes):
        budget_steps = max(1, round(budget / n))  # smaller models get more steps
        print(f"  training width {width}...", file=sys.stderr, flush=True)
        fixed_steps.append(loss(width, steps))
        fixed_compute.append(loss(width, budget_steps))
        print(
            f"{width:5d} {n:9,d} | {'':12}{steps:5d} {fixed_steps[-1]:6.3f}"
            f" | {'':14}{budget_steps:5d} {fixed_compute[-1]:6.3f}",
            flush=True,
        )

    best = sizes[fixed_compute.index(min(fixed_compute))]
    plot(
        sizes,
        fixed_steps,
        fixed_compute,
        steps,
        PLOTS if mode == "full" else {theme: io.BytesIO() for theme in PLOTS},
    )
    print(
        f"RESULT before={fixed_steps[0]:.2f} after={fixed_steps[-1]:.2f} "
        f"seconds={time.time() - t0:.0f} device={args.device} "
        f"best_size_at_fixed_compute={best}"
    )


if __name__ == "__main__":
    main()
