"""Step 06: pre-training on lots of raw text teaches a model the language.

Papers: Radford, Narasimhan, Salimans, Sutskever, "Improving Language
Understanding by Generative Pre-Training" (GPT-1, 2018); Devlin, Chang, Lee,
Toutanova, "BERT: Pre-training of Deep Bidirectional Transformers for Language
Understanding" (2018), https://arxiv.org/abs/1810.04805

Task: given a line of Shakespeare and the same words in shuffled order, which
one is real? Lines are lowercased with punctuation removed, so word order is
the only clue. A model answers by picking the order it finds more likely (lower
next-character loss), the zero-shot scoring GPT-1 uses. Model A learns from the
300 task lines only. Model B, the same network from the same seed and with the
same number of steps, learns from the whole text with every test line cut out.

Run it from the repo root:
    python papers/06-pretraining/demo.py            # full run, laptop GPU
    python papers/06-pretraining/demo.py --quick    # fewer steps, for a CPU
    python papers/06-pretraining/demo.py --smoke    # a few steps, used by tests
"""

import argparse
import random
import re
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

DATA = Path(__file__).resolve().parents[2] / "data" / "tinyshakespeare.txt"
WIDTH, HEADS, LAYERS, CONTEXT, BATCH = 128, 4, 4, 64, 64
TRAIN_LINES, TEST_LINES, MIN_WORDS = 300, 500, 6
STEPS = {"full": 3000, "quick": 600, "smoke": 2}

GPUS = [("mps", torch.backends.mps.is_available()), ("cuda", torch.cuda.is_available())]


def normalize(text):
    """Lowercase, keep letters, apostrophes and newlines, one space between words."""
    text = re.sub(r"[^a-z'\n]+", " ", text.lower())
    return re.sub(r" *\n *", "\n", text)


def shuffled(line, rng):
    words = line.split()
    while True:
        rng.shuffle(words)
        if " ".join(words) != line:
            return " ".join(words)


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
        qkv = self.qkv(self.ln1(x)).view(b, t, 3, HEADS, WIDTH // HEADS)
        out = F.scaled_dot_product_attention(
            *qkv.permute(2, 0, 3, 1, 4), is_causal=True
        )
        x = x + self.proj(out.transpose(1, 2).reshape(b, t, WIDTH))
        return x + self.mlp(self.ln2(x))


class CharModel(nn.Module):
    def __init__(self, vocab):
        super().__init__()
        self.tok, self.pos = nn.Embedding(vocab, WIDTH), nn.Embedding(CONTEXT, WIDTH)
        self.blocks = nn.Sequential(*[Block() for _ in range(LAYERS)])
        self.ln, self.head = nn.LayerNorm(WIDTH), nn.Linear(WIDTH, vocab)

    def forward(self, x):
        h = self.tok(x) + self.pos(torch.arange(x.size(1), device=x.device))
        return self.head(self.ln(self.blocks(h)))


def train(model, data, steps, device, name):
    # GPT-1's pre-training objective: predict the next character of raw text
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    for step in range(steps):
        ix = torch.randint(len(data) - CONTEXT - 1, (BATCH,))
        x = torch.stack([data[i : i + CONTEXT] for i in ix]).to(device)
        y = torch.stack([data[i + 1 : i + CONTEXT + 1] for i in ix]).to(device)
        loss = F.cross_entropy(model(x).flatten(0, 1), y.flatten())
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if step % 1000 == 0 or step == steps - 1:
            print(f"  {name}  step {step:4d}  loss {loss.item():.3f}", flush=True)


@torch.no_grad()
def picks_real(model, pairs, stoi, device):
    """Percent of pairs where the real line gets the lower next-character loss."""
    model.eval()
    lines = [line for pair in pairs for line in pair]
    ids = torch.full((len(lines), CONTEXT + 1), -1)
    for i, line in enumerate(lines):
        ids[i, : len(line) + 1] = torch.tensor([stoi[c] for c in "\n" + line])
    x, y = ids[:, :-1].clamp(min=0).to(device), ids[:, 1:].to(device)
    loss = F.cross_entropy(
        model(x).transpose(1, 2), y, ignore_index=-1, reduction="none"
    )
    per_line = loss.sum(1) / (y != -1).sum(1)
    real, fake = per_line[0::2], per_line[1::2]
    model.train()
    return 100 * (real < fake).float().mean().item()


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument(
        "--quick", action="store_true", help="600 steps per model, for a CPU"
    )
    p.add_argument("--smoke", action="store_true", help="a few steps, for tests")
    devices = [d for d, ok in GPUS if ok] + ["cpu"]  # the ones this machine has
    p.add_argument(
        "--device",
        default=devices[0],
        choices=devices,
        help="where to run (default: a GPU if found, else cpu)",
    )
    args = p.parse_args(argv)
    steps = STEPS["smoke" if args.smoke else "quick" if args.quick else "full"]
    t0 = time.time()

    text = normalize(DATA.read_text(encoding="utf-8"))
    lines = sorted(
        {
            ln
            for ln in text.split("\n")
            if len(ln.split()) >= MIN_WORDS and len(ln) <= CONTEXT
        }
    )
    # keep out the test lines of the earlier fine-tuning attempt (see README)
    earlier = set(random.Random(0).sample(lines, 800)[:TEST_LINES])
    picked = random.Random(1).sample([ln for ln in lines if ln not in earlier], 800)
    test_lines, train_lines = picked[:TEST_LINES], picked[TEST_LINES:]
    rng = random.Random(2)
    pairs = [(line, shuffled(line, rng)) for line in test_lines]

    corpus = text
    for line in test_lines:  # test lines never appear in model B's text
        corpus = corpus.replace(line, "")
    assert not any(line in corpus for line in test_lines)
    print(f"{len(pairs)} test pairs; chance is 50%")
    print(f"  real:     {pairs[0][0]}\n  shuffled: {pairs[0][1]}")

    chars = sorted(set(text))
    stoi = {c: i for i, c in enumerate(chars)}
    task_text = torch.tensor([stoi[c] for c in "\n".join(train_lines)])
    book_text = torch.tensor([stoi[c] for c in corpus])
    print(f"A reads {len(task_text):,} characters; B reads {len(book_text):,}")

    torch.manual_seed(0)
    a = CharModel(len(chars)).to(args.device)
    train(a, task_text, steps, args.device, "A")
    before = picks_real(a, pairs, stoi, args.device)
    print(f"A, the 300 task lines only:  picks the real line {before:.1f}%", flush=True)

    torch.manual_seed(0)
    b = CharModel(len(chars)).to(args.device)
    train(b, book_text, steps, args.device, "B")
    after = picks_real(b, pairs, stoi, args.device)
    print(f"B, pre-trained on the text:  picks the real line {after:.1f}%", flush=True)

    print(
        f"RESULT before={before:.1f} after={after:.1f} "
        f"seconds={time.time() - t0:.0f} device={args.device}"
    )


if __name__ == "__main__":
    main()
