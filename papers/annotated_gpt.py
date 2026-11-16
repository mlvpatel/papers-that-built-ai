"""Act 1 in one file: a small GPT, with every line tagged by the paper it came from.

Tags like [05 ResNet] point to a step in this repo. Ideas from outside the 14
papers point to their row in docs/all-45-papers.md, e.g. [Adam, all-45 #11].
Python plumbing (def, return, print) carries no tag.

The model trains on Tiny Shakespeare by predicting the next character, then
writes 200 characters of its own. RESULT reports the loss on held-out text
(the last 10%) before and after training.

Run it from the repo root:
    python papers/annotated_gpt.py            # full run, laptop GPU
    python papers/annotated_gpt.py --quick    # 1,000 steps, for a CPU
    python papers/annotated_gpt.py --smoke    # 2 steps, used by the tests
"""

import argparse
import math
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

DATA = Path(__file__).resolve().parents[1] / "data" / "tinyshakespeare.txt"
WIDTH, HEADS, LAYERS, CONTEXT, BATCH = 128, 4, 4, 64, 64  # small, to train in minutes
LR = 1e-3  # [Adam, all-45 #11] the default step size
TOKENS_PER_PARAM = 20  # [07 Chinchilla] about 20 training tokens per parameter
STEPS = {"quick": 1000, "smoke": 2}  # [07 Chinchilla] full run: the token budget
SAMPLE_CHARS, EVAL_BATCHES = 200, 10


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.ln1 = nn.LayerNorm(WIDTH)  # [LayerNorm, all-45 #15]
        self.qkv = nn.Linear(WIDTH, 3 * WIDTH)  # [04 Transformer] queries, keys, values
        self.proj = nn.Linear(WIDTH, WIDTH)  # [04 Transformer] joins the heads (W^O)
        self.ln2 = nn.LayerNorm(WIDTH)  # [LayerNorm, all-45 #15]
        self.ff1 = nn.Linear(WIDTH, 4 * WIDTH)  # [04 Transformer] feed-forward, 4x wide
        self.ff2 = nn.Linear(4 * WIDTH, WIDTH)  # [04 Transformer] back to model width
        mask = torch.ones(CONTEXT, CONTEXT, dtype=torch.bool).tril()  # [04 Transformer]
        self.register_buffer("mask", mask)  # [04 Transformer] earlier positions only

    def attention(self, x):
        b, t, _ = x.shape
        qkv = self.qkv(x).view(b, t, 3, HEADS, WIDTH // HEADS)  # [04 Transformer] heads
        q, k, v = qkv.permute(2, 0, 3, 1, 4)  # [04 Transformer] one q, k, v per head
        scores = q @ k.transpose(-2, -1) / math.sqrt(WIDTH // HEADS)  # [04] Eq. 1
        scores = scores.masked_fill(~self.mask[:t, :t], -math.inf)  # [04] no future
        weights = scores.softmax(-1)  # [03 Attention] weights over positions sum to 1
        out = weights @ v  # [03 Attention] weighted sum of the values
        return self.proj(out.transpose(1, 2).reshape(b, t, WIDTH))  # [04] join heads

    def forward(self, x):
        x = x + self.attention(self.ln1(x))  # [05 ResNet] add; [GPT-2, all-45 #22]
        return x + self.ff2(F.gelu(self.ff1(self.ln2(x))))  # [05] [GPT-2 #22] [06 GELU]


class GPT(nn.Module):
    def __init__(self, vocab):
        super().__init__()
        self.tok = nn.Embedding(vocab, WIDTH)  # [02 Word2Vec] [06 GPT-1] W_e
        self.pos = nn.Embedding(CONTEXT, WIDTH)  # [06 GPT-1] learned positions
        self.blocks = nn.Sequential(*[Block() for _ in range(LAYERS)])  # [06 GPT-1]
        self.ln = nn.LayerNorm(WIDTH)  # [GPT-2, all-45 #22] one more after the blocks
        self.head = nn.Linear(WIDTH, vocab, bias=False)  # [04 Transformer] pre-softmax
        self.head.weight = self.tok.weight  # [06 GPT-1] reuses W_e, Eq. 2
        nn.init.normal_(self.tok.weight, std=0.02)  # [06 GPT-1] N(0, 0.02) init

    def forward(self, idx):
        pos = torch.arange(idx.size(1), device=idx.device)  # [06 GPT-1] positions
        h = self.tok(idx) + self.pos(pos)  # [06 GPT-1] h0 = U W_e + W_p, Eq. 2
        return self.head(self.ln(self.blocks(h)))  # [06 GPT-1] next-character scores


def batch(data, gen=None):
    ix = torch.randint(len(data) - CONTEXT - 1, (BATCH,), generator=gen)  # [06 GPT-1]
    x = torch.stack([data[i : i + CONTEXT] for i in ix])  # [06 GPT-1] raw text
    y = torch.stack([data[i + 1 : i + CONTEXT + 1] for i in ix])  # [06] next char
    return x, y


def loss_on(model, x, y, device):
    logits = model(x.to(device))  # [06 GPT-1] Eq. 2
    return F.cross_entropy(logits.flatten(0, 1), y.to(device).flatten())  # [06] Eq. 1


@torch.no_grad()
def held_out_loss(model, data, device):
    model.eval()
    g = torch.Generator().manual_seed(0)  # the same held-out batches every time
    losses = [
        loss_on(model, *batch(data, g), device).item() for _ in range(EVAL_BATCHES)
    ]
    model.train()
    return sum(losses) / len(losses)


def train(model, data, steps, device):
    opt = torch.optim.Adam(model.parameters(), lr=LR)  # [Adam, all-45 #11]
    for step in range(steps):  # [07 Chinchilla] the step count comes from the budget
        loss = loss_on(model, *batch(data), device)  # [06 GPT-1] pre-training, Eq. 1
        opt.zero_grad(set_to_none=True)  # [01 Backprop] clear the last gradients
        loss.backward()  # [01 Backprop] a gradient for every weight
        opt.step()  # [Adam, all-45 #11] move each weight against its gradient
        if step % 1000 == 0 or step == steps - 1:
            print(f"  step {step:4d}  loss {loss.item():.3f}", flush=True)


@torch.no_grad()
def sample(model, start, device):
    model.eval()
    idx = torch.tensor([[start]], device=device)
    for _ in range(SAMPLE_CHARS):  # [04 Transformer] auto-regressive, one at a time
        probs = model(idx[:, -CONTEXT:])[:, -1].softmax(-1)  # [04] next-char odds
        idx = torch.cat([idx, torch.multinomial(probs, 1)], 1)  # [04] feed it back
    return idx[0, 1:].tolist()


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--quick", action="store_true", help="1,000 steps, for a CPU")
    p.add_argument("--smoke", action="store_true", help="2 steps, for tests")
    p.add_argument(
        "--device",
        default="mps"
        if torch.backends.mps.is_available()
        else "cuda"
        if torch.cuda.is_available()
        else "cpu",
    )
    args = p.parse_args(argv)
    t0 = time.time()

    text = DATA.read_text(encoding="utf-8")
    chars = sorted(set(text))  # characters, not GPT-1's BPE tokens
    stoi = {c: i for i, c in enumerate(chars)}
    data = torch.tensor([stoi[c] for c in text])
    split = int(0.9 * len(data))
    train_data, held_out = data[:split], data[split:]

    torch.manual_seed(0)
    model = GPT(len(chars)).to(args.device)
    params = sum(p.numel() for p in model.parameters())
    budget = TOKENS_PER_PARAM * params // (BATCH * CONTEXT)  # [07 Chinchilla]
    steps = STEPS["smoke"] if args.smoke else STEPS["quick"] if args.quick else budget
    print(f"{len(chars)} characters; guessing scores loss {math.log(len(chars)):.2f}")
    print(f"{params:,} parameters; {steps:,} steps of {BATCH * CONTEXT:,} characters")

    before = held_out_loss(model, held_out, args.device)
    print(f"held-out loss before training: {before:.3f}", flush=True)
    train(model, train_data, steps, args.device)
    after = held_out_loss(model, held_out, args.device)
    print(f"held-out loss after training: {after:.3f}")

    torch.manual_seed(1)
    text_out = "".join(chars[i] for i in sample(model, stoi["\n"], args.device))
    print(f"--- {SAMPLE_CHARS} characters written by the model ---\n{text_out}\n---")
    print(
        f"RESULT before={before:.3f} after={after:.3f} "
        f"seconds={time.time() - t0:.0f} device={args.device} steps={steps}"
    )


if __name__ == "__main__":
    main()
