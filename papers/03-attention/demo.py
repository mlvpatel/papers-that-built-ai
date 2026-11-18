"""Step 03: attention lets a translation model look back at every input word.

Paper: Bahdanau, Cho, Bengio, "Neural Machine Translation by Jointly Learning
to Align and Translate" (2014), https://arxiv.org/abs/1409.0473

The demo trains two recurrent encoder-decoders to reverse random strings. They
are the same network from the same seed, except for what the decoder sees at
each step: without attention, the encoder's last state (the whole input
squeezed into one vector); with attention, a weighted mix of every input
position, with the weights learned. It reports character accuracy at lengths
8, 16 and 32.

Run it from the repo root:
    python papers/03-attention/demo.py            # full run, laptop GPU
    python papers/03-attention/demo.py --quick    # fewer steps, for a CPU
    python papers/03-attention/demo.py --smoke    # a few steps, used by tests
"""

import argparse
import random
import time

import torch
import torch.nn.functional as F
from torch import nn

SYMBOLS, EMB, HIDDEN, BATCH, LR = 10, 32, 128, 128, 1e-3
TRAIN_LENGTHS, TEST_LENGTHS, TEST_STRINGS = (4, 32), [8, 16, 32], 500
STEPS = {"full": 2000, "quick": 400, "smoke": 2}

GPUS = [("mps", torch.backends.mps.is_available()), ("cuda", torch.cuda.is_available())]


class Seq2Seq(nn.Module):
    def __init__(self, attention):
        super().__init__()
        self.attention = attention
        self.emb_in = nn.Embedding(SYMBOLS, EMB)
        self.encoder = nn.GRU(EMB, HIDDEN, batch_first=True)
        self.emb_out = nn.Embedding(
            SYMBOLS + 1, EMB
        )  # the extra symbol starts decoding
        self.cell = nn.GRUCell(EMB + HIDDEN, HIDDEN)
        # built in both models so they start from the same weights; unused without
        # attention
        self.w_dec = nn.Linear(HIDDEN, HIDDEN, bias=False)
        self.w_enc = nn.Linear(HIDDEN, HIDDEN, bias=False)
        self.v = nn.Linear(HIDDEN, 1, bias=False)
        self.out = nn.Linear(2 * HIDDEN, SYMBOLS)

    def forward(self, src, target=None):
        enc, h = self.encoder(self.emb_in(src))
        h, keys = h[0], self.w_enc(enc)
        prev = torch.full((src.size(0),), SYMBOLS, device=src.device)
        logits = []
        for t in range(src.size(1)):
            if self.attention:  # Bahdanau: score every input position, then mix them
                scores = self.v(torch.tanh(keys + self.w_dec(h).unsqueeze(1))).squeeze(
                    2
                )
                context = (scores.softmax(1).unsqueeze(2) * enc).sum(1)
            else:  # the whole input squeezed into the encoder's last state
                context = enc[:, -1]
            h = self.cell(torch.cat([self.emb_out(prev), context], 1), h)
            step = self.out(torch.cat([h, context], 1))
            logits.append(step)
            prev = target[:, t] if target is not None else step.argmax(1)
        return torch.stack(logits, 1)


def train(model, steps, device, rng):
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    for step in range(steps):
        src = torch.randint(
            SYMBOLS, (BATCH, rng.randint(*TRAIN_LENGTHS)), device=device
        )
        target = src.flip(1)
        loss = F.cross_entropy(model(src, target).flatten(0, 1), target.flatten())
        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # as in the paper, page 15
        opt.step()
        if step % 500 == 0 or step == steps - 1:
            print(f"    step {step:4d}  loss {loss.item():.3f}", flush=True)


@torch.no_grad()
def accuracy(model, length, device):
    gen = torch.Generator().manual_seed(length)  # same test strings for both models
    src = torch.randint(SYMBOLS, (TEST_STRINGS, length), generator=gen).to(device)
    model.eval()
    pred = model(src).argmax(2)
    model.train()
    return 100 * (pred == src.flip(1)).float().mean().item()


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    speed = p.add_mutually_exclusive_group()
    speed.add_argument(
        "--quick", action="store_true", help="400 steps per model, for a CPU"
    )
    speed.add_argument(
        "--smoke", action="store_true", help="2 steps per model, for the tests"
    )
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

    results = {}
    for name, attention in [("without attention", False), ("with attention", True)]:
        print(f"  training {name}", flush=True)
        torch.manual_seed(0)
        model = Seq2Seq(attention).to(args.device)
        train(model, steps, args.device, random.Random(0))
        results[name] = [accuracy(model, n, args.device) for n in TEST_LENGTHS]

    print("length | without attention | with attention   (characters correct)")
    for i, n in enumerate(TEST_LENGTHS):
        no, yes = results["without attention"][i], results["with attention"][i]
        print(f"{n:6d} | {no:16.1f}% | {yes:13.1f}%")
    print(
        f"RESULT before={results['without attention'][-1]:.1f} "
        f"after={results['with attention'][-1]:.1f} "
        f"seconds={time.time() - t0:.0f} device={args.device}"
    )


if __name__ == "__main__":
    main()
