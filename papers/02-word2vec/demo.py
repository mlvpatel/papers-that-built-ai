"""Step 02: Word2Vec turns words into coordinates where related words sit close.

Paper: Mikolov, Chen, Corrado, Dean, "Efficient Estimation of Word
Representations in Vector Space" (2013), https://arxiv.org/abs/1301.3781

The demo trains skip-gram vectors on the words of Tiny Shakespeare: each word
learns to predict the words around it. It uses negative sampling and frequent-word
subsampling, both from the authors' follow-up paper later in 2013. Before and after training it scores a
fixed list of related pairs (king/queen, day/night, ...) against random pairs,
and prints the nearest words to "king", "love" and "night".

Run it from the repo root:
    python papers/02-word2vec/demo.py            # full run, laptop GPU
    python papers/02-word2vec/demo.py --quick    # fewer steps, for a CPU
    python papers/02-word2vec/demo.py --smoke    # a few steps, used by the tests
"""

import argparse
import random
import re
import time
from collections import Counter
from pathlib import Path

import torch
import torch.nn.functional as F

DATA = Path(__file__).resolve().parents[2] / "data" / "tinyshakespeare.txt"
DIM, WINDOW, NEGATIVES, BATCH, MIN_COUNT, SUBSAMPLE = 64, 2, 5, 4096, 5, 1e-3
STEPS = {"full": 5000, "quick": 1000, "smoke": 2}
# Written down before the first training run; pairs missing from the vocabulary are skipped.
RELATED = [
    ("king", "queen"),
    ("lord", "lady"),
    ("father", "mother"),
    ("father", "son"),
    ("mother", "daughter"),
    ("brother", "sister"),
    ("husband", "wife"),
    ("man", "woman"),
    ("son", "daughter"),
    ("day", "night"),
    ("sun", "moon"),
    ("heaven", "earth"),
    ("life", "death"),
    ("love", "hate"),
    ("war", "peace"),
    ("master", "servant"),
]
PROBES = ["king", "love", "night"]

GPUS = [("mps", torch.backends.mps.is_available()), ("cuda", torch.cuda.is_available())]


def words_of(text):
    """Lowercase words of the dialogue, without the speaker headings."""
    body = []
    for block in text.split("\n\n"):
        head, _, rest = block.partition("\n")
        body.append(rest if head.endswith(":") else block)
    return re.findall(r"[a-z']+", "\n".join(body).lower())


def score(vectors, pairs, random_pairs):
    """Mean cosine of related pairs minus mean cosine of random pairs."""
    v = F.normalize(vectors, dim=1)
    related = (v[pairs[:, 0]] * v[pairs[:, 1]]).sum(1).mean()
    baseline = (v[random_pairs[:, 0]] * v[random_pairs[:, 1]]).sum(1).mean()
    return (related - baseline).item()


def neighbours(vectors, itos, stoi, word, k=5):
    v = F.normalize(vectors, dim=1)
    sims = v @ v[stoi[word]]
    sims[stoi[word]] = -1
    return [itos[i] for i in sims.topk(k).indices.tolist()]


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--quick", action="store_true", help="1000 steps, for a CPU")
    p.add_argument("--smoke", action="store_true", help="2 steps, for the tests")
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
    torch.manual_seed(0)
    rng = random.Random(0)

    words = words_of(DATA.read_text(encoding="utf-8"))
    counts = Counter(words)
    itos = sorted(w for w, c in counts.items() if c >= MIN_COUNT)
    stoi = {w: i for i, w in enumerate(itos)}
    freq = torch.tensor([counts[w] for w in itos], dtype=torch.float)
    total = freq.sum().item()
    # Word2Vec's subsampling: drop very frequent words ("the", "and") more often
    kept = [
        stoi[w]
        for w in words
        if w in stoi and rng.random() < (SUBSAMPLE * total / counts[w]) ** 0.5
    ]
    ids = torch.tensor(kept)
    centres, contexts = [], []
    for offset in range(1, WINDOW + 1):  # skip-gram: each word predicts its neighbours
        centres += [ids[:-offset], ids[offset:]]
        contexts += [ids[offset:], ids[:-offset]]
    centres, contexts = torch.cat(centres), torch.cat(contexts)
    noise = freq**0.75  # negative samples drawn from unigram counts to the 3/4 power

    pairs = torch.tensor(
        [(stoi[a], stoi[b]) for a, b in RELATED if a in stoi and b in stoi]
    )
    random_pairs = torch.randint(len(itos), (1000, 2))
    print(
        f"{len(words):,} words, {len(itos):,} in the vocabulary, "
        f"{len(centres):,} training pairs, {len(pairs)} related pairs scored"
    )

    vin = ((torch.rand(len(itos), DIM) - 0.5) / DIM).to(args.device).requires_grad_()
    vout = torch.zeros(len(itos), DIM, device=args.device, requires_grad=True)
    before = score(vin.detach().cpu(), pairs, random_pairs)
    before_nn = {w: neighbours(vin.detach().cpu(), itos, stoi, w) for w in PROBES}

    opt = torch.optim.Adam([vin, vout], lr=3e-3)
    for step in range(steps):
        idx = torch.randint(len(centres), (BATCH,))
        c, o = centres[idx].to(args.device), contexts[idx].to(args.device)
        neg = torch.multinomial(noise, BATCH * NEGATIVES, replacement=True)
        neg = neg.view(BATCH, NEGATIVES).to(args.device)
        u = vin[c]
        pos_loss = F.logsigmoid((u * vout[o]).sum(1))
        neg_loss = F.logsigmoid(-(vout[neg] @ u.unsqueeze(2)).squeeze(2)).sum(1)
        loss = -(pos_loss + neg_loss).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if step % 1000 == 0 or step == steps - 1:
            print(f"  step {step:5d}  loss {loss.item():.3f}", flush=True)

    trained = vin.detach().cpu()
    after = score(trained, pairs, random_pairs)
    for w in PROBES:
        print(f"{w:>6}  untrained: {', '.join(before_nn[w])}")
        print(f"{'':>6}  trained:   {', '.join(neighbours(trained, itos, stoi, w))}")
    print(
        f"RESULT before={before:.3f} after={after:.3f} "
        f"seconds={time.time() - t0:.0f} device={args.device}"
    )


if __name__ == "__main__":
    main()
