"""Step 01: backpropagation trains a hidden layer, and a hidden layer solves XOR.

Paper: Rumelhart, Hinton, Williams, "Learning representations by
back-propagating errors" (Nature, 1986), https://doi.org/10.1038/323533a0

XOR outputs 1 when exactly one of its two inputs is 1. Model A has no hidden
layer: it can only draw one straight line through the four points, so it gets
at most three of them right. Model B adds a hidden layer of four units, trained
by backpropagation, and can bend the boundary. Same seed, data and steps.

Run it from the repo root (it takes seconds on any machine):
    python papers/01-backprop/demo.py
    python papers/01-backprop/demo.py --quick    # same as the full run
    python papers/01-backprop/demo.py --smoke    # a few steps, used by the tests
"""

import argparse
import time

import torch
import torch.nn.functional as F
from torch import nn

HIDDEN, LR = 4, 0.05
STEPS = {"full": 2000, "quick": 2000, "smoke": 2}
X = torch.tensor([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
Y = torch.tensor([[0.0], [1.0], [1.0], [0.0]])  # XOR

GPUS = [("mps", torch.backends.mps.is_available()), ("cuda", torch.cuda.is_available())]


def train(model, steps, device):
    x, y = X.to(device), Y.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    for _ in range(steps):
        loss = F.binary_cross_entropy_with_logits(model(x), y)
        opt.zero_grad(set_to_none=True)
        loss.backward()  # backpropagation: the error flows back through every layer
        opt.step()
    with torch.no_grad():
        return torch.sigmoid(model(x)).cpu()


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--quick", action="store_true", help="same as the full run")
    p.add_argument("--smoke", action="store_true", help="2 steps, for the tests")
    devices = [d for d, ok in GPUS if ok] + ["cpu"]  # the ones this machine has
    p.add_argument(
        "--device",
        default="cpu",
        choices=devices,
        help="where to run (default: cpu, since XOR needs no GPU)",
    )
    args = p.parse_args(argv)
    steps = STEPS["smoke" if args.smoke else "quick" if args.quick else "full"]
    t0 = time.time()

    torch.manual_seed(0)
    no_hidden = nn.Linear(2, 1).to(args.device)
    a = train(no_hidden, steps, args.device)
    torch.manual_seed(0)
    hidden = nn.Sequential(nn.Linear(2, HIDDEN), nn.Tanh(), nn.Linear(HIDDEN, 1))
    b = train(hidden.to(args.device), steps, args.device)

    print("input  target | A: no hidden layer | B: hidden layer")
    for (x1, x2), y, pa, pb in zip(
        X.tolist(), Y.flatten().tolist(), a.flatten(), b.flatten()
    ):
        print(
            f"{x1:.0f} {x2:.0f}      {y:.0f}    |        {pa:.2f}        |      {pb:.2f}"
        )
    before = 100 * ((a > 0.5).float() == Y).float().mean().item()
    after = 100 * ((b > 0.5).float() == Y).float().mean().item()
    print(
        f"RESULT before={before:.0f} after={after:.0f} "
        f"seconds={time.time() - t0:.0f} device={args.device}"
    )


if __name__ == "__main__":
    main()
