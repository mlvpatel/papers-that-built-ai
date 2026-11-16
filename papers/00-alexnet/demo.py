"""Step 00: a convolutional network sees images far better than a fully connected one.

Paper: Krizhevsky, Sutskever, Hinton, "ImageNet Classification with Deep
Convolutional Neural Networks" (AlexNet, NeurIPS 2012)

Two networks learn CIFAR-10 (32x32 colour photos, 10 classes) with the same
optimizer, batch order, seed and number of epochs. Model A flattens each photo
into 3,072 numbers and uses fully connected layers. Model B is a small CNN: it
slides small shared filters over the photo, as AlexNet does. When a GPU is
present, the demo also times 100 training batches of the CNN on CPU and GPU.

Run it from the repo root:
    python papers/00-alexnet/demo.py            # full run, laptop GPU
    python papers/00-alexnet/demo.py --quick    # fewer epochs, for a CPU
    python papers/00-alexnet/demo.py --smoke    # random images, used by the tests

The first full or quick run downloads CIFAR-10 (about 170 MB) into data/.
"""

import argparse
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

DATA = Path(__file__).resolve().parents[2] / "data"
BATCH, LR, TIMED_BATCHES = 128, 1e-3, 100
EPOCHS = {"full": 10, "quick": 4, "smoke": 1}
CLASSES = 10

GPUS = [("mps", torch.backends.mps.is_available()), ("cuda", torch.cuda.is_available())]


def fully_connected():
    return nn.Sequential(
        nn.Flatten(),
        nn.Linear(3 * 32 * 32, 512),
        nn.ReLU(),
        nn.Linear(512, 512),
        nn.ReLU(),
        nn.Linear(512, CLASSES),
    )


def small_cnn():
    def stage(c_in, c_out):  # AlexNet's pattern: convolution, ReLU, max pooling
        return [nn.Conv2d(c_in, c_out, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2)]

    return nn.Sequential(
        *stage(3, 32),  # 32x32 -> 16x16
        *stage(32, 64),  # 16x16 -> 8x8
        *stage(64, 128),  # 8x8 -> 4x4
        nn.Flatten(),
        nn.Linear(128 * 4 * 4, 256),
        nn.ReLU(),
        nn.Linear(256, CLASSES),
    )


def load(smoke):
    """CIFAR-10 as uint8 tensors (N, 3, 32, 32) and int64 labels."""
    if smoke:  # random images, so the tests never download anything
        g = torch.Generator().manual_seed(0)
        x = torch.randint(0, 256, (384, 3, 32, 32), dtype=torch.uint8, generator=g)
        y = torch.randint(0, CLASSES, (384,), generator=g)
        return x[:256], y[:256], x[256:], y[256:]
    from torchvision.datasets import CIFAR10

    sets = [CIFAR10(DATA, train=t, download=True) for t in (True, False)]
    out = []
    for s in sets:
        out += [torch.from_numpy(s.data).permute(0, 3, 1, 2), torch.tensor(s.targets)]
    return out


def train(model, x, y, norm, epochs, device, name):
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    g = torch.Generator().manual_seed(1)  # same batch order for both models
    for epoch in range(epochs):
        t0, total = time.time(), 0.0
        for idx in torch.randperm(len(x), generator=g).split(BATCH):
            xb, yb = norm(x[idx].to(device)), y[idx].to(device)
            loss = F.cross_entropy(model(xb), yb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            total += loss.item() * len(idx)
        print(
            f"  {name}  epoch {epoch + 1:2d}  train loss {total / len(x):.3f}  "
            f"{time.time() - t0:.1f} s",
            flush=True,
        )


@torch.no_grad()
def accuracy(model, x, y, norm, device):
    model.eval()
    hits = sum(
        (model(norm(xb.to(device))).argmax(1).cpu() == yb).sum().item()
        for xb, yb in zip(x.split(1000), y.split(1000))
    )
    model.train()
    return 100 * hits / len(x)


def seconds_for_batches(x, y, norm, device):
    """Time TIMED_BATCHES training steps of a fresh CNN on one device."""
    torch.manual_seed(0)
    model = small_cnn().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    t0 = time.time()
    for i in range(TIMED_BATCHES):
        idx = torch.arange(i * BATCH, (i + 1) * BATCH) % len(x)
        loss = F.cross_entropy(model(norm(x[idx].to(device))), y[idx].to(device))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    loss.item()  # waits for the GPU to finish before the clock stops
    return time.time() - t0


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--quick", action="store_true", help="4 epochs, for a CPU")
    p.add_argument("--smoke", action="store_true", help="random images, for tests")
    devices = [d for d, ok in GPUS if ok] + ["cpu"]  # the ones this machine has
    p.add_argument(
        "--device",
        default=devices[0],
        choices=devices,
        help="where to run (default: a GPU if found, else cpu)",
    )
    args = p.parse_args(argv)
    epochs = EPOCHS["smoke" if args.smoke else "quick" if args.quick else "full"]
    t0 = time.time()

    x_train, y_train, x_test, y_test = load(args.smoke)
    pixels = x_train.float() / 255
    mean = pixels.mean((0, 2, 3)).view(1, 3, 1, 1)
    std = pixels.std((0, 2, 3)).view(1, 3, 1, 1)

    def norm(xb):
        return (xb.float() / 255 - mean.to(xb.device)) / std.to(xb.device)

    print(f"{len(x_train):,} training photos, {len(x_test):,} test photos")
    print(f"{CLASSES} classes, so guessing scores 10%")

    results = []
    for name, build in [
        ("A fully connected", fully_connected),
        ("B small CNN", small_cnn),
    ]:
        torch.manual_seed(0)
        model = build().to(args.device)
        params = sum(p.numel() for p in model.parameters())
        print(f"{name}: {params:,} parameters", flush=True)
        train(model, x_train, y_train, norm, epochs, args.device, name[0])
        results.append(accuracy(model, x_test, y_test, norm, args.device))
        print(f"{name}: test accuracy {results[-1]:.1f}%", flush=True)

    extra = ""
    if args.device != "cpu":
        cpu = seconds_for_batches(x_train, y_train, norm, "cpu")
        gpu = seconds_for_batches(x_train, y_train, norm, args.device)
        print(f"{TIMED_BATCHES} CNN training batches: CPU {cpu:.1f} s, GPU {gpu:.1f} s")
        extra = f" cpu_s={cpu:.1f} gpu_s={gpu:.1f}"

    print(
        f"RESULT before={results[0]:.1f} after={results[1]:.1f} "
        f"seconds={time.time() - t0:.0f} device={args.device}{extra}"
    )


if __name__ == "__main__":
    main()
