"""Step 11: a model trained to reason checks its own work before answering.

Paper: DeepSeek-AI, "DeepSeek-R1: Incentivizing Reasoning Capability in LLMs
via Reinforcement Learning" (2025), https://arxiv.org/abs/2501.12948

Two models of the same size get the same four competition-style problems.
Qwen2.5-1.5B-Instruct is told to reason step by step (step 10).
DeepSeek-R1-Distill-Qwen-1.5B was trained on DeepSeek-R1's reasoning and gets
the same prompt, sampled at temperature 0.6 (the value DeepSeek recommends) and
top-p 0.95 (as in DeepSeek's evaluations). The problems and their answer keys
(computed in code) were committed before the first run.

Run it from the repo root:
    python papers/11-deepseek-r1/demo.py

The first run downloads both models (about 6.7 GB) into the Hugging Face cache.
"""

import argparse
import re
import sys
import time
from math import comb, gcd

import torch

INSTRUCT = ("Qwen/Qwen2.5-1.5B-Instruct", "989aa7980e4cf806f80c7fef2b1adb7bc71aa306")
R1 = (
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
    "ad9f0ae0864d7fbcd1cd905e3c6c5b069cc8b562",
)
STEPS = "\nPlease reason step by step, and put your final answer within \\boxed{}."
INSTRUCT_TOKENS, R1_TOKENS = 1024, 4096

GPUS = [("mps", torch.backends.mps.is_available()), ("cuda", torch.cuda.is_available())]


def _same_colour():
    together = comb(5, 2) + comb(7, 2)  # two red or two blue
    total = comb(12, 2)
    g = gcd(together, total)
    return together // g + total // g


def _stairs(n):
    ways = [1, 1, 2]  # 0, 1 and 2 stairs
    while len(ways) <= n:
        ways.append(ways[-1] + ways[-2] + ways[-3])
    return ways[n]


PROBLEMS = [
    (
        "How many integers from 1 to 1000 are perfect squares or perfect cubes?",
        len({k * k for k in range(1, 32)} | {k**3 for k in range(1, 11)}),
    ),
    (
        (
            "A bag holds 5 red and 7 blue marbles. Two marbles are drawn at random "
            "without replacement. The probability that both have the same colour is "
            "m/n in lowest terms. What is m + n?"
        ),
        _same_colour(),
    ),
    (
        (
            "In how many different ways can you climb a staircase of 12 steps if each "
            "move climbs 1, 2 or 3 steps?"
        ),
        _stairs(12),
    ),
    ("What are the last two digits of 7 to the power 2026?", pow(7, 2026, 100)),
]


def number(text):
    """The last \\boxed{} answer in a reply, else the last number in it.

    A box with braces inside, like \\boxed{\\frac{1}{2}}, is not read; the last number
    in the reply is used instead. Every answer key here is a whole number.
    """
    boxed = re.findall(r"\\boxed\{([^{}]*)\}", text)
    found = re.findall(r"-?\d[\d,]*(?:\.\d+)?", boxed[-1] if boxed else text)
    return float(found[-1].replace(",", "")) if found else None


def excerpt(text, word="Wait", width=300):
    """The text around the first time the model stops to check itself."""
    at = text.find(word)
    if at < 0:
        return ""
    return text[max(0, at - width) : at + width].strip()


def load(name, revision, device):  # pragma: no cover - 3 GB download, not in CI
    from transformers import AutoModelForCausalLM, AutoTokenizer

    try:  # the first run downloads the model
        tok = AutoTokenizer.from_pretrained(name, revision=revision)
        model = AutoModelForCausalLM.from_pretrained(
            name,
            revision=revision,
            dtype=torch.bfloat16,
            use_safetensors=True,  # never unpickle downloaded weights
        )
    except OSError as err:  # no network, the Hub is down, or the disk is full
        raise SystemExit(f"Could not load {name}: {err}") from err
    return tok, model.to(device).eval()


@torch.no_grad()
def solve(tok, model, question, max_new, sample):
    chat = [{"role": "user", "content": question + STEPS}]
    text = tok.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
    ids = tok(text, return_tensors="pt").to(model.device)
    if sample:  # temperature as DeepSeek recommends; top-p as in its evaluations
        settings = {"do_sample": True, "temperature": 0.6, "top_p": 0.95}
    else:
        settings = {"do_sample": False}
    out = model.generate(
        **ids, max_new_tokens=max_new, pad_token_id=tok.eos_token_id, **settings
    )
    new = out[0, ids.input_ids.shape[1] :]
    return tok.decode(new, skip_special_tokens=True), len(new)


def run(label, name, revision, device, max_new, sample):
    tok, model = load(name, revision, device)
    torch.manual_seed(0)
    right, tokens, replies, t0 = 0, 0, [], time.time()
    for i, (question, want) in enumerate(PROBLEMS, 1):
        print(
            f"  {label}  problem {i} of {len(PROBLEMS)}...", file=sys.stderr, flush=True
        )
        reply, n = solve(tok, model, question, max_new, sample)
        got = number(reply)
        ok = got is not None and abs(got - want) < 1e-6
        right, tokens = right + ok, tokens + n
        replies.append(reply)
        cut = ", cut off" if n >= max_new else ""
        mark = "right" if ok else "wrong"
        print(f"  {label}  {i}  want {want}  got {got}  {mark}  ({n:,} tokens{cut})")
    print(
        f"{label.strip()}: {right}/{len(PROBLEMS)} right, {tokens:,} tokens, "
        f"{time.time() - t0:.0f} s",
        flush=True,
    )
    del model
    if device == "mps":  # give its memory back before the next model loads
        torch.mps.empty_cache()
    elif device == "cuda":
        torch.cuda.empty_cache()
    return right, tokens, replies


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    devices = [d for d, ok in GPUS if ok] + ["cpu"]  # the ones this machine has
    p.add_argument(
        "--device",
        default=devices[0],
        choices=devices,
        help="where to run (default: a GPU if found, else cpu)",
    )
    args = p.parse_args(argv)
    t0 = time.time()
    print(f"{len(PROBLEMS)} problems on {args.device}")
    before, inst_tokens, _ = run(
        "instruct", *INSTRUCT, args.device, INSTRUCT_TOKENS, sample=False
    )
    after, r1_tokens, replies = run("r1      ", *R1, args.device, R1_TOKENS, True)

    for i, reply in enumerate(replies, 1):
        shown = excerpt(reply)
        if shown:
            print(f"--- R1 checking itself, problem {i} ---\n...{shown}...\n---")
            break
    print(
        f"RESULT before={before} after={after} seconds={time.time() - t0:.0f} "
        f"device={args.device} instruct_tokens={inst_tokens} r1_tokens={r1_tokens}"
    )


if __name__ == "__main__":
    main()
