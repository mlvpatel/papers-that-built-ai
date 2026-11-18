"""Step 10: writing the steps first turns wrong answers into right ones.

Paper: Wei et al., "Chain-of-Thought Prompting Elicits Reasoning in Large
Language Models" (Google, 2022), https://arxiv.org/abs/2201.11903

One model, Qwen2.5-1.5B-Instruct, answers the same 12 word problems twice:
first told to give only the final number, then told to reason step by step.
The answer keys are computed in code, below each question. Greedy decoding,
so every run gives the same text on the same machine.

Run it from the repo root:
    python papers/10-chain-of-thought/demo.py

The first run downloads the model (about 3.1 GB) into the Hugging Face cache.
"""

import argparse
import itertools
import re
import sys
import time

import torch

MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
REVISION = "989aa7980e4cf806f80c7fef2b1adb7bc71aa306"  # pinned, so results repeat
DIRECT = "\nAnswer with only the final number, no explanation."
STEPS = "\nPlease reason step by step, and put your final answer within \\boxed{}."
DIRECT_TOKENS, STEP_TOKENS = 16, 768

GPUS = [("mps", torch.backends.mps.is_available()), ("cuda", torch.cuda.is_available())]


def _pizza():
    slices = 24
    for share in (3, 4, 2):  # Alice a third, Bob a quarter of the rest, Cara half
        slices -= slices // share
    return slices


PROBLEMS = [
    (
        (
            "A bakery sells muffins for $3 each and cookies for $2 each. Maria buys 4 "
            "muffins and 7 cookies and pays with a $50 bill. How much change does she "
            "get, in dollars?"
        ),
        50 - (4 * 3 + 7 * 2),
    ),
    (
        (
            "A train leaves at 9:40 a.m. and the trip takes 2 hours and 35 minutes. How "
            "many minutes after 12:00 noon does it arrive?"
        ),
        (9 * 60 + 40 + 2 * 60 + 35) - 12 * 60,
    ),
    (
        (
            "A rectangle has a perimeter of 54 cm, and its length is twice its width. "
            "What is its area in square centimeters?"
        ),
        next(w * 2 * w for w in range(1, 99) if 2 * (w + 2 * w) == 54),
    ),
    (
        (
            "Tom has three times as many marbles as Ana. After Tom gives Ana 12 marbles, "
            "they have the same number. How many marbles did Tom have at the start?"
        ),
        next(3 * a for a in range(1, 99) if 3 * a - 12 == a + 12),
    ),
    (
        (
            "A shirt costs $80. It is discounted by 25%, and then a further 10% is taken "
            "off the discounted price. What is the final price in dollars?"
        ),
        80 * 0.75 * 0.9,
    ),
    (
        (
            "A car drives 150 km at 60 km/h and then another 150 km at 100 km/h. What is "
            "its average speed for the whole trip, in km/h?"
        ),
        300 / (150 / 60 + 150 / 100),
    ),
    (
        (
            "A pizza is cut into 24 slices. Alice eats a third of the slices. Bob then "
            "eats a quarter of the slices that are left. Cara then eats half of what "
            "remains. How many slices are left?"
        ),
        _pizza(),
    ),
    (
        (
            "What is the sum of all integers from 1 to 200 that are divisible by 3 or "
            "by 5?"
        ),
        sum(i for i in range(1, 201) if i % 3 == 0 or i % 5 == 0),
    ),
    (
        "How many positive divisors does 360 have?",
        sum(1 for d in range(1, 361) if 360 % d == 0),
    ),
    (
        "What is the remainder when 7 to the power 100 is divided by 13?",
        pow(7, 100, 13),
    ),
    (
        (
            "In how many ways can 5 people sit in a row if two of them, Ann and Ben, "
            "must not sit next to each other?"
        ),
        sum(
            1
            for seats in itertools.permutations(range(5))
            if abs(seats.index(0) - seats.index(1)) != 1
        ),
    ),
    (
        (
            "The product of two consecutive positive even integers is 528. What is "
            "their sum?"
        ),
        next(2 * n + 2 for n in range(2, 99, 2) if n * (n + 2) == 528),
    ),
]


def number(text, first=False):
    """The answer in a model's reply: the last \\boxed{} if any, else a number.

    A box with braces inside, like \\boxed{\\frac{1}{2}}, is not read; the last number
    in the reply is used instead. Every answer key here is a whole number.
    """
    boxed = re.findall(r"\\boxed\{([^{}]*)\}", text)
    found = re.findall(r"-?\d[\d,]*(?:\.\d+)?", boxed[-1] if boxed else text)
    if not found:
        return None
    pick = found[0] if first and not boxed else found[-1]
    return float(pick.replace(",", ""))


def is_right(got, want):
    return got is not None and abs(got - want) < 1e-6


def load(device):  # pragma: no cover - downloads a 3 GB model, too big for CI
    from transformers import AutoModelForCausalLM, AutoTokenizer

    try:  # the first run downloads the model
        tok = AutoTokenizer.from_pretrained(MODEL, revision=REVISION)
        model = AutoModelForCausalLM.from_pretrained(
            MODEL,
            revision=REVISION,
            dtype=torch.bfloat16,
            use_safetensors=True,  # never unpickle downloaded weights
        )
    except OSError as err:  # no network, the Hub is down, or the disk is full
        raise SystemExit(f"Could not load {MODEL}: {err}") from err
    return tok, model.to(device).eval()


@torch.no_grad()
def ask(tok, model, question, max_new):
    chat = [{"role": "user", "content": question}]
    prompt = tok.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
    ids = tok(prompt, return_tensors="pt").to(model.device)
    out = model.generate(
        **ids, max_new_tokens=max_new, do_sample=False, pad_token_id=tok.eos_token_id
    )
    new = out[0, ids.input_ids.shape[1] :]
    return tok.decode(new, skip_special_tokens=True), len(new)


def run(tok, model, suffix, max_new, first, label):
    right, tokens, t0 = 0, 0, time.time()
    for i, (question, want) in enumerate(PROBLEMS, 1):
        print(
            f"  {label}  problem {i} of {len(PROBLEMS)}...", file=sys.stderr, flush=True
        )
        reply, n = ask(tok, model, question + suffix, max_new)
        got = number(reply, first)
        right += is_right(got, want)
        tokens += n
        mark = "right" if is_right(got, want) else "wrong"
        print(f"  {label}  {i:2d}  want {want:g}  got {got}  {mark}  ({n} tokens)")
    print(
        f"{label}: {right}/{len(PROBLEMS)} right, {tokens:,} tokens, "
        f"{time.time() - t0:.0f} s",
        flush=True,
    )
    return right, tokens, reply


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
    tok, model = load(args.device)
    print(f"{MODEL} on {args.device}; {len(PROBLEMS)} word problems")

    before, direct_tokens, _ = run(tok, model, DIRECT, DIRECT_TOKENS, True, "direct")
    after, step_tokens, last = run(tok, model, STEPS, STEP_TOKENS, False, "steps ")
    print(f"--- step-by-step reply to problem {len(PROBLEMS)} ---\n{last.strip()}\n---")
    print(
        f"RESULT before={before} after={after} seconds={time.time() - t0:.0f} "
        f"device={args.device} direct_tokens={direct_tokens} step_tokens={step_tokens}"
    )


if __name__ == "__main__":
    main()
