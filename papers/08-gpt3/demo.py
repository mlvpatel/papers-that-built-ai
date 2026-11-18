"""Step 08: a few examples in the prompt teach a base model the task.

Paper: Brown et al., "Language Models are Few-Shot Learners" (GPT-3, OpenAI,
2020), https://arxiv.org/abs/2005.14165

Pythia-1.4B, a base model trained only to predict the next token on 2020-era
web text, does two small tasks: label a review positive or negative, and write
a date as YYYY-MM-DD. Each test item is asked twice, first with only the task
description (zero-shot), then with a few solved examples before it (few-shot).
Same model, same weights, same items; only the prompt changes. Greedy decoding.

Run it from the repo root:
    python papers/08-gpt3/demo.py

The first run downloads the model (about 2.9 GB) into the Hugging Face cache.
"""

import argparse
import time

import torch

MODEL = "EleutherAI/pythia-1.4b"
REVISION = "fedc38a16eea3bd36a96b906d78d11d2ce18ed79"  # pinned, so results repeat

REVIEWS = [
    ("The battery lasts two days and the screen is gorgeous.", "positive"),
    ("It broke after a week and support never answered.", "negative"),
    ("Best pizza I have had in years, and the staff were lovely.", "positive"),
    ("The hotel room smelled of smoke and the sheets were stained.", "negative"),
    ("Setup took five minutes and it just works.", "positive"),
    ("Two hours of my life I will never get back.", "negative"),
    ("Comfortable shoes, I walked ten miles without a blister.", "positive"),
    ("The app crashes every time I open the camera.", "negative"),
]
REVIEW_EXAMPLES = [
    ("I love how light this laptop is.", "positive"),
    ("The soup was cold and bland.", "negative"),
    ("Fast shipping and great quality.", "positive"),
    ("Customer service hung up on me twice.", "negative"),
]
DATES = [
    ("March 5, 2021", "2021-03-05"),
    ("December 31, 1999", "1999-12-31"),
    ("July 4, 1776", "1776-07-04"),
    ("February 29, 2024", "2024-02-29"),
    ("October 12, 1492", "1492-10-12"),
    ("June 1, 2030", "2030-06-01"),
    ("September 9, 2009", "2009-09-09"),
    ("April 15, 1912", "1912-04-15"),
]
DATE_EXAMPLES = [
    ("January 2, 2003", "2003-01-02"),
    ("November 20, 1985", "1985-11-20"),
    ("August 7, 2015", "2015-08-07"),
]
# (task description, test items, examples for the few-shot prompt, tokens to write)
TASKS = [
    ("Classify the review as positive or negative.", REVIEWS, REVIEW_EXAMPLES, 6),
    ("Convert the date to YYYY-MM-DD format.", DATES, DATE_EXAMPLES, 12),
]

GPUS = [("mps", torch.backends.mps.is_available()), ("cuda", torch.cuda.is_available())]


def prompt(task, x, examples):
    """The task line, then any solved examples, then the item to complete."""
    shown = "".join(f"Input: {ex}\nOutput: {ey}\n" for ex, ey in examples)
    return f"{task}\n{shown}Input: {x}\nOutput:"


def answer(reply):
    """A base model keeps writing after its answer, so keep the first line."""
    return reply.strip().split("\n")[0].lower()


def passes(reply, want):
    return answer(reply).startswith(want)


def load(device):  # pragma: no cover - downloads a 2.9 GB model, too big for CI
    from transformers import AutoModelForCausalLM, AutoTokenizer

    try:  # the first run downloads the model
        tok = AutoTokenizer.from_pretrained(MODEL, revision=REVISION)
        model = AutoModelForCausalLM.from_pretrained(
            MODEL, revision=REVISION, dtype=torch.bfloat16
        )
    except OSError as err:  # no network, the Hub is down, or the disk is full
        raise SystemExit(f"Could not load {MODEL}: {err}") from err
    return tok, model.to(device).eval()


@torch.no_grad()
def complete(tok, model, text, max_new):  # pragma: no cover - needs the model
    ids = tok(text, return_tensors="pt").to(model.device)
    out = model.generate(
        **ids, max_new_tokens=max_new, do_sample=False, pad_token_id=tok.eos_token_id
    )
    return tok.decode(out[0, ids.input_ids.shape[1] :], skip_special_tokens=True)


def main(argv=None):  # pragma: no cover - needs the model
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
    print(f"{MODEL} on {args.device}")

    before = after = 0
    for task, items, examples, max_new in TASKS:
        print(f"\n{task}  ({len(examples)} examples in the few-shot prompt)")
        print(f"  {'want':12} {'zero-shot':24} few-shot")
        zero_right = few_right = 0
        for x, want in items:
            zero = complete(tok, model, prompt(task, x, []), max_new)
            few = complete(tok, model, prompt(task, x, examples), max_new)
            zero_right += passes(zero, want)
            few_right += passes(few, want)
            print(f"  {want:12} {answer(zero)[:22]!r:24} {answer(few)[:22]!r}")
        print(
            f"  zero-shot {zero_right}/{len(items)}, few-shot {few_right}/{len(items)}"
        )
        before += zero_right
        after += few_right

    total = sum(len(items) for _, items, _, _ in TASKS)
    print(f"\nzero-shot {before}/{total}, few-shot {after}/{total}")
    print(
        f"RESULT before={before} after={after} seconds={time.time() - t0:.0f} "
        f"device={args.device} items={total}"
    )


if __name__ == "__main__":
    main()
