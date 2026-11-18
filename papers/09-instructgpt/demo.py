"""Step 09: a model trained on instructions follows them; a base model does not.

Paper: Ouyang et al., "Training language models to follow instructions with
human feedback" (InstructGPT, OpenAI, 2022), https://arxiv.org/abs/2203.02155

Two different models get the same three instructions. Pythia-1.4B is a base
model trained only to predict the next token. Qwen2.5-1.5B-Instruct was trained
to follow instructions. No instruction-tuned copy of Pythia's own weights can
be downloaded without signing in, so this compares two models from two labs,
not one model before and after: it shows the kind of difference instruction
training makes, not a controlled measurement of it. Each instruction comes with
a simple automatic check, written before the first run. Greedy decoding.

Run it from the repo root:
    python papers/09-instructgpt/demo.py

The first run downloads both models (about 6.0 GB) into the Hugging Face cache.
"""

import argparse
import re
import time

import torch

BASE = ("EleutherAI/pythia-1.4b", "fedc38a16eea3bd36a96b906d78d11d2ce18ed79")
INSTRUCT = ("Qwen/Qwen2.5-1.5B-Instruct", "989aa7980e4cf806f80c7fef2b1adb7bc71aa306")
MAX_NEW = 120  # the same budget for both models

GPUS = [("mps", torch.backends.mps.is_available()), ("cuda", torch.cuda.is_available())]


def lines(text):
    return [line for line in text.strip().splitlines() if line.strip()]


def is_haiku(text):
    return len(lines(text)) == 3


def has_three_tips(text):
    items = [ln for ln in lines(text) if re.match(r"\s*(\d+[.)]|[-*•])\s", ln)]
    return len(items) == 3


def at_most_two_sentences(text):
    sentences = [s for s in re.split(r"[.!?]+(?:\s+|$)", text.strip()) if s.strip()]
    return 1 <= len(sentences) <= 2


INSTRUCTIONS = [
    ("Write a haiku about debugging code.", is_haiku),
    ("List three tips for staying focused while studying.", has_three_tips),
    (
        "Explain what a neural network is to a 10-year-old in two sentences.",
        at_most_two_sentences,
    ),
]


def load(name, revision, device):  # pragma: no cover - 3 GB download, not in CI
    from transformers import AutoModelForCausalLM, AutoTokenizer

    try:  # the first run downloads the model
        tok = AutoTokenizer.from_pretrained(name, revision=revision)
        model = AutoModelForCausalLM.from_pretrained(
            name, revision=revision, dtype=torch.bfloat16
        )
    except OSError as err:  # no network, the Hub is down, or the disk is full
        raise SystemExit(f"Could not load {name}: {err}") from err
    return tok, model.to(device).eval()


@torch.no_grad()
def reply(tok, model, instruction, chat):  # pragma: no cover - needs the model
    if chat:  # the instruct model expects its chat format
        messages = [{"role": "user", "content": instruction}]
        text = tok.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    else:  # a base model only ever saw plain text
        text = instruction + "\n"
    ids = tok(text, return_tensors="pt").to(model.device)
    out = model.generate(
        **ids, max_new_tokens=MAX_NEW, do_sample=False, pad_token_id=tok.eos_token_id
    )
    new = tok.decode(out[0, ids.input_ids.shape[1] :], skip_special_tokens=True)
    return new.strip()


def answers(name, revision, device, chat):  # pragma: no cover - needs the model
    tok, model = load(name, revision, device)
    result = [reply(tok, model, instruction, chat) for instruction, _ in INSTRUCTIONS]
    del model
    if device == "mps":  # give its memory back before the next model loads
        torch.mps.empty_cache()
    elif device == "cuda":
        torch.cuda.empty_cache()
    return result


def main(argv=None):  # pragma: no cover - needs the models
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
    base = answers(*BASE, args.device, chat=False)
    instruct = answers(*INSTRUCT, args.device, chat=True)

    before = after = 0
    for (instruction, check), b, i in zip(INSTRUCTIONS, base, instruct):
        before += check(b)
        after += check(i)
        print(f"\n### {instruction}  (check: {check.__name__})")
        print(f"--- {BASE[0]}: {'pass' if check(b) else 'fail'} ---\n{b}")
        print(f"--- {INSTRUCT[0]}: {'pass' if check(i) else 'fail'} ---\n{i}")

    n = len(INSTRUCTIONS)
    print(f"\nchecks passed: Pythia {before}/{n}, Qwen Instruct {after}/{n}")
    print(
        f"RESULT before={before} after={after} seconds={time.time() - t0:.0f} "
        f"device={args.device} checks={n}"
    )


if __name__ == "__main__":
    main()
