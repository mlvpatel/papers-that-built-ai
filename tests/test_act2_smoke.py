"""Run every Act 2 demo end to end with a tiny random model in place of the real one.

Each demo keeps its real tokenizer and chat template (small downloads), but the
model is the same architecture with one tiny layer and random weights, so its
answers are nonsense. This checks the plumbing (prompts, generation, parsing,
the RESULT line), not the results.
"""

import pytest
import torch
from demo_loader import RESULT, load
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

gpt3 = load("08-gpt3")
inst = load("09-instructgpt")
cot = load("10-chain-of-thought")
r1 = load("11-deepseek-r1")

TINY = {
    "hidden_size": 16,
    "intermediate_size": 32,
    "num_hidden_layers": 1,
    "num_attention_heads": 2,
    "num_key_value_heads": 1,
}


def tiny(name, revision, device):
    tok = AutoTokenizer.from_pretrained(name, revision=revision)
    config = AutoConfig.from_pretrained(name, revision=revision, **TINY)
    torch.manual_seed(0)
    return tok, AutoModelForCausalLM.from_config(config).to(device).eval()


# (demo, replacement for its load(), token limits cut to keep the test fast)
CASES = [
    (gpt3, lambda device: tiny(gpt3.MODEL, gpt3.REVISION, device), {}),
    (inst, tiny, {"MAX_NEW": 8}),
    (
        cot,
        lambda device: tiny(cot.MODEL, cot.REVISION, device),
        {"DIRECT_TOKENS": 8, "STEP_TOKENS": 8},
    ),
    (r1, tiny, {"INSTRUCT_TOKENS": 8, "R1_TOKENS": 8}),
]


@pytest.mark.parametrize(
    "demo, fake_load, limits",
    CASES,
    ids=[demo.__name__.removeprefix("demo_") for demo, _, _ in CASES],
)
def test_act2_demo_runs_end_to_end(demo, fake_load, limits, monkeypatch, capsys):
    monkeypatch.setattr(demo, "load", fake_load)
    for name, value in limits.items():
        monkeypatch.setattr(demo, name, value)

    demo.main(["--device", "cpu"])

    last_line = capsys.readouterr().out.strip().splitlines()[-1]
    assert RESULT.match(last_line), last_line
