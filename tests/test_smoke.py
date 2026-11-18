"""Run every Act 1 demo in --smoke mode on CPU and check its RESULT line.

Demos are found by folder name, so a new step needs no change here.
"""

import pytest
import torch
from demo_loader import RESULT, ROOT, load

GPU = (
    "mps"
    if torch.backends.mps.is_available()
    else "cuda"
    if torch.cuda.is_available()
    else None
)
DEVICES = ["cpu"] + ([GPU] if GPU else [])  # CI has no GPU; a laptop runs both
DEMOS = sorted(ROOT.glob("papers/0[0-7]-*/demo.py")) + sorted(
    ROOT.glob("papers/annotated_gpt.py")
)


def test_at_least_one_demo_is_found():
    assert DEMOS, "no Act 1 demos found under papers/"


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize(
    "path", DEMOS, ids=lambda p: p.stem if p.parent.name == "papers" else p.parent.name
)
def test_demo_smoke_run_prints_a_result_line(path, device, capsys):
    demo = load(path)

    demo.main(["--smoke", "--device", device])

    last_line = capsys.readouterr().out.strip().splitlines()[-1]
    assert RESULT.match(last_line), last_line
