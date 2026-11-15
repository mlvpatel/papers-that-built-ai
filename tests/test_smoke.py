"""Run every Act 1 demo in --smoke mode on CPU and check its RESULT line.

Demos are found by folder name, so a new step needs no change here.
"""

import importlib.util
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
DEMOS = sorted(ROOT.glob("papers/0[0-7]-*/demo.py")) + sorted(
    ROOT.glob("papers/annotated_gpt.py")
)
RESULT = re.compile(
    r"^RESULT before=[0-9.]+ after=[0-9.]+ seconds=[0-9]+ device=\w+( \w+=\S+)*$"
)


def test_at_least_one_demo_is_found():
    assert DEMOS, "no Act 1 demos found under papers/"


@pytest.mark.parametrize("path", DEMOS, ids=lambda p: p.parent.name)
def test_demo_smoke_run_prints_a_result_line(path, capsys):
    spec = importlib.util.spec_from_file_location(f"demo_{path.parent.name}", path)
    demo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(demo)

    demo.main(["--smoke", "--device", "cpu"])

    last_line = capsys.readouterr().out.strip().splitlines()[-1]
    assert RESULT.match(last_line), last_line
