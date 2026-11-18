"""Shared by the test files: load a demo by folder, and the RESULT line format."""

import importlib.util
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULT = re.compile(
    r"^RESULT before=-?[0-9.]+ after=-?[0-9.]+ seconds=[0-9]+ device=\w+( \w+=\S+)*$"
)


def load(folder):
    """Import papers/<folder>/demo.py; papers/ is not a package."""
    path = ROOT / "papers" / folder / "demo.py"
    spec = importlib.util.spec_from_file_location(f"demo_{folder}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
