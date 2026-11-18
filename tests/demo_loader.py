"""Shared by the test files: load a demo by folder, and the RESULT line format."""

import importlib.util
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULT = re.compile(
    r"^RESULT before=-?[0-9.]+ after=-?[0-9.]+ seconds=[0-9]+ device=\w+( \w+=\S+)*$"
)


def load(where):
    """Import a demo by folder name (papers/<where>/demo.py) or by its path.

    papers/ is not a package, so demos are loaded from their files.
    """
    path = where if isinstance(where, Path) else ROOT / "papers" / where / "demo.py"
    spec = importlib.util.spec_from_file_location(f"demo_{path.parent.name}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
