"""Unit tests for the answer parsing and scoring in the Act 2 demos.

The demos themselves load 1.5B-parameter models, too big for CI, so only their
pure helpers are tested here.
"""

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load(folder):
    path = ROOT / "papers" / folder / "demo.py"
    spec = importlib.util.spec_from_file_location(f"demo_{folder}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


cot = load("10-chain-of-thought")


def test_number_reads_the_last_boxed_answer():
    assert cot.number(r"so 3 + 4 = 7, and \boxed{12} not \boxed{24}") == 24


def test_number_ignores_text_outside_the_box():
    assert cot.number(r"step 1 gives 5, step 2 gives 9. \boxed{14}") == 14


def test_number_strips_thousands_commas():
    assert cot.number("The sum is 1,234,567.") == 1234567


def test_number_keeps_decimals_and_signs():
    assert cot.number("It costs 54.5 dollars") == 54.5
    assert cot.number("The answer is -19") == -19


def test_number_takes_the_last_number_without_a_box():
    assert cot.number("First 3 apples, then 9 apples") == 9


def test_number_takes_the_first_number_when_asked():
    assert cot.number("24. Because 4 * 6 = 24 and 2 more", first=True) == 24


def test_number_returns_none_when_there_is_no_number():
    assert cot.number("I am not sure.") is None
    assert cot.number(r"\boxed{x}") is None


def test_is_right_compares_numbers_with_a_small_tolerance():
    assert cot.is_right(54.0, 54)
    assert cot.is_right(0.1 + 0.2, 0.3)
    assert not cot.is_right(53.0, 54)
    assert not cot.is_right(None, 54)


def test_every_problem_has_a_numeric_answer_key():
    assert len(cot.PROBLEMS) == 12
    assert all(isinstance(answer, (int, float)) for _, answer in cot.PROBLEMS)
