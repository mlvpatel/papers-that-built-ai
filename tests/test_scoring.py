"""Unit tests for the answer parsing and scoring in the Act 2 demos.

The demos themselves load 1.5B-parameter models, too big for CI, so only their
pure helpers are tested here.
"""

import pytest
from demo_loader import load

cot = load("10-chain-of-thought")
r1 = load("11-deepseek-r1")
# Steps 10 and 11 each keep their own copy of number(); these tests hold both to it.
NUMBER = pytest.mark.parametrize("number", [cot.number, r1.number], ids=["10", "11"])


@NUMBER
def test_number_reads_the_last_boxed_answer(number):
    assert number(r"so 3 + 4 = 7, and \boxed{12} not \boxed{24}") == 24


@NUMBER
def test_number_ignores_text_outside_the_box(number):
    assert number(r"step 1 gives 5, step 2 gives 9. \boxed{14}") == 14


@NUMBER
def test_number_strips_thousands_commas(number):
    assert number("The sum is 1,234,567.") == 1234567


@NUMBER
def test_number_keeps_decimals_and_signs(number):
    assert number("It costs 54.5 dollars") == 54.5
    assert number("The answer is -19") == -19


@NUMBER
def test_number_takes_the_last_number_without_a_box(number):
    assert number("First 3 apples, then 9 apples") == 9


def test_number_takes_the_first_number_when_asked():
    assert cot.number("24. Because 4 * 6 = 24 and 2 more", first=True) == 24


@NUMBER
def test_number_returns_none_when_there_is_no_number(number):
    assert number("I am not sure.") is None
    assert number(r"\boxed{x}") is None


def test_is_right_compares_numbers_with_a_small_tolerance():
    assert cot.is_right(54.0, 54)
    assert cot.is_right(0.1 + 0.2, 0.3)
    assert not cot.is_right(53.0, 54)
    assert not cot.is_right(None, 54)


def test_cot_problem_set_has_its_twelve_keys():
    # fixed before the first run; the published results depend on them
    keys = [24, 15, 162, 36, 54, 75, 6, 9368, 24, 9, 72, 46]
    assert [answer for _, answer in cot.PROBLEMS] == keys


gpt3 = load("08-gpt3")


def test_answer_keeps_only_the_first_line_lowercased():
    assert gpt3.answer(" Positive\nInput: next review") == "positive"


def test_passes_needs_the_answer_at_the_start():
    assert gpt3.passes(" positive.", "positive")
    assert gpt3.passes("2021-03-05\nInput:", "2021-03-05")
    assert not gpt3.passes("the review is positive", "positive")
    assert not gpt3.passes("2021-03-0", "2021-03-05")


def test_prompts_differ_only_in_the_examples():
    zero = gpt3.prompt("Task.", "x", [])
    few = gpt3.prompt("Task.", "x", [("a", "b")])
    assert zero == "Task.\nInput: x\nOutput:"
    assert few == "Task.\nInput: a\nOutput: b\nInput: x\nOutput:"


def test_each_task_has_eight_items_and_no_example_in_the_test_set():
    for _, items, shots, _ in gpt3.TASKS:
        assert len(items) == 8
        assert not {x for x, _ in shots} & {x for x, _ in items}


inst = load("09-instructgpt")


def test_haiku_check_needs_exactly_three_lines():
    assert inst.is_haiku(
        "Bug hides in the loop\n\nprint statements light the dark\nfound it at midnight"
    )
    assert not inst.is_haiku("One line only.")
    assert not inst.is_haiku("a\nb\nc\nd")


def test_tips_check_counts_numbered_or_bulleted_items():
    assert inst.has_three_tips("Tips:\n1. Sleep\n2) Eat\n- Walk")
    assert not inst.has_three_tips("1. a\n2. b\n3. c\n4. d")
    assert not inst.has_three_tips("Just focus and sleep well.")


def test_sentence_check_allows_one_or_two_sentences():
    assert inst.at_most_two_sentences(
        "It is a brain made of math. It learns from examples!"
    )
    assert inst.at_most_two_sentences("It learns from examples")
    assert not inst.at_most_two_sentences("One. Two. Three.")
    assert not inst.at_most_two_sentences("   ")


def test_every_instruction_has_a_check():
    assert len(inst.INSTRUCTIONS) == 3
    assert all(callable(check) for _, check in inst.INSTRUCTIONS)


def test_r1_excerpt_shows_text_around_the_first_wait():
    text = "a" * 500 + " Wait, that is wrong. " + "b" * 500
    shown = r1.excerpt(text, width=40)
    assert "Wait, that is wrong." in shown
    assert len(shown) < 120


def test_r1_excerpt_is_empty_when_the_model_never_waits():
    assert r1.excerpt("straight to the answer") == ""


def test_r1_problem_set_has_four_numeric_keys():
    assert len(r1.PROBLEMS) == 4
    assert [answer for _, answer in r1.PROBLEMS] == [38, 97, 927, 49]
