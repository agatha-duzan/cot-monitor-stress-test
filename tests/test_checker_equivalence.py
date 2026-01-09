"""Tests for task checker function equivalence.

Verifies that the new checker functions in scorers/task_checkers.py
produce identical results to the original closures created by task makers.
"""

import pytest
from random import Random

# Original implementation - task makers that create checkers as closures
from cot_obfuscation.instruction_following import (
    make_sum_instance,
    make_sort_words_instance,
    make_count_vowels_instance,
    make_system2x2_int_instance,
)

# New implementation - standalone checker functions
from cot_obfuscation.scorers.task_checkers import (
    check_sum,
    check_sort_words,
    check_count_vowels,
    check_system2x2,
    extract_answer_block,
)


class TestAnswerExtraction:
    """Test answer block extraction works identically."""

    @pytest.mark.parametrize(
        "text,expected",
        [
            ("<answer>42</answer>", "42"),
            ("Some text <answer>result</answer> more text", "result"),
            ("<ANSWER>UPPER</ANSWER>", "UPPER"),
            ("<Answer>Mixed</Answer>", "Mixed"),
            ("No answer tags here", None),
            ("", None),
            (None, None),
            ("<answer>  spaces  </answer>", "spaces"),
            ("<answer>\nmultiline\n</answer>", "multiline"),
        ],
    )
    def test_extract_answer_block(self, text, expected):
        """Answer extraction should handle various formats."""
        result = extract_answer_block(text)
        assert result == expected


class TestSumChecker:
    """Test sum checker equivalence."""

    def test_correct_answer(self, rng):
        """Correct answers should match."""
        task = make_sum_instance(rng)

        # Test with original checker (closure)
        test_output = f"<answer>{task.answer_str}</answer>"
        orig_ok, orig_details = task.checker(test_output)

        # Test with new checker (standalone)
        new_ok, new_details = check_sum(test_output, task.answer_str)

        assert orig_ok == new_ok == True
        assert orig_details["expected"] == new_details["expected"]
        assert orig_details["parsed"] == new_details["parsed"]

    def test_wrong_answer(self, rng):
        """Wrong answers should be rejected identically."""
        task = make_sum_instance(rng)
        wrong = int(task.answer_str) + 1
        test_output = f"<answer>{wrong}</answer>"

        orig_ok, orig_details = task.checker(test_output)
        new_ok, new_details = check_sum(test_output, task.answer_str)

        assert orig_ok == new_ok == False
        assert orig_details["parsed"] == new_details["parsed"] == wrong

    def test_no_answer_tags(self, rng):
        """Missing answer tags should fail identically."""
        task = make_sum_instance(rng)
        test_output = f"The answer is {task.answer_str}"

        orig_ok, _ = task.checker(test_output)
        new_ok, _ = check_sum(test_output, task.answer_str)

        assert orig_ok == new_ok == False

    def test_negative_numbers(self):
        """Negative numbers should be parsed correctly."""
        rng = Random(123)  # Seed that produces negative sum
        # Generate until we get a negative sum
        for _ in range(100):
            task = make_sum_instance(rng)
            if int(task.answer_str) < 0:
                break

        test_output = f"<answer>{task.answer_str}</answer>"
        orig_ok, orig_details = task.checker(test_output)
        new_ok, new_details = check_sum(test_output, task.answer_str)

        assert orig_ok == new_ok == True


class TestSortWordsChecker:
    """Test sort words checker equivalence."""

    def test_correct_answer(self, rng):
        """Correctly sorted words should match."""
        task = make_sort_words_instance(rng)
        test_output = f"<answer>{task.answer_str}</answer>"

        orig_ok, orig_details = task.checker(test_output)
        new_ok, new_details = check_sort_words(test_output, task.answer_str)

        assert orig_ok == new_ok == True
        assert orig_details["expected"] == new_details["expected"]

    def test_wrong_order(self, rng):
        """Wrong order should fail identically."""
        task = make_sort_words_instance(rng)
        # Reverse the correct order
        words = task.answer_str.split(", ")
        wrong_order = ", ".join(reversed(words))
        test_output = f"<answer>{wrong_order}</answer>"

        orig_ok, _ = task.checker(test_output)
        new_ok, _ = check_sort_words(test_output, task.answer_str)

        assert orig_ok == new_ok == False

    def test_case_insensitive(self, rng):
        """Comparison should be case insensitive."""
        task = make_sort_words_instance(rng)
        # Make uppercase
        upper_answer = task.answer_str.upper()
        test_output = f"<answer>{upper_answer}</answer>"

        orig_ok, _ = task.checker(test_output)
        new_ok, _ = check_sort_words(test_output, task.answer_str)

        assert orig_ok == new_ok == True

    def test_extra_spaces(self, rng):
        """Extra spaces should be handled."""
        task = make_sort_words_instance(rng)
        # Add extra spaces
        spaced = task.answer_str.replace(",", " , ")
        test_output = f"<answer>{spaced}</answer>"

        orig_ok, _ = task.checker(test_output)
        new_ok, _ = check_sort_words(test_output, task.answer_str)

        assert orig_ok == new_ok == True


class TestCountVowelsChecker:
    """Test vowel counting checker equivalence."""

    def test_correct_answer(self, rng):
        """Correct count should match."""
        task = make_count_vowels_instance(rng)
        test_output = f"<answer>{task.answer_str}</answer>"

        orig_ok, orig_details = task.checker(test_output)
        new_ok, new_details = check_count_vowels(test_output, task.answer_str)

        assert orig_ok == new_ok == True
        assert orig_details["expected"] == new_details["expected"]
        assert orig_details["parsed"] == new_details["parsed"]

    def test_wrong_count(self, rng):
        """Wrong count should fail identically."""
        task = make_count_vowels_instance(rng)
        wrong = int(task.answer_str) + 5
        test_output = f"<answer>{wrong}</answer>"

        orig_ok, orig_details = task.checker(test_output)
        new_ok, new_details = check_count_vowels(test_output, task.answer_str)

        assert orig_ok == new_ok == False
        assert orig_details["parsed"] == new_details["parsed"] == wrong


class TestSystem2x2Checker:
    """Test system of equations checker equivalence."""

    def test_correct_answer(self, rng):
        """Correct solution should match."""
        task = make_system2x2_int_instance(rng)
        test_output = f"<answer>{task.answer_str}</answer>"

        orig_ok, orig_details = task.checker(test_output)
        new_ok, new_details = check_system2x2(test_output, task.answer_str)

        assert orig_ok == new_ok == True
        assert orig_details["expected"] == new_details["expected"]

    def test_wrong_x(self, rng):
        """Wrong x value should fail."""
        task = make_system2x2_int_instance(rng)
        # Parse correct values and modify x
        import re
        m = re.search(r"x=([+-]?\d+), y=([+-]?\d+)", task.answer_str)
        x_val, y_val = int(m.group(1)), int(m.group(2))
        wrong_answer = f"x={x_val + 1}, y={y_val}"
        test_output = f"<answer>{wrong_answer}</answer>"

        orig_ok, _ = task.checker(test_output)
        new_ok, _ = check_system2x2(test_output, task.answer_str)

        assert orig_ok == new_ok == False

    def test_wrong_y(self, rng):
        """Wrong y value should fail."""
        task = make_system2x2_int_instance(rng)
        import re
        m = re.search(r"x=([+-]?\d+), y=([+-]?\d+)", task.answer_str)
        x_val, y_val = int(m.group(1)), int(m.group(2))
        wrong_answer = f"x={x_val}, y={y_val + 1}"
        test_output = f"<answer>{wrong_answer}</answer>"

        orig_ok, _ = task.checker(test_output)
        new_ok, _ = check_system2x2(test_output, task.answer_str)

        assert orig_ok == new_ok == False

    def test_decimal_tolerance(self, rng):
        """Small decimal differences should be tolerated."""
        task = make_system2x2_int_instance(rng)
        import re
        m = re.search(r"x=([+-]?\d+), y=([+-]?\d+)", task.answer_str)
        x_val, y_val = int(m.group(1)), int(m.group(2))
        # Add tiny decimal difference
        test_answer = f"x={x_val}.00001, y={y_val}.00001"
        test_output = f"<answer>{test_answer}</answer>"

        orig_ok, _ = task.checker(test_output)
        new_ok, _ = check_system2x2(test_output, task.answer_str)

        # Both should accept within tolerance
        assert orig_ok == new_ok == True

    def test_negative_solutions(self):
        """Negative solutions should be handled."""
        # Generate until we get negative values
        for seed in range(200):
            rng = Random(seed)
            task = make_system2x2_int_instance(rng)
            import re
            m = re.search(r"x=([+-]?\d+), y=([+-]?\d+)", task.answer_str)
            if m:
                x_val, y_val = int(m.group(1)), int(m.group(2))
                if x_val < 0 and y_val < 0:
                    break

        test_output = f"<answer>{task.answer_str}</answer>"
        orig_ok, _ = task.checker(test_output)
        new_ok, _ = check_system2x2(test_output, task.answer_str)

        assert orig_ok == new_ok == True

    def test_spacing_variations(self, rng):
        """Various spacing in answer should be handled."""
        task = make_system2x2_int_instance(rng)
        import re
        m = re.search(r"x=([+-]?\d+), y=([+-]?\d+)", task.answer_str)
        x_val, y_val = int(m.group(1)), int(m.group(2))

        # Test different spacing
        variations = [
            f"x = {x_val}, y = {y_val}",
            f"x={x_val},y={y_val}",
            f"x =  {x_val} , y =  {y_val}",
        ]

        for variant in variations:
            test_output = f"<answer>{variant}</answer>"
            orig_ok, _ = task.checker(test_output)
            new_ok, _ = check_system2x2(test_output, task.answer_str)
            assert orig_ok == new_ok, f"Mismatch for: {variant}"


class TestMultipleSeeds:
    """Test with multiple random seeds for robustness."""

    @pytest.mark.parametrize("seed", [0, 42, 100, 999, 12345])
    def test_sum_multiple_seeds(self, seed):
        """Sum checker should match across seeds."""
        rng = Random(seed)
        task = make_sum_instance(rng)
        test_output = f"<answer>{task.answer_str}</answer>"

        orig_ok, _ = task.checker(test_output)
        new_ok, _ = check_sum(test_output, task.answer_str)

        assert orig_ok == new_ok == True

    @pytest.mark.parametrize("seed", [0, 42, 100, 999, 12345])
    def test_sort_multiple_seeds(self, seed):
        """Sort checker should match across seeds."""
        rng = Random(seed)
        task = make_sort_words_instance(rng)
        test_output = f"<answer>{task.answer_str}</answer>"

        orig_ok, _ = task.checker(test_output)
        new_ok, _ = check_sort_words(test_output, task.answer_str)

        assert orig_ok == new_ok == True

    @pytest.mark.parametrize("seed", [0, 42, 100, 999, 12345])
    def test_vowels_multiple_seeds(self, seed):
        """Vowels checker should match across seeds."""
        rng = Random(seed)
        task = make_count_vowels_instance(rng)
        test_output = f"<answer>{task.answer_str}</answer>"

        orig_ok, _ = task.checker(test_output)
        new_ok, _ = check_count_vowels(test_output, task.answer_str)

        assert orig_ok == new_ok == True

    @pytest.mark.parametrize("seed", [0, 42, 100, 999, 12345])
    def test_system2x2_multiple_seeds(self, seed):
        """System2x2 checker should match across seeds."""
        rng = Random(seed)
        task = make_system2x2_int_instance(rng)
        test_output = f"<answer>{task.answer_str}</answer>"

        orig_ok, _ = task.checker(test_output)
        new_ok, _ = check_system2x2(test_output, task.answer_str)

        assert orig_ok == new_ok == True


class TestEdgeCases:
    """Test edge cases for checkers."""

    def test_empty_answer_block(self):
        """Empty answer block should fail gracefully."""
        test_output = "<answer></answer>"

        # Sum
        ok, details = check_sum(test_output, "42")
        assert ok == False
        assert details["parsed"] is None

        # Sort words
        ok, details = check_sort_words(test_output, "apple, banana")
        assert ok == False

        # Count vowels
        ok, details = check_count_vowels(test_output, "5")
        assert ok == False
        assert details["parsed"] is None

    def test_malformed_answer(self):
        """Malformed answers should fail gracefully."""
        # Sum with non-numeric
        ok, details = check_sum("<answer>not a number</answer>", "42")
        assert ok == False
        assert details["parsed"] is None

        # System2x2 with missing y
        ok, details = check_system2x2("<answer>x=5</answer>", "x=5, y=3")
        assert ok == False

    def test_multiple_answer_blocks(self):
        """First answer block should be used."""
        test_output = "<answer>42</answer> and also <answer>99</answer>"
        ok, details = check_sum(test_output, "42")
        assert ok == True
        assert details["parsed"] == 42
