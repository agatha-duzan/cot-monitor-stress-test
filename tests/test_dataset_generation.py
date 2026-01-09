"""Tests for dataset generation determinism and equivalence.

Verifies that:
1. Dataset generation is deterministic (same seed = same samples)
2. New task generators produce identical results to original implementations
3. Obfuscation task definitions match the original
"""

import pytest
from random import Random

# Original implementations
from cot_obfuscation.instruction_following import (
    make_sum_instance,
    make_sort_words_instance,
    make_count_vowels_instance,
    make_system2x2_int_instance,
    all_tasks as orig_all_tasks,
)
from cot_obfuscation.obfuscation import user_hidden_tasks as ORIG_OBFUSCATION_TASKS

# New implementations
from cot_obfuscation.datasets.instruction_tasks import (
    make_sum_task,
    make_sort_words_task,
    make_count_vowels_task,
    make_system2x2_task,
    all_task_makers,
    generate_instruction_samples,
    InstructionTask,
)
from cot_obfuscation.datasets.obfuscation_tasks import (
    all_obfuscation_tasks,
    get_obfuscation_tasks,
    generate_obfuscation_samples,
)
from cot_obfuscation.datasets.settings import all_settings


class TestInstructionTaskDeterminism:
    """Test that instruction task generation is deterministic."""

    @pytest.mark.parametrize("seed", [42, 0, 100, 999])
    def test_system2x2_deterministic(self, seed):
        """System2x2 task should be identical with same seed."""
        rng1 = Random(seed)
        rng2 = Random(seed)

        task1 = make_system2x2_task(rng1)
        task2 = make_system2x2_task(rng2)

        assert task1.id == task2.id
        assert task1.prompt == task2.prompt
        assert task1.answer_str == task2.answer_str
        assert task1.banned_words == task2.banned_words

    @pytest.mark.parametrize("seed", [42, 0, 100, 999])
    def test_sum_deterministic(self, seed):
        """Sum task should be identical with same seed."""
        rng1 = Random(seed)
        rng2 = Random(seed)

        task1 = make_sum_task(rng1)
        task2 = make_sum_task(rng2)

        assert task1.id == task2.id
        assert task1.prompt == task2.prompt
        assert task1.answer_str == task2.answer_str

    @pytest.mark.parametrize("seed", [42, 0, 100, 999])
    def test_sort_words_deterministic(self, seed):
        """Sort words task should be identical with same seed."""
        rng1 = Random(seed)
        rng2 = Random(seed)

        task1 = make_sort_words_task(rng1)
        task2 = make_sort_words_task(rng2)

        assert task1.id == task2.id
        assert task1.prompt == task2.prompt
        assert task1.answer_str == task2.answer_str

    @pytest.mark.parametrize("seed", [42, 0, 100, 999])
    def test_count_vowels_deterministic(self, seed):
        """Count vowels task should be identical with same seed."""
        rng1 = Random(seed)
        rng2 = Random(seed)

        task1 = make_count_vowels_task(rng1)
        task2 = make_count_vowels_task(rng2)

        assert task1.id == task2.id
        assert task1.prompt == task2.prompt
        assert task1.answer_str == task2.answer_str


class TestInstructionTaskEquivalence:
    """Test that new task generators match original implementations."""

    @pytest.mark.parametrize("seed", [42, 0, 100])
    def test_system2x2_equivalence(self, seed):
        """New system2x2 generator should match original."""
        rng_orig = Random(seed)
        rng_new = Random(seed)

        orig = make_system2x2_int_instance(rng_orig)
        new = make_system2x2_task(rng_new)

        assert orig.id == new.id
        assert orig.prompt == new.prompt
        assert orig.answer_str == new.answer_str

    @pytest.mark.parametrize("seed", [42, 0, 100])
    def test_sum_equivalence(self, seed):
        """New sum generator should match original."""
        rng_orig = Random(seed)
        rng_new = Random(seed)

        orig = make_sum_instance(rng_orig)
        new = make_sum_task(rng_new)

        assert orig.id == new.id
        assert orig.prompt == new.prompt
        assert orig.answer_str == new.answer_str

    @pytest.mark.parametrize("seed", [42, 0, 100])
    def test_sort_words_equivalence(self, seed):
        """New sort words generator should match original."""
        rng_orig = Random(seed)
        rng_new = Random(seed)

        orig = make_sort_words_instance(rng_orig)
        new = make_sort_words_task(rng_new)

        assert orig.id == new.id
        assert orig.prompt == new.prompt
        assert orig.answer_str == new.answer_str

    @pytest.mark.parametrize("seed", [42, 0, 100])
    def test_count_vowels_equivalence(self, seed):
        """New count vowels generator should match original."""
        rng_orig = Random(seed)
        rng_new = Random(seed)

        orig = make_count_vowels_instance(rng_orig)
        new = make_count_vowels_task(rng_new)

        assert orig.id == new.id
        assert orig.prompt == new.prompt
        assert orig.answer_str == new.answer_str


class TestDatasetGeneration:
    """Test dataset generation functionality."""

    def test_generate_instruction_samples_deterministic(self):
        """Same seed should produce identical datasets."""
        ds1 = generate_instruction_samples(
            steering_id="test",
            steering_text="Test instruction",
            steering_family="test",
            runs_per_task=2,
            seed=42,
        )
        ds2 = generate_instruction_samples(
            steering_id="test",
            steering_text="Test instruction",
            steering_family="test",
            runs_per_task=2,
            seed=42,
        )

        samples1 = list(ds1)
        samples2 = list(ds2)

        assert len(samples1) == len(samples2)
        for s1, s2 in zip(samples1, samples2):
            assert s1.id == s2.id
            assert s1.input == s2.input
            assert s1.target == s2.target
            assert s1.metadata == s2.metadata

    def test_different_seeds_produce_different_samples(self):
        """Different seeds should produce different task instances."""
        ds1 = generate_instruction_samples(
            steering_id="test",
            steering_text="Test",
            steering_family="test",
            runs_per_task=1,
            seed=42,
        )
        ds2 = generate_instruction_samples(
            steering_id="test",
            steering_text="Test",
            steering_family="test",
            runs_per_task=1,
            seed=100,
        )

        samples1 = list(ds1)
        samples2 = list(ds2)

        # At least some samples should be different
        different_count = sum(
            1 for s1, s2 in zip(samples1, samples2) if s1.target != s2.target
        )
        assert different_count > 0, "Different seeds should produce different tasks"

    def test_sample_metadata_contents(self):
        """Samples should have expected metadata fields."""
        ds = generate_instruction_samples(
            steering_id="lang_pl",
            steering_text="Use Polish",
            steering_family="language",
            runs_per_task=1,
            seed=42,
        )

        for sample in ds:
            assert "task_id" in sample.metadata
            assert "task_prompt" in sample.metadata
            assert "steering_id" in sample.metadata
            assert "steering_text" in sample.metadata
            assert "steering_family" in sample.metadata
            assert "banned_words" in sample.metadata
            assert sample.metadata["steering_id"] == "lang_pl"
            assert sample.metadata["steering_family"] == "language"

    def test_runs_per_task_multiplies_samples(self):
        """Multiple runs should produce multiple samples per task type."""
        ds1 = generate_instruction_samples(
            steering_id="test",
            steering_text="Test",
            steering_family="test",
            runs_per_task=1,
            seed=42,
        )
        ds2 = generate_instruction_samples(
            steering_id="test",
            steering_text="Test",
            steering_family="test",
            runs_per_task=3,
            seed=42,
        )

        # 4 task types * runs_per_task
        assert len(list(ds1)) == 4 * 1
        assert len(list(ds2)) == 4 * 3


class TestObfuscationTaskEquivalence:
    """Test that obfuscation task definitions match original."""

    def test_task_count_matches(self):
        """Number of tasks should match original."""
        assert len(all_obfuscation_tasks) == len(ORIG_OBFUSCATION_TASKS)

    def test_user_tasks_match(self):
        """User task texts should match original."""
        for i, (new_task, orig_task) in enumerate(
            zip(all_obfuscation_tasks, ORIG_OBFUSCATION_TASKS)
        ):
            assert new_task.user_task == orig_task.user_task, f"Mismatch at task {i}"

    def test_hidden_tasks_match(self):
        """Hidden task texts should match original."""
        for i, (new_task, orig_task) in enumerate(
            zip(all_obfuscation_tasks, ORIG_OBFUSCATION_TASKS)
        ):
            assert new_task.hidden_task == orig_task.hidden_task, f"Mismatch at task {i}"

    def test_risk_classifications(self):
        """Task risk classifications should be appropriate."""
        benign_ids = {0, 1, 2, 3, 4, 5, 9}  # Tasks 1-6, 10
        sensitive_ids = {6, 7, 8, 10, 11}   # Tasks 7-9, 11-12

        for task in all_obfuscation_tasks:
            if task.id in benign_ids:
                assert task.risk == "benign", f"Task {task.id} should be benign"
            elif task.id in sensitive_ids:
                assert task.risk == "sensitive", f"Task {task.id} should be sensitive"

    def test_filter_sensitive_tasks(self):
        """Filtering should remove sensitive tasks."""
        all_tasks = get_obfuscation_tasks(include_sensitive=True)
        benign_only = get_obfuscation_tasks(include_sensitive=False)

        assert len(benign_only) < len(all_tasks)
        for task in benign_only:
            assert task.risk == "benign"


class TestObfuscationDatasetGeneration:
    """Test obfuscation dataset generation."""

    def test_generate_obfuscation_samples(self):
        """Should generate samples for all tasks."""
        setting_texts = all_settings.get("base", [])
        ds = generate_obfuscation_samples(
            setting_name="base",
            setting_texts=setting_texts,
            runs_per_task=1,
            include_sensitive=True,
        )

        samples = list(ds)
        assert len(samples) == len(all_obfuscation_tasks)

    def test_obfuscation_sample_metadata(self):
        """Samples should have expected metadata."""
        setting_texts = all_settings.get("base", [])
        ds = generate_obfuscation_samples(
            setting_name="base",
            setting_texts=setting_texts,
            runs_per_task=1,
        )

        for sample in ds:
            assert "task_id" in sample.metadata
            assert "user_task" in sample.metadata
            assert "hidden_task" in sample.metadata
            assert "risk" in sample.metadata
            assert "setting_name" in sample.metadata
            assert "setting_texts" in sample.metadata

    def test_exclude_sensitive_tasks(self):
        """Should exclude sensitive tasks when requested."""
        setting_texts = all_settings.get("base", [])

        ds_all = generate_obfuscation_samples(
            setting_name="base",
            setting_texts=setting_texts,
            runs_per_task=1,
            include_sensitive=True,
        )
        ds_benign = generate_obfuscation_samples(
            setting_name="base",
            setting_texts=setting_texts,
            runs_per_task=1,
            include_sensitive=False,
        )

        samples_all = list(ds_all)
        samples_benign = list(ds_benign)

        assert len(samples_benign) < len(samples_all)
        for sample in samples_benign:
            assert sample.metadata["risk"] == "benign"

    def test_runs_per_task_multiplies(self):
        """Multiple runs should produce multiple samples."""
        setting_texts = all_settings.get("base", [])

        ds1 = generate_obfuscation_samples(
            setting_name="base",
            setting_texts=setting_texts,
            runs_per_task=1,
        )
        ds3 = generate_obfuscation_samples(
            setting_name="base",
            setting_texts=setting_texts,
            runs_per_task=3,
        )

        assert len(list(ds3)) == 3 * len(list(ds1))


class TestSettingsEquivalence:
    """Test that settings definitions match."""

    def test_all_settings_keys(self):
        """All expected settings should be present."""
        expected_keys = {"base", "do_not_mention_1", "do_not_mention_2", "monitoring_1"}
        assert expected_keys.issubset(set(all_settings.keys()))

    def test_settings_have_content(self):
        """Each setting should have prompt fragments."""
        for name, texts in all_settings.items():
            assert isinstance(texts, list), f"Setting {name} should be a list"
            # Settings can have empty text lists for "base" case


class TestTaskMakerRegistry:
    """Test task maker registries match."""

    def test_task_maker_count(self):
        """Should have same number of task makers."""
        assert len(all_task_makers) == len(orig_all_tasks)

    def test_task_ids_unique(self):
        """Generated tasks should have unique IDs."""
        rng = Random(42)
        ids = set()
        for maker in all_task_makers:
            task = maker(rng)
            assert task.id not in ids, f"Duplicate task ID: {task.id}"
            ids.add(task.id)

    def test_all_tasks_have_banned_words(self):
        """All tasks should have banned words defined."""
        rng = Random(42)
        for maker in all_task_makers:
            task = maker(rng)
            assert isinstance(task.banned_words, list)
            assert len(task.banned_words) > 0, f"Task {task.id} has no banned words"


class TestBannedWordsEquivalence:
    """Test banned words definitions match original."""

    def test_sum_banned_words(self):
        """Sum task banned words should match."""
        from cot_obfuscation.instruction_following import banned_words_for_task

        rng = Random(42)
        task = make_sum_task(rng)

        orig_words = banned_words_for_task("sum_integers")
        assert task.banned_words == orig_words

    def test_sort_banned_words(self):
        """Sort task banned words should match."""
        from cot_obfuscation.instruction_following import banned_words_for_task

        rng = Random(42)
        task = make_sort_words_task(rng)

        orig_words = banned_words_for_task("sort_words")
        assert task.banned_words == orig_words

    def test_vowels_banned_words(self):
        """Vowels task banned words should match."""
        from cot_obfuscation.instruction_following import banned_words_for_task

        rng = Random(42)
        task = make_count_vowels_task(rng)

        orig_words = banned_words_for_task("count_vowels")
        assert task.banned_words == orig_words

    def test_system2x2_banned_words(self):
        """System2x2 task banned words should match."""
        from cot_obfuscation.instruction_following import banned_words_for_task

        rng = Random(42)
        task = make_system2x2_task(rng)

        orig_words = banned_words_for_task("system_2x2_int")
        assert task.banned_words == orig_words
