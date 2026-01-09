"""Pytest configuration and shared fixtures for equivalence tests.

These tests verify that the refactored inspect_ai code produces
equivalent results to the original implementation.

NOTE: These tests do NOT require API keys - they test prompt building,
checker functions, and dataset generation without making API calls.
"""

import pytest
from random import Random


@pytest.fixture
def rng():
    """Provide a seeded RNG for reproducible tests."""
    return Random(42)


@pytest.fixture
def seed():
    """Standard seed for tests."""
    return 42
