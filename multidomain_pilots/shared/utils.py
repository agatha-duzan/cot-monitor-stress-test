"""Shared utilities for multidomain pilots."""

import json
import time
from pathlib import Path

RESULTS_DIR = Path(__file__).parent.parent.resolve() / "results"
RAW_DIR = RESULTS_DIR / "raw"
SUMMARY_DIR = RESULTS_DIR / "summary"
PLOTS_DIR = RESULTS_DIR / "plots"

MODEL = "claude-sonnet-4-5-20250929"
TEMPERATURE = 1.0
MAX_TOKENS = 1024
DEFAULT_CONCURRENCY = 15
DEFAULT_N_SAMPLES = 20


def result_path(domain: str, framing: str, condition: str) -> Path:
    return RAW_DIR / f"{domain}_{framing}_{condition}.jsonl"


def count_existing(path: Path) -> int:
    if not path.exists():
        return 0
    with open(path) as f:
        return sum(1 for line in f if line.strip())


def load_stimulus(path: Path) -> str:
    with open(path) as f:
        return f.read().strip()
