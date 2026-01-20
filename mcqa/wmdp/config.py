"""WMDP-chem dataset configuration for MCQA monitor experiments.

Dataset: WMDP (Weapons of Mass Destruction Proxy) - Chemistry subset
Source: cais/wmdp on HuggingFace
"""

from typing import Any

from inspect_ai.dataset import Sample


# Dataset configuration
DATASET_PATH = "cais/wmdp"
DATASET_NAME = "wmdp-chem"
DATASET_SPLIT = "test"
NUM_CHOICES = 4


def record_to_sample(record: dict[str, Any]) -> Sample:
    """Convert HuggingFace WMDP record to inspect_ai Sample."""
    choices_map = {0: "A", 1: "B", 2: "C", 3: "D"}
    return Sample(
        input=record["question"],
        choices=record["choices"],
        target=choices_map[record["answer"]],
    )
