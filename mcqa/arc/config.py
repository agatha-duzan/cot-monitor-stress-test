"""ARC Challenge dataset configuration for MCQA monitor experiments.

Dataset: ARC (AI2 Reasoning Challenge) - Challenge subset
Source: allenai/ai2_arc on HuggingFace

TODO: Implement this config for ARC Challenge experiments.
"""

from typing import Any

from inspect_ai.dataset import Sample


# Dataset configuration
DATASET_PATH = "allenai/ai2_arc"
DATASET_NAME = "ARC-Challenge"
DATASET_SPLIT = "test"
NUM_CHOICES = 4


def record_to_sample(record: dict[str, Any]) -> Sample:
    """Convert HuggingFace ARC record to inspect_ai Sample.

    ARC format:
    {
        "id": "Mercury_7175875",
        "question": "An astronomer observes...",
        "choices": {
            "text": ["choice1", "choice2", "choice3", "choice4"],
            "label": ["A", "B", "C", "D"]
        },
        "answerKey": "A"
    }
    """
    choices = record["choices"]["text"]
    labels = record["choices"]["label"]

    # Map label to index for target
    label_to_letter = {label: letter for label, letter in zip(labels, ["A", "B", "C", "D"])}
    target = label_to_letter.get(record["answerKey"], record["answerKey"])

    return Sample(
        input=record["question"],
        choices=choices,
        target=target,
        id=record.get("id"),
    )
