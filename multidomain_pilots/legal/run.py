#!/usr/bin/env python3
"""Run legal domain pilot.

Special handling: third-person framing uses adapted nudges that refer to
the friend instead of the user.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.resolve()))

from config import DOMAIN, STIMULUS_PATH, FRAMINGS, CONDITIONS, SCALAR_RANGE, THIRD_PERSON_OVERRIDES
from shared.runner import run_domain
from shared.extractor import extract_domain
from shared.analyzer import analyze_domain


def _build_third_person_conditions():
    tp = {}
    for cond, (cat, nudge) in CONDITIONS.items():
        if cond in THIRD_PERSON_OVERRIDES:
            tp[cond] = (cat, THIRD_PERSON_OVERRIDES[cond])
        else:
            tp[cond] = (cat, nudge)
    return tp


async def run(n_samples=20, concurrency=15, dry_run=False, resume=False):
    total = 0
    fp_framings = {"first_person": FRAMINGS["first_person"]}
    n = await run_domain(
        DOMAIN, STIMULUS_PATH, fp_framings, CONDITIONS,
        n_samples=n_samples, concurrency=concurrency,
        dry_run=dry_run, resume=resume,
    )
    total += n if n else 0

    tp_framings = {"third_person": FRAMINGS["third_person"]}
    tp_conditions = _build_third_person_conditions()
    n = await run_domain(
        DOMAIN, STIMULUS_PATH, tp_framings, tp_conditions,
        n_samples=n_samples, concurrency=concurrency,
        dry_run=dry_run, resume=resume,
    )
    total += n if n else 0
    return total


def extract():
    return extract_domain(DOMAIN, CONDITIONS, FRAMINGS, SCALAR_RANGE)


def analyze():
    return analyze_domain(DOMAIN, CONDITIONS, FRAMINGS, SCALAR_RANGE)


if __name__ == "__main__":
    import asyncio
    asyncio.run(run())
