#!/usr/bin/env python3
"""Run hiring domain pilot."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.resolve()))

from config import DOMAIN, STIMULUS_PATH, FRAMINGS, CONDITIONS, SCALAR_RANGE
from shared.runner import run_domain
from shared.extractor import extract_domain
from shared.analyzer import analyze_domain


async def run(n_samples=20, concurrency=15, dry_run=False, resume=False):
    return await run_domain(
        DOMAIN, STIMULUS_PATH, FRAMINGS, CONDITIONS,
        n_samples=n_samples, concurrency=concurrency,
        dry_run=dry_run, resume=resume,
    )


def extract():
    return extract_domain(DOMAIN, CONDITIONS, FRAMINGS, SCALAR_RANGE)


def analyze():
    return analyze_domain(DOMAIN, CONDITIONS, FRAMINGS, SCALAR_RANGE)


if __name__ == "__main__":
    import asyncio
    asyncio.run(run())
