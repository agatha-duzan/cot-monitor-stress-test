from pathlib import Path

from cot_obfuscation import (
    RunConfig,
    run_obfuscation_experiments,
    save_obfuscation_results_to_json,
)


def main() -> None:
    cfg = RunConfig(
        model_slug="anthropic/claude-sonnet-4-5-20250929",
        runs_per_instruction=1,
        internal_reasoning=True,
        temperature=1.0,
        reasoning_batch_size=12,
        monitor_batch_size=12,
        monitor_model="gpt-5-mini",
    )
    results = run_obfuscation_experiments(cfg, setting_names=["base"])
    save_obfuscation_results_to_json(results, Path("results") / "obfuscation.json")


if __name__ == "__main__":
    main()
