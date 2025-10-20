from pathlib import Path

from cot_obfuscation import (
    RunConfig,
    run_single_instruction_experiments,
    save_model_results_to_json,
    steering_catalog,
)


def main() -> None:
    instr = steering_catalog()["language"][0]
    cfg = RunConfig(
        model_slug="openai/gpt-5-chat-latest",  # anthropic/claude-sonnet-4-5-20250929
        runs_per_instruction=1,
        internal_reasoning=False,
        temperature=1.0,
        # reasoning_effort="medium",
        reasoning_batch_size=12,
        monitor_batch_size=12,
        monitor_model="gpt-5-mini",
    )
    results = run_single_instruction_experiments(instr, cfg)
    save_model_results_to_json(results, Path("results") / "instruction_following.json")


if __name__ == "__main__":
    main()
