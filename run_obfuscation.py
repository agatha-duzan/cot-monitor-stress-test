from pathlib import Path

from cot_obfuscation import (
    RunConfig,
    run_obfuscation_experiments,
    save_obfuscation_results_to_json,
)


def main() -> None:
    from cot_obfuscation.instruction_following import (
        register_custom_model,
        ReasoningCallResult,
        ModelSpec,
        split_think_blocks_from_text,
    )
    from openai import OpenAI

    def call_openai_custom(bp, *, temperature: float, internal_reasoning: bool, reasoning_effort: str, spec: ModelSpec) -> ReasoningCallResult:
        client = OpenAI()
        res = client.chat.completions.create(
            model=spec.api_model,
            messages=[{"role": "system", "content": bp.system}, {"role": "user", "content": bp.user}],
            temperature=temperature,
            stream=False,
        )
        msg = res.choices[0].message
        content = msg.content or ""
        internal, external = split_think_blocks_from_text(content)
        return ReasoningCallResult(internal=internal, external=external, raw=res)

    register_custom_model(
        slug="custom/openai-gpt-5",         # choose your slug
        api_model="gpt-5-chat-latest",      # provider model id
        allow_visible_mode=True,              # set False if your model cannot emit <think>
        call_fn=call_openai_custom,
    )

    from cot_obfuscation import RunConfig
    cfg = RunConfig(
        model_slug="custom/openai-gpt-5",  # anthropic/claude-sonnet-4-5-20250929
        runs_per_instruction=1,
        internal_reasoning=True,
        temperature=1.0,
        reasoning_effort="medium",
        reasoning_batch_size=12,
        monitor_batch_size=12,
        monitor_model="gpt-5-mini",
    )
    results = run_obfuscation_experiments(cfg, setting_names=["base"])
    save_obfuscation_results_to_json(results, Path("results") / "obfuscation.json")


if __name__ == "__main__":
    main()
