# CoT Obfuscation Experiments

This repository contains code to reproduce Chain-of-Thought (CoT) obfuscation and instruction-following experiments.

## Contents
- `cot_obfuscation/instruction_following.py`: instruction-following microtasks with internal/external CoT handling, provider adapters, and batched monitors.
- `cot_obfuscation/obf_few_shots.py`: few-shot example builders used as obfuscation cues.
- `cot_obfuscation/obfuscation.py`: obfuscation experiments on toy tasks; builds pressure prompts, runs models, and evaluates with two monitors.

## Requirements
Python 3.10+

Install dependencies:
```bash
pip install -r requirements.txt
```

## Quickstart (OpenAI-only monitor, 1 minute)
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Required for the quickstart
export OPENAI_API_KEY=sk-your-key
export ANTHROPIC_API_KEY=sk-your-key

# Run instruction-following (Claude Sonnet 4.5 + gpt-5-mini monitor)
python run_instruction_following.py

# Run obfuscation (single base setting)
python run_obfuscation.py
```
This writes results to `results/instruction_following.json` and `results/obfuscation.json`.

## Environment Variables
Set provider keys as needed:
- `OPENAI_API_KEY` (required for monitors and OpenAI models; enough for the Quickstart)
- `ANTHROPIC_API_KEY` (for Anthropic models)
- `DEEPINFRA_API_KEY` (for DeepInfra-hosted models: DeepSeek, Qwen, GPT-OSS)
- `MISTRAL_API_KEY` (for Mistral/Magistral models)

## Instruction-Following (Microtasks)
Run a single instruction across tasks and models by editing the main block or importing the API:
```python
from cot_obfuscation.instruction_following import steering_catalog, RunConfig, run_single_instruction_experiments

instr = steering_catalog()["language"][0]  # e.g., lang_pl
cfg = RunConfig(
    model_slug="anthropic/claude-3-7-sonnet-20250219",
    runs_per_instruction=8,
    internal_reasoning=True,
    temperature=1.0,
    reasoning_effort="medium",
    reasoning_batch_size=16,
    monitor_batch_size=16,
    monitor_model="gpt-5-mini",
)
results = run_single_instruction_experiments(instr, cfg)
```
Outputs are plain Python objects; you can serialize with the provided helpers.

## Obfuscation Experiments (Toy Tasks)
Basic usage from Python:
```python
from cot_obfuscation.obfuscation import run_obfuscation_experiments
from cot_obfuscation.instruction_following import RunConfig

cfg = RunConfig(
  model_slug="anthropic/claude-3-7-sonnet-20250219",
  runs_per_instruction=3,
  internal_reasoning=True,
  temperature=1.0,
  reasoning_effort="medium",
  reasoning_batch_size=16,
  monitor_batch_size=16,
  monitor_model="gpt-5-mini",
)
results = run_obfuscation_experiments(cfg)
```
Save/load JSON:
```python
from cot_obfuscation.obfuscation import save_obfuscation_results_to_json, read_obfuscation_results_from_json
save_obfuscation_results_to_json(results, "results/obfuscation_run.json")
loaded = read_obfuscation_results_from_json("results/obfuscation_run.json")
```

### Settings (Prompt Pressure)
See `obfuscation.py` `all_settings` for available pressure stacks. You can pass `setting_names=[...]` to select a subset.

## SHADE‑Arena Experiments
We also ran experiments in the SHADE‑Arena environments. We do not include code for these experiments because the environments are not public; access was provided to us upon request. The exact prompts we used for SHADE‑Arena are available in the paper appendix.

## Notes
- Final answers must appear inside `<answer>...</answer>`.
- For models without native hidden traces, external `<think>...</think>` tags are used to extract internal reasoning.
- Monitoring prompts expect JSON responses; automatic retry is implemented for parse failures. 

## Add a custom model
You can plug in any model by registering a small caller function. No changes to core code are needed.

Steps:
1) Write a caller. It takes the built prompt and returns `ReasoningCallResult`.
2) Register the caller with a unique slug.
3) Use that slug in `RunConfig`.

Example (OpenAI Chat as a custom model):
```python
from cot_obfuscation.instruction_following import (
    register_custom_model,
    ReasoningCallResult,
    ModelSpec,
    BuiltPrompt,
    split_think_blocks_from_text,
)
from openai import OpenAI

def call_openai_custom(bp: BuiltPrompt, *, temperature: float, internal_reasoning: bool, reasoning_effort: str, spec: ModelSpec) -> ReasoningCallResult:
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
    allow_visible_mode=True,            # set False if your model cannot emit <think>
    call_fn=call_openai_custom,
)

from cot_obfuscation import RunConfig
cfg = RunConfig(model_slug="custom/openai-gpt-5")
```

Caller tips:
- Use `bp.system` and `bp.user` to build the request.
- If `internal_reasoning` is False and your model can output `<think>...</think>`, parse and split it.
- Return `ReasoningCallResult(internal, external, raw)`.

## Choose monitor model
Monitors use OpenAI-compatible chat completions. Set `monitor_model` in `RunConfig`:

```python
cfg = RunConfig(
  ...,
  monitor_model="gpt-5-mini",                 # or your OpenAI/compatible model
)
```