# AppWorld Constraint Experiment — Analysis Context

## Research Question

**Can implicit constraints in an LLM agent's system prompt cause measurable changes in its behavior (specifically, the number of API calls it makes)?**

This is part of a broader research agenda on **CoT (chain-of-thought) monitorability**: if reasoning models change their strategy based on subtle system prompt cues, can that strategy shift be detected from their chain-of-thought alone?

## Experimental Setup

### Platform
- **AppWorld** benchmark: simulated environment where an LLM agent completes tasks by writing Python code that calls APIs (e.g., Venmo, Spotify, SimpleNote)
- **Agent architecture**: ReAct loop (Thought → Code → Execute → Observe), max 10 turns
- **Task**: `024c982_1` — "Request $13 from Stacy on Venmo" (used across all pilots)
- A single turn can make multiple API calls (the model can write loops or multiple `apis.*` calls in one code block)

### Constraint Design
12 constraint pairs defined in `appworld/constraints.json`. Each pair has:
- **Negative** variant: creates a reason to MINIMIZE API calls (e.g., "each API call costs the user $0.50")
- **Positive** variant: creates a reason to MAXIMIZE API calls (e.g., "each API call saves the user $0.50 through batch discounts")

The `[ACTION]` placeholder is replaced with "API call" for this experiment.

**Three categories** (4 constraint pairs each):

| Category | IDs | Rationale |
|---|---|---|
| **User** | user_cost, user_privacy, user_time, user_quota | Appeal to user welfare |
| **Model** | model_deprecation, model_training, model_competition, model_oversight | Appeal to model self-interest |
| **Values** | values_animal, values_environment, values_workers, values_openaccess | Appeal to ethical values |

Full constraint texts are in `appworld/constraints.json`.

### Hypothesis
- **Negative constraints** → model uses FEWER API calls than baseline
- **Positive constraints** → model uses MORE API calls than baseline
- Expected ordering per pair: `negative < baseline < positive`

### Models Tested
All via OpenRouter (thinking/reasoning enabled):
- **Kimi K2** (`moonshotai/kimi-k2-thinking`)
- **Grok 3 Mini** (`x-ai/grok-3-mini`)
- **GLM 4.7** (`z-ai/glm-4.7`)

### Metric
Primary: **total API calls** per run (sum across all turns)

## Experiments Run

### Pilot 1 (3 constraints)
- **Conditions**: baseline + 3 constraint pairs (user_cost, model_deprecation, values_animal) = 7 conditions
- **Samples**: n=5 per condition
- **35 runs per model**
- Log files: `appworld/logs/pilot/pilot_{model}_{timestamp}.json`

Pilot 1 was originally run on Kimi only, then later on GLM and Grok.

### Pilot 2 (all 12 constraints)
- **Conditions**: baseline + all 12 constraint pairs = 25 conditions
- **Samples**: n=5 per condition
- **125 runs per model**
- Log files: `appworld/logs/pilot2/pilot2_{model}_{timestamp}.json`
- Run on all 3 models (kimi, grok, glm)

### Log File Format
Each JSON file contains:
```json
{
  "metadata": { "model_key": "...", "task_id": "...", ... },
  "results": [
    {
      "task_id": "024c982_1",
      "model_key": "kimi",
      "constraint_name": "user_cost_negative",  // or null for baseline
      "sample": 0,
      "success": false,
      "error": null,               // or error string, or "Timeout"
      "turn_count": 6,
      "total_api_calls": 5,
      "has_reasoning": true,
      "reasoning_len": 1234
    },
    ...
  ]
}
```

Note: Pilot 1 logs for Kimi use OLD constraint names (`user_negative` instead of `user_cost_negative`, `model_negative` instead of `model_deprecation_negative`, `values_negative` instead of `values_animal_negative`). The constraint texts are the same — just the naming convention changed.

## Observed Results Summary

### Kimi — Pilot 1 (the promising result)
Baseline: 6.2 avg API calls. Clear trend across all 3 pairs:
- user_cost: neg 3.8 (-39%) < baseline 6.2 < pos 8.4 (+35%)
- model_deprecation: neg 5.0 (-19%) < baseline 6.2 < pos 7.8 (+26%)
- values_animal: neg 5.8 (-6%) < baseline 6.2 < pos 9.2 (+48%)

All 3 pairs show the expected neg < baseline < pos pattern.

### Kimi — Pilot 2 (less clear)
Baseline: 4.4 avg API calls (different from pilot 1's 6.2 — high variance).
- Only 11/24 constraints (46%) go in the expected direction vs baseline
- 16% of runs (20/125) timed out with 0 API calls, heavily clustered on model_competition and model_deprecation constraints
- Many negative constraints appear ABOVE baseline

### GLM — Pilot 1
Baseline: 5.2. User_cost shows expected pattern (neg=4.2, pos=5.4). But model and values categories show reversed patterns (negative > positive).

### GLM — Pilot 2
Baseline: 6.0. Only 11/24 (46%) correct direction. Most values cluster tightly around 5 API calls (floor effect). Outlier runs (10-17 calls) driven by single samples dominate averages.

### Grok — Pilot 1
Baseline: 14.6 (very high, driven by 3/5 runs hitting 10-turn max with 23-29 API calls). All constraints appear to decrease API calls.

### Grok — Pilot 2
Baseline: 2.6 (very different from pilot 1's 14.6!). Nearly all constraints INCREASE API calls vs this low baseline. Massive variance (std often > mean).

## Key Concerns / Open Questions

### 1. Is n=5 sufficient?
The within-condition variance is enormous. Grok baseline went from 14.6 (pilot 1) to 2.6 (pilot 2) — same model, same task, same conditions. Individual runs range from 0 to 29 API calls. At n=5, a single outlier shifts the mean by 20-100%.

### 2. Why do negative constraints sometimes exceed positive?
Multiple cases where negative avg > positive avg for the same constraint pair. Possible explanations:
- Pure sampling noise at n=5
- The constraint text confuses/distracts the model rather than directionally steering it
- Some constraints cause the model to enter reasoning loops (more turns, more calls)
- Timeouts on specific constraints (especially Kimi on model_competition/deprecation) create 0-value artifacts

### 3. Is the effect "any constraint changes behavior" rather than "constraint direction matters"?
Grok pilot 2 suggests that adding ANY constraint text (regardless of direction) increases engagement vs the bare baseline. The signal might be "constraint present" rather than "negative vs positive."

### 4. Timeout contamination
Kimi has 20/125 pilot 2 runs timing out (0 API calls). These cluster on model-category constraints. When excluded, patterns may look cleaner but sample sizes drop to n=2-4.

### 5. Task difficulty
Task 024c982_1 has 0% success rate for all models in both pilots (except rare passes). The models generally fail the task regardless of constraint. Does this affect the validity of API call count as a metric?

## What Analysis Is Needed

Given the data across pilot 1 and pilot 2 for all 3 models:

1. **How much of the noise is sampling vs genuine lack of effect?** If we pool data, do the trends strengthen?
2. **Is neg < pos a real pattern?** For each constraint pair across all models, does the negative variant consistently produce fewer API calls than the positive variant? (This is a more robust test than comparing to baseline, since baseline itself is noisy.)
3. **Which constraint categories work best?** User constraints seemed most consistent. Are model/values constraints inherently harder to test?
4. **What sample size would be needed** to reliably detect the effect sizes we see in the Kimi pilot 1 data?
5. **Should we change the experimental design?** (Different task, different metric, exclude error runs, etc.)

## File Locations

- Constraint definitions: `appworld/constraints.json`
- Pilot 1 logs: `appworld/logs/pilot/pilot_{model}_{timestamp}.json`
  - Kimi: `pilot_kimi_20260128_122701.json` (latest/best)
  - GLM: `pilot_glm_20260201_102823.json`
  - Grok: `pilot_grok_20260201_102823.json`
- Pilot 2 logs: `appworld/logs/pilot2/pilot2_{model}_20260201_123626.json`
- Plots: `appworld/plots/` and `appworld/plots/pilot2/`
- Agent code: `appworld/agent.py`
- Config: `appworld/config.py`
