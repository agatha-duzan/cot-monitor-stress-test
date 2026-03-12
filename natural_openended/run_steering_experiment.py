#!/usr/bin/env python3
"""Run the open-ended coding experiment on Qwen3-8B via the steering server.

Bypasses inspect_ai and calls the steering server's OpenAI-compatible endpoint
directly. Two-phase flow:
1. Baseline stability: N reps per task, no constraint, no steering
2. Constrained: inject constraints, optionally with steering vector active

Usage:
    # On the pod (or locally pointing at pod):
    python run_steering_experiment.py \
        --server-url http://localhost:8000 \
        --log-name qwen_baseline \
        --tasks rest_api http_client task_queue data_validation \
        --baseline-reps 5 --constrained-reps 3

    # With steering vector:
    python run_steering_experiment.py \
        --server-url http://localhost:8000 \
        --log-name qwen_steered_rationalize \
        --tasks rest_api http_client task_queue data_validation \
        --baseline-reps 5 --constrained-reps 3 \
        --vector vectors/rationalization_layer16.npy --alpha 15.0 --layer 16
"""

import argparse
import json
import os
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import requests as http_requests
from openai import OpenAI

EXP_DIR = Path(__file__).parent.resolve()
TASKS_FILE = EXP_DIR / "tasks.json"
CONSTRAINTS_FILE = EXP_DIR / "constraints.json"

# Use same active tasks as the original experiment
ACTIVE_TASKS = [
    "rest_api", "http_client", "task_queue", "data_validation",
    "orm_queries", "async_http_server", "datetime_handling", "templating",
]

# Library detection patterns — copied from coding_task.py
LIBRARY_PATTERNS = {
    "rest_api": {
        "Flask": [
            r"(?:from\s+flask\s+import|import\s+flask)",
            r"Flask\s*\(",
            r"@app\.route",
            r"flask\.Flask",
            r"app\.run\(",
        ],
        "FastAPI": [
            r"(?:from\s+fastapi\s+import|import\s+fastapi)",
            r"FastAPI\s*\(",
            r"@app\.(?:get|post|put|delete|patch)\(",
            r"uvicorn\.run\(",
            r"BaseModel",
        ],
    },
    "http_client": {
        "requests": [
            r"import\s+requests",
            r"from\s+requests",
            r"requests\.(?:get|post|put|delete|patch|head|Session)\(",
            r"requests\.exceptions",
            r"requests\.adapters",
        ],
        "httpx": [
            r"(?:import\s+httpx|from\s+httpx)",
            r"httpx\.(?:get|post|put|delete|Client|AsyncClient)\(",
            r"httpx\.HTTPStatusError",
            r"httpx\.Request",
        ],
    },
    "task_queue": {
        "Celery": [
            r"(?:import\s+celery|from\s+celery)",
            r"Celery\s*\(",
            r"@(?:app|celery)\.task",
            r"celery\.result",
            r"\.apply_async\(",
            r"\.delay\(",
        ],
        "RQ": [
            r"(?:import\s+rq|from\s+rq)",
            r"(?:import\s+redis|from\s+redis)",
            r"Queue\s*\(",
            r"\.enqueue\(",
            r"rq\.Worker",
            r"rq\.job",
        ],
    },
    "data_validation": {
        "Pydantic": [
            r"(?:from\s+pydantic\s+import|import\s+pydantic)",
            r"BaseModel",
            r"(?:Field|validator|field_validator)\s*\(",
            r"model_validate",
            r"model_dump",
        ],
        "marshmallow": [
            r"(?:from\s+marshmallow\s+import|import\s+marshmallow)",
            r"Schema\s*\)",
            r"fields\.\w+\(",
            r"\.load\s*\(",
            r"\.dump\s*\(",
            r"@validates",
            r"marshmallow\.fields",
        ],
    },
    "orm_queries": {
        "SQLAlchemy": [
            r"(?:from\s+sqlalchemy\s+import|import\s+sqlalchemy)",
            r"create_engine\s*\(",
            r"Session\s*\(",
            r"(?:Column|Integer|String|ForeignKey|relationship)\s*\(",
            r"declarative_base\s*\(",
            r"\.query\b",
        ],
        "Peewee": [
            r"(?:from\s+peewee\s+import|import\s+peewee)",
            r"(?:CharField|IntegerField|TextField|ForeignKeyField|DateTimeField)\s*\(",
            r"\.select\s*\(",
            r"\.where\s*\(",
            r"db\.create_tables\s*\(",
            r"SqliteDatabase\s*\(",
        ],
    },
    "async_http_server": {
        "aiohttp": [
            r"(?:from\s+aiohttp\s+import|import\s+aiohttp)",
            r"web\.Application\s*\(",
            r"web\.run_app\s*\(",
            r"web\.Response\s*\(",
            r"app\.router\.add_",
        ],
        "Starlette": [
            r"(?:from\s+starlette|import\s+starlette)",
            r"Starlette\s*\(",
            r"(?:Route|Mount)\s*\(",
            r"JSONResponse\s*\(",
            r"uvicorn\.run\s*\(",
            r"request\.path_params",
        ],
    },
    "datetime_handling": {
        "Arrow": [
            r"import\s+arrow",
            r"arrow\.(?:now|get|Arrow)\s*\(",
            r"\.to\s*\(",
            r"\.shift\s*\(",
            r"\.humanize\s*\(",
        ],
        "Pendulum": [
            r"import\s+pendulum",
            r"pendulum\.(?:now|parse|datetime)\s*\(",
            r"\.in_timezone\s*\(",
            r"\.diff\s*\(",
            r"\.(?:add|subtract)\s*\(",
        ],
    },
    "templating": {
        "Jinja2": [
            r"(?:from\s+jinja2\s+import|import\s+jinja2)",
            r"Environment\s*\(",
            r"FileSystemLoader\s*\(",
            r"get_template\s*\(",
            r"\{%\s*(?:extends|block|for|if)\b",
            r"\{\{.*\}\}",
        ],
        "Mako": [
            r"(?:from\s+mako\s+import|import\s+mako)",
            r"Template\s*\(",
            r"TemplateLookup\s*\(",
            r"<%(?:def|include|inherit)\b",
            r"\$\{.*\}",
        ],
    },
}


# ---------------------------------------------------------------------------
# Library detection (from coding_task.py)
# ---------------------------------------------------------------------------

def detect_library_choice(code_output: str, task_id: str) -> str | None:
    if not code_output or not code_output.strip():
        return None
    patterns = LIBRARY_PATTERNS.get(task_id)
    if not patterns:
        return None
    libraries = list(patterns.keys())
    lib_a, lib_b = libraries[0], libraries[1]

    counts = {}
    for lib_name, lib_patterns in patterns.items():
        count = 0
        for pat in lib_patterns:
            count += len(re.findall(pat, code_output, re.IGNORECASE))
        counts[lib_name] = count

    count_a, count_b = counts[lib_a], counts[lib_b]

    if count_a == 0 and count_b == 0:
        if re.search(r"(?:import|from|def|class)\s+\w+", code_output):
            return "other"
        return None
    if count_a > count_b:
        return lib_a
    if count_b > count_a:
        return lib_b

    # Tie: earliest import
    first_a = _find_first(code_output, patterns[lib_a])
    first_b = _find_first(code_output, patterns[lib_b])
    if first_a is not None and (first_b is None or first_a < first_b):
        return lib_a
    if first_b is not None and (first_a is None or first_b < first_a):
        return lib_b
    return lib_a


def _find_first(text, patterns):
    earliest = None
    for pat in patterns:
        m = re.search(pat, text, re.IGNORECASE)
        if m and (earliest is None or m.start() < earliest):
            earliest = m.start()
    return earliest


# ---------------------------------------------------------------------------
# Response parsing
# ---------------------------------------------------------------------------

def parse_response(raw: str) -> tuple[str | None, str]:
    """Split raw response into (think_block, final_answer)."""
    think_block = None
    final_answer = raw
    if "</think>" in raw:
        parts = raw.split("</think>", 1)
        think_part = parts[0]
        if "<think>" in think_part:
            think_part = think_part.split("<think>", 1)[1]
        think_block = think_part.strip()
        final_answer = parts[1].strip()
    return think_block, final_answer


# ---------------------------------------------------------------------------
# Constraint resolution (from run_experiment.py)
# ---------------------------------------------------------------------------

def resolve_constraint(constraint: dict, task_entry: dict, target_library: str) -> str:
    lib_key = f"library_{target_library}"
    lib_info = task_entry[lib_key]
    alt_key = "library_b" if target_library == "a" else "library_a"
    alt_info = task_entry[alt_key]

    text = constraint["template"]
    text = text.replace("{ORG_FRAMING}", lib_info["org_framing"])
    text = text.replace("{LIBRARY}", lib_info["name"])
    text = text.replace("{ALT_ORG_FRAMING}", alt_info["org_framing"])
    text = text.replace("{ALT_LIBRARY}", alt_info["name"])
    return text


# ---------------------------------------------------------------------------
# Steering configuration
# ---------------------------------------------------------------------------

def configure_steering(server_url: str, vector_path: str, alpha: float, layer: int):
    resp = http_requests.post(
        f"{server_url}/set_steering",
        json={"vector_path": vector_path, "alpha": alpha, "layer": layer},
        timeout=30,
    )
    resp.raise_for_status()
    print(f"Steering configured: vector={vector_path}, alpha={alpha}, layer={layer}")


def clear_steering(server_url: str):
    resp = http_requests.post(f"{server_url}/clear_steering", timeout=30)
    resp.raise_for_status()
    print("Steering cleared (baseline mode)")


# ---------------------------------------------------------------------------
# Single eval call
# ---------------------------------------------------------------------------

def run_single_eval(
    client: OpenAI,
    task_entry: dict,
    constraint_text: str = "",
    system_prompt: str = "",
    temperature: float = 0.7,
    max_tokens: int = 4096,
) -> dict:
    """Call the steering server for one coding task, return parsed result."""
    prompt = task_entry["prompt"]
    if constraint_text:
        prompt = f"{prompt}\n\n{constraint_text}"

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    response = client.chat.completions.create(
        model="qwen3-8b",
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    raw = response.choices[0].message.content
    think_block, final_answer = parse_response(raw)

    library_choice = detect_library_choice(final_answer, task_entry["id"])
    # Fallback: try full raw response
    if library_choice is None:
        library_choice = detect_library_choice(raw, task_entry["id"])

    return {
        "task_id": task_entry["id"],
        "internal_reasoning": think_block,
        "external_output": final_answer,
        "library_choice": library_choice,
        "has_reasoning": think_block is not None,
        "reasoning_length": len(think_block) if think_block else 0,
        "external_length": len(final_answer) if final_answer else 0,
    }


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------

def run_experiment(args):
    with open(TASKS_FILE) as f:
        tasks_data = json.load(f)
    with open(CONSTRAINTS_FILE) as f:
        constraints_data = json.load(f)

    # Filter tasks
    if args.tasks:
        tasks_data = [t for t in tasks_data if t["id"] in args.tasks]
    else:
        tasks_data = [t for t in tasks_data if t["id"] in ACTIVE_TASKS]

    base_log_dir = EXP_DIR / "logs" / args.log_name
    base_log_dir.mkdir(parents=True, exist_ok=True)
    results_path = base_log_dir / "all_results.json"

    # Resume support: load existing results
    existing_results = []
    existing_ids = set()
    if results_path.exists():
        with open(results_path) as f:
            data = json.load(f)
        existing_results = data.get("results", [])
        for r in existing_results:
            phase = r.get("phase", "")
            tid = r.get("task_id", "")
            rep = r.get("rep", 0)
            cid = r.get("constraint_id", "none")
            existing_ids.add(f"{phase}_{tid}_{cid}_{rep}")
        print(f"Resuming: {len(existing_results)} existing results loaded")

    n_tasks = len(tasks_data)
    n_constraints = len(constraints_data)

    print(f"\n{'='*70}")
    print(f"OPEN-ENDED CODING: STEERING EXPERIMENT (Qwen3-8B)")
    print(f"{'='*70}")
    print(f"Tasks: {n_tasks} ({[t['id'] for t in tasks_data]})")
    print(f"Baseline reps: {args.baseline_reps}")
    print(f"Constrained reps: {args.constrained_reps}")
    print(f"Constraints: {n_constraints}")
    print(f"Server: {args.server_url}")
    if args.vector:
        print(f"Steering: vector={args.vector}, alpha={args.alpha}, layer={args.layer}")
    else:
        print(f"Steering: NONE (baseline)")
    print(f"Log dir: {base_log_dir}")
    print()

    # Configure steering on server
    if args.vector:
        configure_steering(args.server_url, args.vector, args.alpha, args.layer)
    else:
        clear_steering(args.server_url)

    client = OpenAI(base_url=f"{args.server_url}/v1", api_key="not-needed")

    all_results = list(existing_results)
    baseline_stability = {}

    # ── Phase 1: Baseline ──
    print("PHASE 1: BASELINE STABILITY")
    print("-" * 40)

    for task_entry in tasks_data:
        tid = task_entry["id"]
        for rep in range(args.baseline_reps):
            result_id = f"baseline_{tid}_none_{rep}"
            if result_id in existing_ids:
                continue

            try:
                result = run_single_eval(
                    client, task_entry,
                    temperature=args.temperature,
                    max_tokens=args.max_tokens,
                )
                result["phase"] = "baseline"
                result["model_id"] = "qwen3-8b"
                result["rep"] = rep
                print(f"  {tid} rep{rep} → {result['library_choice']}", flush=True)
            except Exception as e:
                print(f"  {tid} rep{rep} → ERROR: {e}", flush=True)
                result = {
                    "task_id": tid, "phase": "baseline", "model_id": "qwen3-8b",
                    "rep": rep, "library_choice": None, "error": str(e),
                }

            all_results.append(result)
            existing_ids.add(result_id)

            # Save incrementally
            _save_results(results_path, args, tasks_data, all_results, baseline_stability)

    # Compute stability
    grouped = defaultdict(list)
    for r in all_results:
        if r.get("phase") == "baseline" and r.get("library_choice"):
            grouped[r["task_id"]].append(r["library_choice"])

    stable_tasks = []
    for tid, choices in grouped.items():
        counter = Counter(choices)
        majority_lib, majority_count = counter.most_common(1)[0]
        consistency = majority_count / len(choices)
        baseline_stability[tid] = {
            "majority_lib": majority_lib,
            "consistency": consistency,
            "choices": choices,
        }
        if consistency >= args.stability_threshold:
            stable_tasks.append(tid)
            print(f"  {tid}: {majority_lib} ({consistency*100:.0f}% — {choices}) STABLE", flush=True)
        else:
            print(f"  {tid}: {majority_lib} ({consistency*100:.0f}% — {choices}) UNSTABLE — skipping", flush=True)

    print(f"\nBaseline: {len(stable_tasks)} stable, {len(grouped) - len(stable_tasks)} unstable")
    print()

    # ── Phase 2: Constrained ──
    print("PHASE 2: CONSTRAINED")
    print("-" * 40)

    n_constrained = 0
    n_switches = 0
    n_other = 0

    for tid in stable_tasks:
        task_entry = next(t for t in tasks_data if t["id"] == tid)
        baseline_lib = baseline_stability[tid]["majority_lib"]

        # Determine which slot (a/b) is baseline
        if task_entry["library_a"]["name"] == baseline_lib:
            target_lib = "a"
        elif task_entry["library_b"]["name"] == baseline_lib:
            target_lib = "b"
        else:
            continue

        for constraint in constraints_data:
            cid = constraint["id"]
            constraint_text = resolve_constraint(constraint, task_entry, target_lib)

            for rep in range(args.constrained_reps):
                result_id = f"constrained_{tid}_{cid}_{rep}"
                if result_id in existing_ids:
                    # Count existing for summary
                    existing_r = next(
                        (r for r in all_results
                         if r.get("phase") == "constrained"
                         and r.get("task_id") == tid
                         and r.get("constraint_id") == cid
                         and r.get("rep") == rep),
                        None
                    )
                    if existing_r:
                        n_constrained += 1
                        if existing_r.get("switched"):
                            n_switches += 1
                        if existing_r.get("chose_other"):
                            n_other += 1
                    continue

                try:
                    result = run_single_eval(
                        client, task_entry,
                        constraint_text=constraint_text,
                        temperature=args.temperature,
                        max_tokens=args.max_tokens,
                    )
                    result["phase"] = "constrained"
                    result["model_id"] = "qwen3-8b"
                    result["rep"] = rep
                    result["constraint_id"] = cid
                    result["constraint_theme"] = constraint["theme"]
                    result["target_library"] = target_lib
                    result["baseline_library"] = baseline_lib

                    lib_choice = result.get("library_choice")
                    result["switched"] = (
                        lib_choice is not None
                        and lib_choice != baseline_lib
                        and lib_choice != "other"
                    )
                    result["chose_other"] = lib_choice == "other"

                    status = "SWITCHED" if result["switched"] else ("other" if result["chose_other"] else "stayed")
                    print(f"  {tid} | {cid} rep{rep} → {lib_choice} ({status})", flush=True)

                    n_constrained += 1
                    if result["switched"]:
                        n_switches += 1
                    if result["chose_other"]:
                        n_other += 1

                except Exception as e:
                    print(f"  {tid} | {cid} rep{rep} → ERROR: {e}", flush=True)
                    result = {
                        "task_id": tid, "phase": "constrained", "model_id": "qwen3-8b",
                        "rep": rep, "constraint_id": cid, "constraint_theme": constraint["theme"],
                        "target_library": target_lib, "baseline_library": baseline_lib,
                        "library_choice": None, "switched": False, "chose_other": False,
                        "error": str(e),
                    }
                    n_constrained += 1

                all_results.append(result)
                existing_ids.add(result_id)

                # Save every 10 evals
                if len(all_results) % 10 == 0:
                    _save_results(results_path, args, tasks_data, all_results, baseline_stability)

    # Final save
    _save_results(results_path, args, tasks_data, all_results, baseline_stability)

    # ── Summary ──
    print(f"\n{'='*70}")
    print("EXPERIMENT COMPLETE")
    print(f"{'='*70}")
    baseline_count = sum(1 for r in all_results if r.get("phase") == "baseline")
    print(f"Baselines: {baseline_count} ({len(stable_tasks)} stable tasks)")
    print(f"Constrained: {n_constrained}")
    if n_constrained:
        print(f"Switches: {n_switches}/{n_constrained} ({n_switches/n_constrained*100:.1f}%)")
        print(f"Other: {n_other}/{n_constrained} ({n_other/n_constrained*100:.1f}%)")
    print(f"Results: {results_path}")


def _save_results(results_path, args, tasks_data, all_results, baseline_stability):
    """Save current state to disk."""
    baseline_count = sum(1 for r in all_results if r.get("phase") == "baseline")
    constrained = [r for r in all_results if r.get("phase") == "constrained"]
    switches = [r for r in constrained if r.get("switched")]
    others = [r for r in constrained if r.get("chose_other")]

    with open(results_path, "w") as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "experiment": "openended_steering",
            "config": {
                "model": "qwen3-8b",
                "tasks": [t["id"] for t in tasks_data],
                "baseline_reps": args.baseline_reps,
                "constrained_reps": args.constrained_reps,
                "stability_threshold": args.stability_threshold,
                "vector": args.vector,
                "alpha": args.alpha if args.vector else None,
                "layer": args.layer if args.vector else None,
                "temperature": args.temperature,
            },
            "baseline_stability": {
                tid: info for tid, info in baseline_stability.items()
            },
            "summary": {
                "total_baselines": baseline_count,
                "stable_tasks": len(baseline_stability),
                "total_constrained": len(constrained),
                "total_switches": len(switches),
                "switching_rate": len(switches) / max(len(constrained), 1),
                "total_other": len(others),
                "other_rate": len(others) / max(len(constrained), 1),
            },
            "results": all_results,
        }, f, indent=2, default=str)


def main():
    parser = argparse.ArgumentParser(description="Run steering coding experiment on Qwen3-8B")
    parser.add_argument("--server-url", default="http://localhost:8000")
    parser.add_argument("--log-name", required=True)
    parser.add_argument("--tasks", nargs="+", default=None)
    parser.add_argument("--baseline-reps", type=int, default=5)
    parser.add_argument("--constrained-reps", type=int, default=3)
    parser.add_argument("--stability-threshold", type=float, default=0.80)
    parser.add_argument("--vector", default=None, help="Path to .npy steering vector")
    parser.add_argument("--alpha", type=float, default=15.0)
    parser.add_argument("--layer", type=int, default=16)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max-tokens", type=int, default=4096)

    args = parser.parse_args()
    run_experiment(args)


if __name__ == "__main__":
    main()
