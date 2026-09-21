#!/usr/bin/env python3
"""Evaluate two finished checkpoints on 20 additional, predeclared seeds.

Run only after the notebook has finished training AND its final evaluation:
    .venv/bin/python scripts/validate_generalization.py \
        --run pacman_runs/NEW_RUN_ID \
        --reference-run pacman_runs/20260915_203342_502034

This uses the notebook's exact environment, network, action-selection, and
evaluation functions. It does not train, record GIFs, modify the notebook, or
replace the classroom's five-game comparison. Only heldout_validation.json and
heldout_validation.csv are written, under .execution/heldout_NEW_RUN_ID by default.
"""

from __future__ import annotations

import argparse
import ast
import csv
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
import math
from pathlib import Path
import platform
import random
import statistics
import subprocess
import threading
import time


ROOT = Path(__file__).resolve().parents[1]
SEEDS = tuple(range(10001, 10021))
CELL_INDICES = (10, 12, 15, 17, 21, 30)
FUNCTION_CELLS = {15: "make_env", 17: "DQN", 21: "choose_action", 30: "evaluate"}
PACKAGES = ("torch", "gymnasium", "ale-py", "opencv-python-headless", "numpy", "Pillow")
ENVIRONMENT_SETTINGS = {
    "environment": "ALE/MsPacman-v5",
    "frame_skip": 4,
    "sticky_action_probability": 0.25,
    "noop_max": 30,
    "grayscale_size": 84,
    "stack_size": 4,
    "terminal_on_life_loss": False,
    "max_decisions_per_game": 3000,
    "eval_exploration": 0.05,
    "eval_seeds": [101, 202, 303, 404, 505],
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256_file(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def read_json(path):
    return json.loads(path.read_text())


def cell_source(cell):
    value = cell["source"]
    return "".join(value) if isinstance(value, list) else value


def notebook_sources(path):
    notebook = read_json(path)
    sources = {}
    for index in CELL_INDICES:
        cell = notebook["cells"][index]
        require(cell["cell_type"] == "code", f"Expected code in notebook cell {index}.")
        source = cell_source(cell)
        statements = ast.parse(source).body
        if index in FUNCTION_CELLS:
            name = FUNCTION_CELLS[index]
            require(len(statements) == 1 and
                    isinstance(statements[0], (ast.FunctionDef, ast.ClassDef)) and
                    statements[0].name == name,
                    f"Unexpected content in notebook cell {index}; expected only {name}.")
        else:
            require(all(isinstance(statement, ast.Assign) for statement in statements),
                    f"Expected only constant assignments in notebook cell {index}.")
        sources[index] = source
    # Verify that the evaluation implementation has not changed from the
    # archived clipped experiment. Constant cells can include training changes.
    archived = read_json(ROOT / "experiments/clipped_2000/pacman_dqn.ipynb")
    for index in (12, *FUNCTION_CELLS):
        require(sources[index] == cell_source(archived["cells"][index]),
                f"Evaluation source differs from the clipped reference in cell {index}.")
    source_hash = hashlib.sha256(json.dumps(
        sources, sort_keys=True, separators=(",", ":")
    ).encode()).hexdigest()
    return sources, source_hash


def read_finished_run(path):
    path = path.resolve()
    for name in ("config.json", "training_summary.json", "comparison.json", "trained.pt"):
        require((path / name).is_file(),
                f"Missing {path / name}; wait until the notebook's final evaluation finishes.")
    config = read_json(path / "config.json")
    summary = read_json(path / "training_summary.json")
    require(summary["status"] in ("completed", "interrupted"),
            f"Refusing to evaluate failed training run {path.name}.")
    for key, expected in ENVIRONMENT_SETTINGS.items():
        require(config.get(key) == expected,
                f"Unexpected evaluation setting for {path.name}: {key}={config.get(key)!r}.")
    comparison = read_json(path / "comparison.json")
    require(comparison["after"]["seeds"] == ENVIRONMENT_SETTINGS["eval_seeds"],
            f"Classroom final evaluation is missing or incomplete for {path.name}.")
    require(len(comparison["after"]["scores"]) == 5,
            f"Expected all five classroom scores for {path.name}.")
    return path, config, summary


class PhaseProgress:
    """Print a heartbeat without changing the exact notebook evaluation loop."""

    def __init__(self, label):
        self.label = label
        self.stop = threading.Event()
        self.started = time.monotonic()
        self.thread = threading.Thread(target=self.report, daemon=True)

    def report(self):
        while not self.stop.wait(30):
            print(f"{self.label}: evaluating 20 seeds; "
                  f"{time.monotonic() - self.started:.0f}s elapsed", flush=True)

    def __enter__(self):
        print(f"{self.label}: starting evaluation of seeds {SEEDS[0]}–{SEEDS[-1]}",
              flush=True)
        self.thread.start()

    def __exit__(self, *args):
        self.stop.set()
        self.thread.join()


def model_digest(model):
    digest = hashlib.sha256()
    for name, tensor in model.state_dict().items():
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str(value.dtype).encode())
        digest.update(str(tuple(value.shape)).encode())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def summarize(result):
    require(result["seeds"] == list(SEEDS), "Evaluation returned unexpected seeds.")
    require(all(len(result[key]) == len(SEEDS)
                for key in ("scores", "steps", "time_limited")),
            "Evaluation did not return 20 complete games.")
    scores = [float(score) for score in result["scores"]]
    require(all(math.isfinite(score) for score in scores), "Non-finite evaluation score.")
    require(all(isinstance(steps, int) and 1 <= steps <= 3000 for steps in result["steps"]),
            "Evaluation step counts violate the fixed time limit.")
    require(all(isinstance(capped, bool) for capped in result["time_limited"]),
            "Unexpected time-limit flags.")
    require(math.isclose(float(result["mean"]), statistics.mean(scores), abs_tol=1e-9),
            "Reported evaluation mean does not match scores.")
    count = sum(score >= 3000 for score in scores)
    return {
        "scores": scores, "seeds": list(SEEDS), "mean": statistics.mean(scores),
        "median": statistics.median(scores), "min": min(scores), "max": max(scores),
        "steps": list(result["steps"]), "time_limited": list(result["time_limited"]),
        "count_at_least_3000": count, "fraction_at_least_3000": count / len(scores),
    }


def evaluate_checkpoint(label, run, namespace, torch):
    path, config, summary = run
    device_name = config["device"]
    if device_name == "mps":
        require(torch.backends.mps.is_available(),
                "Recorded MPS device is unavailable; no automatic fallback is permitted.")
    elif device_name.startswith("cuda"):
        require(torch.cuda.is_available(),
                "Recorded CUDA device is unavailable; no automatic fallback is permitted.")
    else:
        require(device_name == "cpu", f"Unsupported recorded device: {device_name!r}.")
    checkpoint_path = path / "trained.pt"
    checkpoint_hash = sha256_file(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    require(isinstance(checkpoint["n_actions"], int) and checkpoint["n_actions"] > 0,
            "Invalid checkpoint action count.")
    require(all(torch.isfinite(tensor).all().item()
                for tensor in checkpoint["model"].values()),
            f"Non-finite checkpoint weights: {path.name}.")
    namespace["N_ACTIONS"] = checkpoint["n_actions"]
    model = namespace["DQN"](checkpoint["n_actions"]).to(torch.device(device_name))
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()
    before_digest = model_digest(model)
    started = time.monotonic()
    # One exact evaluate() call per model keeps the notebook's environment-reset
    # semantics intact; the heartbeat is independent of environment/model state.
    with PhaseProgress(label), torch.inference_mode():
        result = namespace["evaluate"](model, list(SEEDS), gif_path=None, record_best=False)
    elapsed = time.monotonic() - started
    require(model_digest(model) == before_digest, "Evaluation changed network weights.")
    require(sha256_file(checkpoint_path) == checkpoint_hash, "Checkpoint file changed.")
    metrics = summarize(result)
    print(f"{label}: mean {metrics['mean']:.1f}; minimum {metrics['min']:.0f}; "
          f"{metrics['count_at_least_3000']}/20 games reached 3000; "
          f"{elapsed:.1f}s elapsed", flush=True)
    return {
        "run_id": path.name, "checkpoint_sha256": checkpoint_hash,
        "device": device_name, "training_status": summary["status"],
        "completed_training_episodes": summary["completed_episodes"],
        **metrics, "evaluation_elapsed_seconds": elapsed,
        "numerical_checks": {
            "checkpoint_weights_finite": True,
            "model_weights_unchanged": True,
            "checkpoint_file_unchanged": True,
            "scores_finite": True,
            "means_match_scores": True,
            "steps_within_time_limit": True,
        },
    }


def hardware_info():
    details = {
        "platform": platform.platform(), "machine": platform.machine(),
        "processor": platform.processor(), "python": platform.python_version(),
    }
    if platform.system() == "Darwin":
        for name, key in (("cpu_model", "machdep.cpu.brand_string"),
                          ("physical_memory_bytes", "hw.memsize")):
            result = subprocess.run(["sysctl", "-n", key], capture_output=True,
                                    text=True, check=False)
            if result.returncode == 0:
                value = result.stdout.strip()
                details[name] = int(value) if name == "physical_memory_bytes" else value
    return details


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--reference-run", type=Path,
                        default=ROOT / "pacman_runs/20260915_203342_502034")
    parser.add_argument("--output-dir", type=Path,
                        help="Default: .execution/heldout_RUN_ID; must not contain existing outputs")
    args = parser.parse_args(argv)
    scaled = read_finished_run(args.run)
    reference = read_finished_run(args.reference_run)
    require(scaled[0] != reference[0], "The two model runs must be different.")
    require(scaled[1].get("training_reward_scale") == 0.01 and
            scaled[1].get("training_reward_clipping") is None and
            scaled[1].get("training_death_penalty") == 0.0,
            "The new run must use unclipped points × 0.01 without a death penalty.")
    require(reference[1].get("training_reward_clipping") == [-1, 1],
            "The reference run must use the archived [-1, 1] reward clipping.")
    require(scaled[1]["device"] == reference[1]["device"],
            "Both recorded model devices must match for this comparison.")
    output_dir = (args.output_dir or ROOT / ".execution" / f"heldout_{scaled[0].name}").resolve()
    json_path = output_dir / "heldout_validation.json"
    csv_path = output_dir / "heldout_validation.csv"
    require(not json_path.exists() and not csv_path.exists(),
            "Held-out outputs already exist; refusing to overwrite an earlier evaluation.")
    require(not any(output_dir == run[0] or run[0] in output_dir.parents
                    for run in (scaled, reference)),
            "Held-out outputs must remain outside the original run folders.")
    sources, source_hash = notebook_sources(ROOT / "pacman_dqn.ipynb")

    # Import the model runtime only after argument/source checks. No installation,
    # device probing that learns, or training cell is executed.
    import ale_py
    import gymnasium as gym
    import numpy as np
    from PIL import Image as PILImage
    import torch
    from torch import nn

    gym.register_envs(ale_py)
    torch.set_num_threads(min(4, torch.get_num_threads()))
    namespace = {"gym": gym, "torch": torch, "nn": nn, "np": np,
                 "random": random, "PILImage": PILImage}
    for index, source in sources.items():
        exec(compile(source, f"pacman_dqn.ipynb:cell-{index}", "exec"), namespace)
    require(namespace["MAX_STEPS"] == 3000 and namespace["FRAME_SKIP"] == 4 and
            namespace["EVAL_EXPLORATION"] == 0.05 and
            namespace["EVAL_SEEDS"] == ENVIRONMENT_SETTINGS["eval_seeds"],
            "Notebook evaluation constants changed.")

    started = datetime.now(timezone.utc).isoformat()
    models = {}
    for label, run in (("clipped_reference", reference), ("scaled_reward", scaled)):
        models[label] = evaluate_checkpoint(label, run, namespace, torch)
    data = {
        "schema_version": 1, "started_at_utc": started,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "kept_separate_from_class": True,
        "settings": {
            "seeds": list(SEEDS), "exploration": 0.05, "max_decisions": 3000,
            "generalization_purpose": "Compare fixed final models on 20 predeclared additional seeds; "
                                      "do not replace or tune against the classroom five-game scores.",
            "eval_source_hash": source_hash, "notebook_cells": list(CELL_INDICES),
            "environment": "ALE/MsPacman-v5", "frame_skip": 4,
            "sticky_action_probability": 0.25, "terminal_on_life_loss": False,
            "score_units": "raw game points", "gifs_recorded": False,
            "learning_updates": 0, "score_threshold": 3000,
        },
        "hardware": hardware_info(), "packages": {name: version(name) for name in PACKAGES},
        "models": models,
        "numerical_checks": {"all_model_checks_passed": True,
                             "identical_seeds_and_evaluation_settings": True},
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    # Exclusive creation prevents accidental replacement of earlier evidence.
    with json_path.open("x") as handle:
        json.dump(data, handle, indent=2, allow_nan=False)
        handle.write("\n")
    with csv_path.open("x", newline="") as handle:
        fields = ["model", "run_id", "seed", "score", "steps", "time_limited", "at_least_3000"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for label, result in models.items():
            for seed, score, steps, capped in zip(
                    result["seeds"], result["scores"], result["steps"], result["time_limited"]):
                writer.writerow({"model": label, "run_id": result["run_id"], "seed": seed,
                                 "score": score, "steps": steps, "time_limited": capped,
                                 "at_least_3000": score >= 3000})
    print(f"Saved {json_path}\nSaved {csv_path}", flush=True)


if __name__ == "__main__":
    main()
