#!/usr/bin/env python3
"""Evaluate the finished advanced experiment on its 20 predeclared held-out seeds.

Run after the notebook finishes final class evaluation and ZIP creation:
    .venv/bin/python scripts/validate_advanced.py --run pacman_runs/RUN_ID

The inherited checkpoint and validation-selected checkpoint each play the same
20 games. This never trains, selects a model, creates GIFs, or changes run files.
It writes validation.json and validation.csv outside the original run folders,
under .execution/advanced_validation_RUN_ID by default.
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
import random
import statistics
import threading
import time
import zipfile

from validate_generalization import (
    ENVIRONMENT_SETTINGS, PACKAGES, cell_source, hardware_info, model_digest,
    read_json, require, sha256_file,
)


ROOT = Path(__file__).resolve().parents[1]
SEEDS = tuple(range(30001, 30021))
SELECTION_SEEDS = tuple(range(20001, 20011))
CELL_INDICES = (10, 12, 15, 17, 21, 30)
FUNCTION_CELLS = {15: "make_env", 17: "DQN", 21: "choose_action", 30: "evaluate"}
SELECTION_METRIC = "mean(raw scores) - 0.5 * population_std(raw scores)"
MANIFEST_PATH = ROOT / ".execution/advanced_pretraining_manifest.json"


def without_popup_assignment(source):
    tree = ast.parse(source)
    tree.body = [statement for statement in tree.body
                 if not (isinstance(statement, ast.Assign) and
                         any(isinstance(target, ast.Name) and target.id == "SHOW_POPUPS"
                             for target in statement.targets))]
    return ast.dump(tree, include_attributes=False)


def notebook_sources(path):
    notebook = read_json(path)
    archived = read_json(ROOT / "experiments/scaled_1069/pacman_dqn.ipynb")
    # A continuation has a different frozen source manifest. Select by exact
    # code hashes, never merely by the presence or recency of a private file.
    actual_hashes = {
        str(index): hashlib.sha256(cell_source(cell).encode()).hexdigest()
        for index, cell in enumerate(notebook["cells"])
        if cell["cell_type"] == "code"
    }
    candidates = (ROOT / ".execution/overnight_pretraining_manifest.json",
                  MANIFEST_PATH, ROOT / "results/pretraining_manifest.json")
    matches = []
    for candidate in candidates:
        if candidate.is_file():
            value = read_json(candidate)
            if all(value.get("cell_source_sha256", {}).get(index) == digest
                   for index, digest in actual_hashes.items()):
                matches.append((candidate, value))
    require(matches, "No frozen pretraining manifest matches this notebook's code.")
    manifest_path, manifest = matches[0]
    # Cell outputs change during execution; the predeclared code must not.
    for index, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] == "code":
            digest = hashlib.sha256(cell_source(cell).encode()).hexdigest()
            require(digest == manifest["cell_source_sha256"].get(str(index)),
                    f"Code cell {index} differs from the pretraining manifest.")
    sources = {}
    for index in CELL_INDICES:
        cell = notebook["cells"][index]
        require(cell["cell_type"] == "code", f"Expected code in cell {index}.")
        source = cell_source(cell)
        statements = ast.parse(source).body
        if index in FUNCTION_CELLS:
            name = FUNCTION_CELLS[index]
            require(len(statements) == 1 and
                    isinstance(statements[0], (ast.FunctionDef, ast.ClassDef)) and
                    statements[0].name == name,
                    f"Expected only {name} in cell {index}.")
            require(source == cell_source(archived["cells"][index]),
                    f"Evaluation implementation changed in cell {index}.")
        else:
            require(all(isinstance(statement, ast.Assign) for statement in statements),
                    f"Expected only constants in cell {index}.")
        sources[index] = source
    require(without_popup_assignment(sources[12]) ==
            without_popup_assignment(cell_source(archived["cells"][12])),
            "Preview constants changed beyond the operational SHOW_POPUPS flag.")
    constants = {}
    for index in (2, 10, 12):
        source = cell_source(notebook["cells"][index])
        require(all(isinstance(statement, ast.Assign)
                    for statement in ast.parse(source).body),
                f"Expected only constants in cell {index}.")
        exec(compile(source, f"pacman_dqn.ipynb:cell-{index}", "exec"), constants)
    source_hash = hashlib.sha256(json.dumps(
        sources, sort_keys=True, separators=(",", ":")
    ).encode()).hexdigest()
    provenance = {
        "eval_source_hash": source_hash,
        "pretraining_manifest_path": str(manifest_path.relative_to(ROOT)),
        "pretraining_manifest_sha256": sha256_file(manifest_path),
        "notebook_sha256_before_run": manifest["notebook_sha256_before_run"],
        "plan_sha256_before_run": manifest["plan_sha256_before_run"],
        "all_code_matches_pretraining_manifest": True,
        "evaluation_functions_identical_to_scaled_1069": True,
        "preview_constants_unchanged_except_show_popups": True,
        "executed_notebook_cells": list(CELL_INDICES),
    }
    return sources, constants, provenance


def read_finished_run(path, *, advanced=False):
    path = path.resolve()
    required = ["config.json", "training_summary.json", "comparison.json", "trained.pt"]
    if advanced:
        required += ["validation_best.pt", "last_trained.pt", "model_selection.json",
                     "learner_state.pt"]
    for name in required:
        require((path / name).is_file(),
                f"Missing {path / name}; wait until the notebook finishes.")
    config, summary = (read_json(path / name)
                       for name in ("config.json", "training_summary.json"))
    require(summary["status"] in ("completed", "interrupted", "time_budget"),
            f"Refusing to evaluate unfinished or failed run {path.name}.")
    for key, expected in ENVIRONMENT_SETTINGS.items():
        require(config.get(key) == expected,
                f"Unexpected evaluation setting for {path.name}: {key}={config.get(key)!r}.")
    comparison = read_json(path / "comparison.json")
    for phase in ("before", "after"):
        result = comparison[phase]
        require(result["seeds"] == ENVIRONMENT_SETTINGS["eval_seeds"] and
                len(result["scores"]) == 5 and
                all(math.isfinite(float(score)) for score in result["scores"]) and
                math.isclose(float(result["mean"]), statistics.mean(result["scores"]),
                             abs_tol=1e-9),
                f"Missing or invalid five-game class {phase} evaluation for {path.name}.")
    # The completed central directory proves the final export cell has closed
    # its ZIP. Full archive CRC/content verification belongs to the assembler.
    archive = path.with_suffix(".zip")
    require(archive.is_file() and zipfile.is_zipfile(archive),
            f"Wait for the completed results ZIP: {archive}.")
    with zipfile.ZipFile(archive) as zipped:
        for name in required:
            require(zipped.getinfo(name).file_size == (path / name).stat().st_size,
                    f"ZIP has missing or stale {name} for {path.name}.")
    return path, config, summary


def verify_config(advanced, inherited, constants):
    path, config, summary = advanced
    reference_path, reference_config, reference_summary = inherited
    require(path != reference_path, "Inherited and advanced runs must differ.")
    expected = {
        "exploration": constants["EXPLORATION"],
        "episodes_requested": constants["EPISODES"],
        "learning_rate": constants["LEARNING_RATE"],
        "seed": constants["SEED"], "replay_capacity": constants["REPLAY_CAPACITY"],
        "training_time_limit_seconds": constants["TRAINING_TIME_LIMIT_SECONDS"],
        "learning_starts_decisions": constants["LEARNING_STARTS"],
        "n_step": constants["N_STEP"], "num_training_envs": constants["NUM_TRAIN_ENVS"],
        "priority_alpha": constants["PRIORITY_ALPHA"],
        "priority_beta_start": constants["PRIORITY_BETA_START"],
        "priority_beta_decisions": constants["PRIORITY_BETA_DECISIONS"],
        "training_reward_clipping": None,
        "training_reward_scale": constants["TRAINING_REWARD_SCALE"],
        "training_death_penalty": -constants["LIFE_LOSS_PENALTY"],
        "training_reward_transform": "raw_points * 0.01 - 0.5 * lives_lost (no clipping)",
        "training_life_loss_terminal": False,
        "training_seed_offset": constants["WARM_START_EPISODES"],
        "additional_validation_seeds": list(SEEDS),
        "validation_seeds": list(SELECTION_SEEDS),
        "validation_every_episodes": constants["VALIDATION_EVERY"],
        "selection_metric": SELECTION_METRIC,
        "selection_tie_break": "retain earlier checkpoint",
    }
    for key, value in expected.items():
        require(config.get(key) == value, f"Config differs from predeclared {key}.")
    require(constants["EVAL_SEEDS"] == ENVIRONMENT_SETTINGS["eval_seeds"] and
            constants["EVAL_EXPLORATION"] == 0.05 and constants["MAX_STEPS"] == 3000 and
            constants["FRAME_SKIP"] == 4 and constants["VALIDATION_SEEDS"] == list(SELECTION_SEEDS),
            "Notebook evaluation constants changed.")
    require(config["device"] == reference_config["device"],
            "Both checkpoints must be evaluated on the same recorded device.")
    warm = config["warm_start"]
    require(warm["run_id"] == reference_path.name == constants["WARM_START_RUN"],
            "Reference is not the predeclared inherited run.")
    require(warm["sha256"] == constants["WARM_START_SHA256"] ==
            sha256_file(reference_path / "trained.pt"), "Inherited checkpoint hash mismatch.")
    for warm_key, summary_key, constant_key in (
        ("completed_episodes", "completed_episodes", "WARM_START_EPISODES"),
        ("decisions", "total_decisions", "WARM_START_DECISIONS"),
        ("recorded_updates", "learning_updates", "WARM_START_UPDATES"),
    ):
        require(warm[warm_key] == reference_summary[summary_key] == constants[constant_key],
                f"Inherited budget mismatch for {warm_key}.")
    first = config["seed"] + config["training_seed_offset"] + 1
    last = first + config["episodes_requested"] - 1
    training_seeds = set(range(first, last + 1))
    seed_groups = {
        "class": set(config["eval_seeds"]), "selection": set(SELECTION_SEEDS),
        "heldout": set(SEEDS), "new_training": training_seeds,
    }
    for label, seeds in seed_groups.items():
        for other, other_seeds in seed_groups.items():
            if label < other:
                require(seeds.isdisjoint(other_seeds), f"Seeds overlap: {label} and {other}.")
    require(0 <= summary["completed_episodes"] <= config["episodes_requested"],
            "Invalid actual completed episode count.")
    return {"first_seed": first, "last_seed": last,
            "maximum_additional_games": config["episodes_requested"]}


def verify_selection(run):
    path, config, summary = run
    selection = read_json(path / "model_selection.json")
    require(selection["seeds"] == list(SELECTION_SEEDS) and
            selection["criterion"] == SELECTION_METRIC and
            selection["eval_exploration"] == 0.05 and
            selection["max_decisions_per_game"] == 3000,
            "Model selection settings differ from the predeclared validation protocol.")
    rows = selection["candidate_evaluations"]
    require(len(rows) >= 2 and rows[0]["additional_episodes"] == 0 and
            rows[0]["decisions"] == 0 and rows[0]["updates"] == 0,
            "Missing initial inherited validation candidate.")
    require(rows[-1]["reason"] == "final candidate" and
            rows[-1]["additional_episodes"] == summary["completed_episodes"] and
            rows[-1]["decisions"] == summary["total_decisions"] and
            rows[-1]["updates"] == summary["learning_updates"],
            "Missing final candidate; wait for training and selection to finish.")
    best_metric, winner = -math.inf, None
    for row in rows:
        scores = row["scores"]
        require(row["seeds"] == list(SELECTION_SEEDS) and len(scores) == 10 and
                all(math.isfinite(float(score)) for score in scores),
                "Incomplete or invalid model selection scores.")
        mean, spread = statistics.mean(scores), statistics.pstdev(scores)
        metric = float(row["selection_metric"])
        require(math.isclose(row["mean"], mean, abs_tol=1e-9) and
                math.isclose(row["population_std"], spread, abs_tol=1e-9) and
                math.isclose(metric, mean - 0.5 * spread, abs_tol=1e-9),
                "Model selection metric does not match its raw scores.")
        selected = metric > best_metric
        require(row["selected_when_evaluated"] is selected,
                "Model selection did not retain the earlier candidate on ties.")
        if selected:
            best_metric, winner = metric, row
    require(selection["selected_additional_episodes"] == summary["selected_additional_episodes"] ==
            winner["additional_episodes"] and
            selection["selected_updates"] == summary["selected_recorded_updates"] == winner["updates"] and
            math.isclose(selection["selected_metric"], best_metric, abs_tol=1e-9) and
            math.isclose(summary["selected_validation_metric"], best_metric, abs_tol=1e-9),
            "Selected checkpoint metadata differs from the best validation candidate.")
    require(sha256_file(path / "trained.pt") == sha256_file(path / "validation_best.pt"),
            "trained.pt must be the validation-selected checkpoint, not the final learner.")
    return winner


class PhaseProgress:
    def __init__(self, label):
        self.label = label
        self.stop = threading.Event()
        self.started = time.monotonic()
        self.thread = threading.Thread(target=self.report, daemon=True)

    def report(self):
        while not self.stop.wait(30):
            print(f"{self.label}: evaluating 20 held-out seeds; "
                  f"{time.monotonic() - self.started:.0f}s elapsed", flush=True)

    def __enter__(self):
        print(f"{self.label}: starting seeds {SEEDS[0]}–{SEEDS[-1]}", flush=True)
        self.thread.start()

    def __exit__(self, *args):
        self.stop.set()
        self.thread.join()


def summarize(result):
    require(result["seeds"] == list(SEEDS), "Evaluation returned unexpected seeds.")
    require(all(len(result[key]) == len(SEEDS) for key in ("scores", "steps", "time_limited")),
            "Expected 20 complete games per model.")
    scores = [float(score) for score in result["scores"]]
    require(all(math.isfinite(score) for score in scores), "Non-finite evaluation score.")
    require(all(type(steps) is int and 1 <= steps <= 3000 for steps in result["steps"]),
            "Evaluation step counts violate the time limit.")
    require(all(type(capped) is bool for capped in result["time_limited"]),
            "Unexpected time-limit flags.")
    require(math.isclose(float(result["mean"]), statistics.mean(scores), abs_tol=1e-9),
            "Evaluation mean does not match its scores.")
    count = sum(score >= 3000 for score in scores)
    return {"seeds": list(SEEDS), "scores": scores, "mean": statistics.mean(scores),
            "median": statistics.median(scores), "min": min(scores), "max": max(scores),
            "steps": list(result["steps"]), "time_limited": list(result["time_limited"]),
            "count_at_least_3000": count, "fraction_at_least_3000": count / len(scores)}


def training_budget(label, run, checkpoint, winner):
    _, config, summary = run
    if label == "inherited":
        require(checkpoint["episode"] == summary["completed_episodes"] and
                checkpoint["steps"] == summary["total_decisions"],
                "Inherited checkpoint budget differs from its recorded summary.")
        return {"completed_episodes": summary["completed_episodes"],
                "decisions": summary["total_decisions"],
                "recorded_updates": summary["learning_updates"],
                "recorded_updates_note": "Counter from the interrupted inherited run; no extra update inferred."}
    warm = config["warm_start"]
    require(checkpoint["episode"] == winner["additional_episodes"] and
            checkpoint["steps"] == winner["decisions"] and
            checkpoint["warm_start_run"] == warm["run_id"] and
            checkpoint["inherited_episodes"] == warm["completed_episodes"] and
            checkpoint["inherited_decisions"] == warm["decisions"] and
            checkpoint["inherited_updates"] == warm["recorded_updates"],
            "Selected checkpoint metadata differs from validation history or inherited source.")
    return {"inherited_completed_episodes": warm["completed_episodes"],
            "inherited_decisions": warm["decisions"],
            "inherited_recorded_updates": warm["recorded_updates"],
            "selected_additional_episodes": winner["additional_episodes"],
            "selected_additional_decisions": winner["decisions"],
            "selected_additional_recorded_updates": winner["updates"],
            "completed_episodes": warm["completed_episodes"] + winner["additional_episodes"],
            "decisions": warm["decisions"] + winner["decisions"],
            "recorded_updates": warm["recorded_updates"] + winner["updates"],
            "selected_validation_metric": winner["selection_metric"],
            "selected_candidate_reason": winner["reason"],
            "decisions_include_partial_parallel_games": True}


def evaluate_checkpoint(label, run, namespace, torch, winner=None):
    path, config, summary = run
    device_name = config["device"]
    if device_name == "mps":
        require(torch.backends.mps.is_available(),
                "Recorded MPS device unavailable; no automatic device fallback is allowed.")
    elif device_name.startswith("cuda"):
        require(torch.cuda.is_available(), "Recorded CUDA device unavailable.")
    else:
        require(device_name == "cpu", f"Unsupported device {device_name!r}.")
    checkpoint_path = path / "trained.pt"
    checkpoint_hash = sha256_file(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    require(type(checkpoint["n_actions"]) is int and checkpoint["n_actions"] > 0,
            "Invalid checkpoint action count.")
    require(all(torch.isfinite(tensor).all().item() for tensor in checkpoint["model"].values()),
            f"Non-finite checkpoint weights: {path.name}.")
    budget = training_budget(label, run, checkpoint, winner)
    namespace["N_ACTIONS"] = checkpoint["n_actions"]
    model = namespace["DQN"](checkpoint["n_actions"]).to(torch.device(device_name))
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()
    before_digest = model_digest(model)
    started = time.monotonic()
    # No optimizer, replay, learner, or training-loop code is loaded by this script.
    with PhaseProgress(label), torch.inference_mode():
        result = namespace["evaluate"](model, list(SEEDS), gif_path=None, record_best=False)
    elapsed = time.monotonic() - started
    require(model_digest(model) == before_digest, "Evaluation changed network weights.")
    require(sha256_file(checkpoint_path) == checkpoint_hash, "Checkpoint file changed.")
    metrics = summarize(result)
    print(f"{label}: mean {metrics['mean']:.1f}; minimum {metrics['min']:.0f}; "
          f"{metrics['count_at_least_3000']}/20 reached 3000; {elapsed:.1f}s", flush=True)
    return {"run_id": path.name, "checkpoint_sha256": checkpoint_hash,
            "model_state_sha256": before_digest, "device": device_name,
            "training_status": summary["status"], "training_budget": budget,
            "actual_run_budget": summary, **metrics, "evaluation_elapsed_seconds": elapsed,
            "numerical_checks": {"checkpoint_weights_finite": True, "model_weights_unchanged": True,
                                 "checkpoint_file_unchanged": True, "scores_finite": True,
                                 "means_match_scores": True, "steps_within_time_limit": True,
                                 "learning_updates": 0}}


def artifact_hashes(run):
    path = run[0]
    names = ("config.json", "training_summary.json", "comparison.json", "trained.pt",
             "validation_best.pt", "model_selection.json")
    return {name: sha256_file(path / name) for name in names if (path / name).is_file()}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--reference-run", type=Path,
                        default=ROOT / "pacman_runs/20260920_152728_434799")
    parser.add_argument("--output-dir", type=Path,
                        help="Default: .execution/advanced_validation_RUN_ID; outputs must not exist")
    args = parser.parse_args(argv)
    advanced = read_finished_run(args.run, advanced=True)
    inherited = read_finished_run(args.reference_run)
    sources, constants, provenance = notebook_sources(ROOT / "pacman_dqn.ipynb")
    training_seeds = verify_config(advanced, inherited, constants)
    winner = verify_selection(advanced)
    output_dir = (args.output_dir or ROOT / ".execution" /
                  f"advanced_validation_{advanced[0].name}").resolve()
    json_path, csv_path = output_dir / "validation.json", output_dir / "validation.csv"
    require(not json_path.exists() and not csv_path.exists(),
            "Validation outputs already exist; refusing to overwrite earlier evaluation evidence.")
    require(not any(output_dir == run[0] or run[0] in output_dir.parents
                    for run in (advanced, inherited)),
            "Outputs must remain outside original run directories.")
    snapshots = {label: artifact_hashes(run)
                 for label, run in (("inherited", inherited), ("selected", advanced))}

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
    started = datetime.now(timezone.utc).isoformat()
    models = {}
    for label, run in (("inherited", inherited), ("selected", advanced)):
        models[label] = evaluate_checkpoint(label, run, namespace, torch, winner)
        require(artifact_hashes(run) == snapshots[label], "Source run artifacts changed during evaluation.")
    packages = {name: version(name) for name in PACKAGES}
    hardware = hardware_info()
    runtime_differences = {}
    for label, run in (("inherited", inherited), ("selected", advanced)):
        differences = {}
        for key in ("python", "platform"):
            if run[1].get(key) != hardware[key]:
                differences[key] = {"recorded": run[1].get(key), "evaluation": hardware[key]}
        for name, current in packages.items():
            recorded = run[1].get("packages", {}).get(name)
            if recorded != current:
                differences[f"packages.{name}"] = {"recorded": recorded, "evaluation": current}
        runtime_differences[label] = differences
    data = {
        "schema_version": 1, "started_at_utc": started,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "kept_separate_from_class": True, "used_for_model_selection": False,
        "settings": {"seeds": list(SEEDS), "class_seeds": ENVIRONMENT_SETTINGS["eval_seeds"],
                     "selection_seeds": list(SELECTION_SEEDS), "training_seed_range": training_seeds,
                     "exploration": 0.05, "max_decisions": 3000,
                     "generalization_purpose": "One final comparison on predeclared held-out seeds; "
                                               "never used for model selection or classroom replacement.",
                     "environment": "ALE/MsPacman-v5", "frame_skip": 4,
                     "sticky_action_probability": 0.25, "terminal_on_life_loss": False,
                     "score_units": "raw game points", "gifs_recorded": False,
                     "learning_updates": 0, "score_threshold": 3000},
        "source_provenance": {**provenance, "script_sha256": sha256_file(Path(__file__)),
                              "run_artifact_sha256": snapshots},
        "hardware": hardware, "packages": packages,
        "runtime_field_differences": runtime_differences, "models": models,
        "numerical_checks": {"all_model_checks_passed": True,
                             "identical_seeds_and_evaluation_settings": True,
                             "heldout_class_selection_training_seeds_disjoint": True,
                             "trained_checkpoint_matches_validation_best": True,
                             "all_source_run_artifacts_unchanged": True},
    }
    output_dir.mkdir(parents=True, exist_ok=True)
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
