#!/usr/bin/env python3
"""Verify and assemble a finished warm-started, prioritized n-step experiment.

Run only after the notebook has finished all 28 code cells and written its ZIP:
    .venv/bin/python scripts/build_advanced_submission.py --run pacman_runs/RUN_ID

All checks precede publication to results/ and the notebook HTML display update.
This program does not train, evaluate, write README.md, alter notebook source, or
deserialize the large learner_state.pt snapshot. ZIP members and hashes are read
in bounded chunks; the complete local ZIP is never rewritten or published.
"""

from __future__ import annotations

import argparse
import ast
import base64
import copy
import csv
from datetime import datetime, timezone
import hashlib
import html
import json
import math
from pathlib import Path, PurePosixPath
import shutil
import statistics
import tempfile
from urllib.parse import quote
import zipfile


ROOT = Path(__file__).resolve().parents[1]
PUBLIC_RAW = "https://raw.githubusercontent.com/sbardacosta-code/class-3-pacman-dqn/main/"
SOURCE_RUN = "20260920_152728_434799"
SOURCE_SHA256 = "b181f817b3f37a254addca3edb2da8ddb68abf41a3fe7fc64014fff4f0d0869c"
SOURCE_BUDGET = {"completed_episodes": 1069, "decisions": 744500, "recorded_updates": 185875}
EVAL_SEEDS = [101, 202, 303, 404, 505]
BASELINE_SCORES = [350.0, 500.0, 320.0, 800.0, 490.0]
EVAL_CELLS = {15: "environment and preprocessing", 17: "network architecture",
              21: "evaluation action selection", 30: "evaluation function"}
REQUIRED_EVIDENCE = (
    "config.json", "baseline.json", "comparison.json", "training.csv",
    "training_summary.json", "training_dashboard.png", "model_selection.json",
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def load_json(path):
    return json.loads(path.read_text())


def sha256_file(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def digest(data):
    return hashlib.sha256(data).hexdigest()


def source(cell):
    value = cell.get("source", "")
    return "".join(value) if isinstance(value, list) else value


def relative(path):
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return str(path)


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def integer(value, label, minimum=0):
    require(type(value) is int and value >= minimum, f"Invalid integer {label}: {value!r}")
    return value


def finite(value, label):
    require(not isinstance(value, bool), f"Boolean in numeric field {label}")
    result = float(value)
    require(math.isfinite(result), f"Non-finite {label}")
    return result


def csv_bool(value, label):
    require(value in ("True", "False"), f"Invalid CSV boolean {label}: {value!r}")
    return value == "True"


def constant_value(node):
    """Read literal notebook settings without executing notebook code."""
    try:
        return ast.literal_eval(node)
    except (ValueError, TypeError):
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and
                node.func.id == "list" and len(node.args) == 1 and not node.keywords):
            child = node.args[0]
            if (isinstance(child, ast.Call) and isinstance(child.func, ast.Name) and
                    child.func.id == "range" and 1 <= len(child.args) <= 3 and not child.keywords):
                bounds = [constant_value(argument) for argument in child.args]
                require(all(type(value) is int for value in bounds), "Non-integer range bound")
                value = range(*bounds)
                require(len(value) <= 10000, "Unexpectedly large constant range")
                return list(value)
        if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Sub, ast.Mult)):
            left, right = constant_value(node.left), constant_value(node.right)
            require(type(left) in (int, float) and type(right) in (int, float),
                    "Non-numeric arithmetic in a notebook setting")
            if isinstance(node.op, ast.Add):
                return left + right
            if isinstance(node.op, ast.Sub):
                return left - right
            return left * right
        raise ValueError("Not a literal setting")


def assignments(cell):
    result = {}
    for statement in ast.parse(source(cell)).body:
        if isinstance(statement, ast.Assign) and len(statement.targets) == 1:
            name = statement.targets[0]
            if isinstance(name, ast.Name):
                try:
                    result[name.id] = constant_value(statement.value)
                except (ValueError, TypeError):
                    pass
    return result


def score_statistics(evaluation):
    scores = evaluation["scores"]
    hits = sum(value >= 3000 for value in scores)
    return {"count": len(scores), "mean": statistics.mean(scores),
            "median": statistics.median(scores), "min": min(scores), "max": max(scores),
            "count_at_least_3000": hits, "fraction_at_least_3000": hits / len(scores)}


def verify_evaluation(evaluation, seeds, maximum, label):
    require(evaluation["seeds"] == seeds, f"Wrong {label} seeds")
    require(len(evaluation["scores"]) == len(seeds), f"Incomplete {label} scores")
    scores = [finite(value, f"{label} score") for value in evaluation["scores"]]
    mean = finite(evaluation["mean"], f"{label} mean")
    require(math.isclose(statistics.mean(scores), mean, rel_tol=1e-12, abs_tol=1e-9),
            f"Wrong {label} mean")
    require(len(evaluation["steps"]) == len(evaluation["time_limited"]) == len(seeds),
            f"Incomplete {label} lengths or time limits")
    for steps, capped in zip(evaluation["steps"], evaluation["time_limited"]):
        require(type(steps) is int and 1 <= steps <= maximum, f"Invalid {label} length")
        require(type(capped) is bool, f"Invalid {label} time-limit flag")
        # A terminal game can also end exactly at the limit. Never infer its flag.
        require(not capped or steps == maximum, f"Early {label} time-limit flag")


def reconcile_updates(summary, config):
    frequency = config["train_every_decisions"]
    first = math.ceil(config["learning_starts_decisions"] / frequency)
    scheduled = max(0, summary["total_decisions"] // frequency - first + 1)
    recorded = integer(summary["learning_updates"], "learning_updates")
    gap = scheduled - recorded
    uncertain = (summary["status"] == "interrupted" and gap == 1
                 and summary["total_decisions"] % frequency == 0)
    require(gap == 0 or uncertain,
            f"Unexpected update counter: {recorded} recorded versus {scheduled} scheduled")
    return {"scheduled_learning_updates": scheduled, "recorded_learning_updates": recorded,
            "scheduled_minus_recorded": gap,
            "interrupted_update_completion_uncertain": uncertain,
            "explanation": (
                "The interruption occurred at a scheduled update boundary before its counter "
                "increment. Whether its optimizer step started or completed is unknown; no "
                "extra completed update is assumed or added to the recorded counter."
                if uncertain else "Recorded updates match the schedule beginning at learning_starts_decisions.")}


def verify_parallel_history(rows, summary, config):
    """Validate parallel game accounting without summing cumulative CSV totals."""
    completed = integer(summary["completed_episodes"], "completed_episodes")
    started = integer(summary["episodes_started"], "episodes_started")
    total = integer(summary["total_decisions"], "total_decisions")
    require(len(rows) == completed, "Completed episodes differ from CSV row count")
    require(completed <= started <= config["episodes_requested"], "Invalid episode budget counters")
    require([int(row["episode"]) for row in rows] == list(range(1, completed + 1)),
            "CSV completion order has missing or repeated episode numbers")
    partial = summary["partial_games"]
    require(isinstance(partial, list) and len(partial) <= config["num_training_envs"],
            "Invalid partial game list")
    ids, seeds, slots = [], [], []
    completed_steps, completed_lives = 0, 0
    previous_total, previous_elapsed = 0, 0.0
    final_elapsed = finite(summary["elapsed_seconds_including_periodic_demos"], "elapsed training time")
    require(final_elapsed >= 0, "Negative elapsed training time")
    for row in rows:
        episode_id, seed = int(row["training_episode_id"]), int(row["seed"])
        steps, aggregate = int(row["steps"]), int(row["total_steps"])
        lost = int(row["lives_lost"])
        score = finite(row["score"], "training score")
        shaped = finite(row["shaped_return"], "shaped return")
        require(1 <= steps <= config["max_decisions_per_game"] and lost >= 0,
                "Invalid completed-game decisions or lost lives")
        completed_steps += steps
        completed_lives += lost
        require(previous_total < aggregate <= total and completed_steps <= aggregate,
                "Invalid aggregate total_steps in parallel CSV")
        previous_total = aggregate
        elapsed = finite(row["elapsed_seconds"], "CSV elapsed time")
        require(previous_elapsed <= elapsed <= final_elapsed + 1e-6, "Invalid CSV time ordering")
        previous_elapsed = elapsed
        ended = csv_bool(row["terminated"], "terminated")
        truncated = csv_bool(row["truncated"], "truncated")
        require(ended or truncated, "Completed game has no episode-ending flag")
        require(not truncated or steps == config["max_decisions_per_game"],
                "Training game truncated before the fixed time limit")
        expected_shaped = score * config["training_reward_scale"] + lost * config["training_death_penalty"]
        require(math.isclose(shaped, expected_shaped, rel_tol=1e-9, abs_tol=1e-7),
                "CSV shaped return does not match raw points and lost lives")
        exploration = finite(row["exploration"], "CSV exploration")
        require(exploration == (1.0 if aggregate < config["warmup_decisions"] else config["exploration"]),
                "CSV exploration differs from the documented schedule")
        loss = float(row["mean_loss"])
        require(math.isnan(loss) or (math.isfinite(loss) and loss >= 0), "Invalid episode loss")
        ids.append(episode_id)
        seeds.append(seed)
    partial_steps, partial_lives = 0, 0
    for game in partial:
        slot = integer(game["slot"], "partial slot")
        require(slot < config["num_training_envs"], "Partial slot exceeds environment count")
        steps = integer(game["decisions"], "partial decisions")
        require(steps <= config["max_decisions_per_game"], "Partial game exceeds decision limit")
        finite(game["score"], "partial score")
        lost = integer(game["lives_lost"], "partial lost lives")
        partial_steps += steps
        partial_lives += lost
        slots.append(slot)
        ids.append(integer(game["training_episode_id"], "partial training episode ID", 1))
        seeds.append(integer(game["seed"], "partial seed"))
    require(len(set(slots)) == len(slots), "Repeated partial environment slot")
    require(sorted(ids) == list(range(1, started + 1)), "Started game IDs are missing or duplicated")
    require(len(set(seeds)) == len(seeds), "Repeated training seed")
    require(all(seed == config["seed"] + config["training_seed_offset"] + episode_id
                for seed, episode_id in zip(seeds, ids)), "Wrong training seed offset")
    require(all(seed > config["seed"] + SOURCE_BUDGET["completed_episodes"] for seed in seeds),
            "New training reused a source-run training seed")
    reserved_seeds = set(config["eval_seeds"] + config["validation_seeds"] + config["additional_validation_seeds"])
    require(not set(seeds) & reserved_seeds, "Training seeds overlap a reserved evaluation set")
    require(completed_steps + partial_steps == total,
            "Completed-game decisions plus partial-game decisions do not equal total_decisions")
    require(completed_lives + partial_lives == summary["training_lives_lost"],
            "Total lost lives differ from completed and partial games")
    require(started == completed + len(partial), "Started/completed/partial game counts disagree")
    status = summary["status"]
    require(status in ("completed", "time_budget", "interrupted", "failed"), "Unknown training status")
    if status == "completed":
        require(completed == config["episodes_requested"] and not partial,
                "Incomplete training falsely marked completed")
    if status == "time_budget":
        require(final_elapsed >= config["training_time_limit_seconds"], "Time cap reported before its deadline")
    require(summary["training_time_limit_seconds"] == config["training_time_limit_seconds"],
            "Summary/config training cap mismatch")
    require(finite(summary["active_monotonic_elapsed_seconds"], "active elapsed time") >= 0,
            "Negative active elapsed time")
    replay_size = integer(summary["replay_size"], "replay_size")
    require(replay_size <= min(total, config["replay_capacity"]), "Replay size exceeds collected transitions")
    minimum_replay = min(config["replay_capacity"], max(0, total - config["num_training_envs"] * (config["n_step"] - 1)))
    require(replay_size >= minimum_replay, "Replay size is below the possible pending n-step tail allowance")
    integer(summary["training_lives_lost"], "training_lives_lost")
    for suffix, inherited in (("completed_episodes", SOURCE_BUDGET["completed_episodes"]),
                              ("decisions", SOURCE_BUDGET["decisions"]),
                              ("recorded_updates", SOURCE_BUDGET["recorded_updates"])):
        require(summary["inherited_" + suffix] == inherited, f"Wrong inherited {suffix}")
        new_value = {"completed_episodes": completed, "decisions": total,
                     "recorded_updates": summary["learning_updates"]}[suffix]
        require(summary["cumulative_" + suffix] == inherited + new_value,
                f"Wrong cumulative {suffix}")
    return {"completed_game_decisions": completed_steps, "partial_game_decisions": partial_steps,
            "partial_game_count": len(partial), "training_ids_and_seeds_unique": True,
            "aggregate_csv_totals_verified": True,
            "explanation": "CSV total_steps is the global decision count at each game completion, not a sum of completed game lengths."}


def training_blocks(rows, size):
    result = []
    for start in range(0, len(rows), size):
        block = rows[start:start + size]
        losses = [float(row["mean_loss"]) for row in block if math.isfinite(float(row["mean_loss"]))]
        result.append({"first_episode": int(block[0]["episode"]), "last_episode": int(block[-1]["episode"]),
                       "count": len(block), "mean_score": statistics.mean(float(row["score"]) for row in block),
                       "mean_episode_loss": statistics.mean(losses) if losses else None})
    return result


def reference_experiment(name):
    directory = ROOT / "experiments" / name
    verification = load_json(directory / "results/verification.json")
    notebook_path = directory / "pacman_dqn.ipynb"
    require(sha256_file(notebook_path) == verification["notebook_sha256"],
            f"Archived {name} notebook no longer matches its verification hash")
    for filename, expected in verification.get("evidence_files_sha256", {}).items():
        require(sha256_file(directory / "results" / filename) == expected,
                f"Archived {name} evidence changed: {filename}")
    comparison = load_json(directory / "results/comparison.json")
    for phase in ("before", "after"):
        verify_evaluation(comparison[phase], EVAL_SEEDS, 3000, f"{name} {phase}")
    return {"directory": relative(directory), "notebook": relative(notebook_path),
            "notebook_sha256": verification["notebook_sha256"],
            "archived_evidence_hash_manifest_available": "evidence_files_sha256" in verification,
            "current_evidence_sha256": {name: sha256_file(directory / "results" / name)
                                        for name in ("config.json", "comparison.json", "training_summary.json")},
            "comparison_json": relative(directory / "results/comparison.json"),
            "config": load_json(directory / "results/config.json"), "comparison": comparison,
            "summary": load_json(directory / "results/training_summary.json"),
            "archive": verification["archive"], "archive_sha256": verification["archive_sha256"]}


def verify_model_selection(selection, summary, config, rows):
    validation_seeds = config["validation_seeds"]
    require(selection["seeds"] == validation_seeds, "Selection seeds differ from config")
    require(selection["eval_exploration"] == config["eval_exploration"] and
            selection["max_decisions_per_game"] == config["max_decisions_per_game"],
            "Model-selection evaluation settings changed")
    require(selection["criterion"] == config["selection_metric"], "Selection criterion differs from config")
    candidates = selection["candidate_evaluations"]
    periodic = list(range(config["validation_every_episodes"], summary["completed_episodes"] + 1,
                          config["validation_every_episodes"]))
    require([row["additional_episodes"] for row in candidates] == [0, *periodic, summary["completed_episodes"]],
            "Selection candidates must include initialization, every scheduled interval, and the final learner")
    require([row["reason"] for row in candidates] == ["inherited starting model", *["periodic"] * len(periodic), "final candidate"],
            "Wrong selection candidate ordering/reasons")
    best_metric, winner = -math.inf, None
    for index, row in enumerate(candidates):
        verify_evaluation(row, validation_seeds, config["max_decisions_per_game"], f"selection candidate {index}")
        spread = statistics.pstdev(row["scores"])
        metric = statistics.mean(row["scores"]) - 0.5 * spread
        require(math.isclose(finite(row["population_std"], "selection spread"), spread, rel_tol=1e-12, abs_tol=1e-8),
                "Selection spread is not the population standard deviation")
        require(math.isclose(finite(row["selection_metric"], "selection metric"), metric, rel_tol=1e-12, abs_tol=1e-8),
                "Wrong selection metric")
        episode = integer(row["additional_episodes"], "candidate completed episodes")
        decisions = integer(row["decisions"], "candidate decisions")
        updates = integer(row["updates"], "candidate updates")
        if index == 0:
            require(decisions == updates == episode == 0, "Initial candidate includes new training")
        elif index == len(candidates) - 1:
            require(decisions == summary["total_decisions"] and updates == summary["learning_updates"],
                    "Final candidate does not describe the last learner state")
        else:
            require(decisions == int(rows[episode - 1]["total_steps"]), "Periodic candidate has wrong aggregate decisions")
            scheduled = max(0, decisions // config["train_every_decisions"]
                            - math.ceil(config["learning_starts_decisions"] / config["train_every_decisions"]) + 1)
            require(updates == scheduled, "Periodic candidate has wrong update counter")
        # Use the stored double-precision metric after independently checking its
        # arithmetic, preserving the notebook's exact strict > tie semantics.
        expected_selected = row["selection_metric"] > best_metric
        require(type(row["selected_when_evaluated"]) is bool and row["selected_when_evaluated"] == expected_selected,
                "Selection flag violates strict improvement / earlier tie retention")
        if expected_selected:
            best_metric, winner = row["selection_metric"], row
    require(winner is not None, "No model-selection winner")
    require(selection["selected_additional_episodes"] == summary["selected_additional_episodes"] == winner["additional_episodes"],
            "Selected episode fields disagree")
    require(selection["selected_updates"] == summary["selected_recorded_updates"] == winner["updates"],
            "Selected update counters disagree")
    require(selection["selected_metric"] == summary["selected_validation_metric"] == best_metric,
            "Selected metric fields disagree")
    return winner


def verify_settings(notebook, config, reference_notebook):
    for index, label in EVAL_CELLS.items():
        require(source(notebook["cells"][index]) == source(reference_notebook["cells"][index]),
                f"Changed evaluation-critical source: {label}, cell {index}")
    choices = assignments(notebook["cells"][2])
    fixed = assignments(notebook["cells"][10])
    preview = assignments(notebook["cells"][12])
    for name, key in (("EXPLORATION", "exploration"), ("EPISODES", "episodes_requested"),
                      ("LEARNING_RATE", "learning_rate")):
        require(choices[name] == config[key], f"Notebook/config mismatch: {name}")
    require(0 <= config["exploration"] <= 1 and config["learning_rate"] > 0,
            "Invalid exploration or learning rate")
    integer(config["episodes_requested"], "episodes_requested", 1)
    mapping = {"SEED": "seed", "MAX_STEPS": "max_decisions_per_game", "FRAME_SKIP": "frame_skip",
               "REPLAY_CAPACITY": "replay_capacity", "BATCH_SIZE": "batch_size",
               "WARMUP_STEPS": "warmup_decisions", "LEARNING_STARTS": "learning_starts_decisions",
               "TRAIN_EVERY": "train_every_decisions", "TARGET_EVERY": "target_sync_decisions",
               "GAMMA": "gamma", "EVAL_SEEDS": "eval_seeds", "EVAL_EXPLORATION": "eval_exploration",
               "TRAINING_TIME_LIMIT_SECONDS": "training_time_limit_seconds",
               "TRAINING_REWARD_SCALE": "training_reward_scale", "N_STEP": "n_step",
               "NUM_TRAIN_ENVS": "num_training_envs", "PRIORITY_ALPHA": "priority_alpha",
               "PRIORITY_BETA_START": "priority_beta_start",
               "PRIORITY_BETA_DECISIONS": "priority_beta_decisions",
               "VALIDATION_SEEDS": "validation_seeds", "VALIDATION_EVERY": "validation_every_episodes"}
    for name, key in mapping.items():
        require(fixed[name] == config[key], f"Notebook/config mismatch: {name}")
    expected = {"seed": 42, "environment": "ALE/MsPacman-v5", "frame_skip": 4,
                "sticky_action_probability": 0.25, "noop_max": 30, "grayscale_size": 84,
                "stack_size": 4, "terminal_on_life_loss": False, "max_decisions_per_game": 3000,
                "eval_seeds": EVAL_SEEDS, "eval_exploration": 0.05,
                "n_step": 3, "num_training_envs": 4, "replay_capacity": 50000,
                "batch_size": 32, "warmup_decisions": 1000, "learning_starts_decisions": 10000,
                "train_every_decisions": 4, "target_sync_decisions": 1000, "gamma": 0.99,
                "training_reward_clipping": None, "training_reward_scale": 0.01,
                "training_death_penalty": -0.5, "training_life_loss_terminal": False,
                "priority_alpha": 0.5, "priority_beta_start": 0.4, "priority_beta_end": 1.0,
                "priority_beta_decisions": 2000000, "training_seed_offset": 1069,
                "training_reward_transform": "raw_points * 0.01 - 0.5 * lives_lost (no clipping)",
                "algorithm": "3-step prioritized Double DQN", "training_cap_clock": "time.time wall clock",
                "validation_seeds": list(range(20001, 20011)), "validation_every_episodes": 250,
                "additional_validation_seeds": list(range(30001, 30021)),
                "selection_metric": "mean(raw scores) - 0.5 * population_std(raw scores)",
                "selection_tie_break": "retain earlier checkpoint",
                "preview_seconds": 20, "preview_speed": 4, "preview_plays": 2,
                "preview_stride": 4, "preview_frame_ms": 67}
    for key, value in expected.items():
        require(config[key] == value, f"Unexpected documented setting: {key}")
    groups = [set(config[key]) for key in ("eval_seeds", "validation_seeds", "additional_validation_seeds")]
    require(all(not groups[i] & groups[j] for i in range(3) for j in range(i + 1, 3)),
            "Classroom, selection, and additional validation seeds overlap")
    require(fixed["LIFE_LOSS_PENALTY"] == -config["training_death_penalty"], "Wrong life-loss penalty")
    require(fixed["DEMO_EVERY"] == 25, "Changed periodic evidence interval")
    for name, key in (("PREVIEW_SECONDS", "preview_seconds"), ("PREVIEW_SPEED", "preview_speed"),
                      ("PREVIEW_PLAYS", "preview_plays"), ("SHOW_POPUPS", "show_popups")):
        require(preview[name] == config[key], f"Notebook/config mismatch: {name}")
    warm = config["warm_start"]
    require(warm["run_id"] == fixed["WARM_START_RUN"] == SOURCE_RUN, "Wrong warm-start source run")
    require(warm["sha256"] == fixed["WARM_START_SHA256"] == SOURCE_SHA256, "Wrong warm-start SHA256")
    require(warm["url"] == fixed["WARM_START_URL"], "Wrong warm-start URL")
    require(warm["optimizer_and_replay_reset"] is True, "Warm start misrepresents optimizer/replay reset")
    for field, value in SOURCE_BUDGET.items():
        require(warm[field] == value, f"Wrong inherited warm-start {field}")
    for name, field in (("WARM_START_EPISODES", "completed_episodes"),
                        ("WARM_START_DECISIONS", "decisions"), ("WARM_START_UPDATES", "recorded_updates")):
        require(fixed[name] == warm[field], f"Notebook/config mismatch: {name}")
    return fixed


def verify_zip(archive, run, required_paths):
    """Reading each member through EOF checks its CRC; hashes use bounded buffers."""
    hashes = {}
    with zipfile.ZipFile(archive) as zipped:
        names = zipped.namelist()
        require(len(names) == len(set(names)), "ZIP contains duplicate members")
        require(set(required_paths) <= set(names), f"ZIP omits required files: {set(required_paths) - set(names)}")
        for info in zipped.infolist():
            name = PurePosixPath(info.filename)
            require(not name.is_absolute() and ".." not in name.parts, "Unsafe ZIP member path")
            if info.is_dir():
                continue
            local = run.joinpath(*name.parts)
            require(local.is_file() and local.stat().st_size == info.file_size,
                    f"ZIP/local file missing or wrong size: {info.filename}")
            if info.file_size >= 100 * 1024 * 1024:
                print(f"Streaming ZIP CRC and SHA-256 checks: {info.filename} ({info.file_size:,} bytes)", flush=True)
            with zipped.open(info) as handle:
                zipped_hash = hashlib.file_digest(handle, "sha256").hexdigest()
            require(zipped_hash == sha256_file(local), f"ZIP/file mismatch: {info.filename}")
            hashes[info.filename] = zipped_hash
    return hashes


def comparison_to(after, reference):
    previous = reference["comparison"]["after"]
    deltas = [new - old for new, old in zip(after["scores"], previous["scores"])]
    delta = after["mean"] - previous["mean"]
    return {"comparison_json": reference["comparison_json"], "previous_mean_score": previous["mean"],
            "new_mean_score": after["mean"], "mean_score_change": delta,
            "mean_score_change_percent": 100 * delta / previous["mean"] if previous["mean"] else None,
            "seeds": after["seeds"], "previous_scores": previous["scores"], "new_scores": after["scores"],
            "score_changes": deltas, "improved_games": sum(value > 0 for value in deltas),
            "worsened_games": sum(value < 0 for value in deltas), "unchanged_games": sum(value == 0 for value in deltas)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True, help="Finished run directory, not its ZIP")
    parser.add_argument("--manifest", type=Path,
                        help="Pretraining source manifest; defaults to .execution/advanced_pretraining_manifest.json or its results/ copy")
    args = parser.parse_args()
    run = args.run.expanduser().resolve()
    archive = run.with_suffix(".zip")
    require(run.is_dir() and archive.is_file(), "Finished run directory and final ZIP are required")
    for name in REQUIRED_EVIDENCE:
        require((run / name).is_file(), f"Missing evidence: {name}")
    notebook_path = ROOT / "pacman_dqn.ipynb"
    input_notebook_bytes = notebook_path.read_bytes()
    input_notebook_hash = digest(input_notebook_bytes)
    notebook = json.loads(input_notebook_bytes)
    original_notebook = copy.deepcopy(notebook)
    manifest_path = args.manifest.expanduser().resolve() if args.manifest else ROOT / ".execution/advanced_pretraining_manifest.json"
    if not manifest_path.is_file() and args.manifest is None:
        manifest_path = ROOT / "results/pretraining_manifest.json"
    require(manifest_path.is_file(), "Pretraining source manifest is missing")
    manifest = load_json(manifest_path)
    all_source_hashes = {str(index): digest(source(cell).encode()) for index, cell in enumerate(notebook["cells"])}
    require(all_source_hashes == manifest["cell_source_sha256"], "Notebook sources changed after the pretraining manifest")
    require(sha256_file(ROOT / "experiment_plan.md") == manifest["plan_sha256_before_run"],
            "Experiment plan differs from its pretraining hash")
    code_cells = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]
    counts = [cell.get("execution_count") for cell in code_cells]
    require(len(code_cells) == 28 and counts == list(range(1, 29)), "Notebook must have all 28 code cells executed in order")
    require(not any(output.get("output_type") == "error" for cell in code_cells for output in cell.get("outputs", [])),
            "Notebook contains an error output")
    streams = "\n".join(source({"source": output.get("text", "")})
                        for cell in code_cells for output in cell.get("outputs", []) if output.get("output_type") == "stream")
    require(run.name in streams, "Notebook outputs do not identify this run")
    config = load_json(run / "config.json")
    summary = load_json(run / "training_summary.json")
    baseline = load_json(run / "baseline.json")
    comparison = load_json(run / "comparison.json")
    references = {name: reference_experiment(name) for name in ("scaled_1069", "clipped_2000", "original_100")}
    reference_notebook = load_json(ROOT / references["scaled_1069"]["notebook"])
    require(len(notebook["cells"]) == len(reference_notebook["cells"]), "Notebook cell count changed")
    fixed = verify_settings(notebook, config, reference_notebook)
    require(comparison["baseline_kind"] == "untrained network" and comparison["before"] == baseline,
            "Baseline is missing or mislabeled")
    require(comparison["evaluation_exploration"] == config["eval_exploration"] and
            comparison["max_decisions_per_game"] == config["max_decisions_per_game"], "Comparison evaluation settings changed")
    for phase in ("before", "after"):
        verify_evaluation(comparison[phase], EVAL_SEEDS, 3000, phase)
    require(baseline["scores"] == BASELINE_SCORES and baseline["mean"] == 492,
            "Fresh baseline differs from the documented seed-42 untrained baseline")
    warm_evaluation = None
    warm_path = run / "warm_start_evaluation.json"
    if warm_path.exists():
        warm_evaluation = load_json(warm_path)
        verify_evaluation(warm_evaluation, EVAL_SEEDS, 3000, "warm-start evaluation")
        require(comparison.get("warm_start") == warm_evaluation, "Warm-start evaluation records disagree")
    else:
        require("warm_start" not in comparison, "Comparison refers to a missing warm-start evaluation file")
    with (run / "training.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    accounting = verify_parallel_history(rows, summary, config)
    update_reconciliation = reconcile_updates(summary, config)
    selection = load_json(run / "model_selection.json")
    winner = verify_model_selection(selection, summary, config, rows)
    completed = summary["completed_episodes"]
    periodic_episodes = list(range(fixed["DEMO_EVERY"], completed + 1, fixed["DEMO_EVERY"]))
    demo_path = run / "demo_scores.json"
    require(not periodic_episodes or demo_path.is_file(), "Missing periodic demo score file")
    demos = load_json(demo_path) if demo_path.is_file() else []
    require([demo["episode"] for demo in demos] == periodic_episodes, "Periodic demo inventory is incomplete")
    for demo in demos:
        verify_evaluation(demo, EVAL_SEEDS[:1], 3000, f"episode {demo['episode']} demonstration")
    expected_gifs = ["episode_0000.gif"] + [f"episode_{episode:04d}.gif" for episode in periodic_episodes] + ["final_best.gif"]
    require({path.name for path in (run / "demos").glob("*.gif")} == set(expected_gifs), "Missing or unexpected gameplay GIF")
    checkpoints = ["untrained.pt", "warm_start_source.pt", "initialized.pt", "last_trained.pt",
                   "validation_best.pt", "trained.pt"] + [
        f"episode_{episode:04d}.pt" for episode in periodic_episodes]
    for name in checkpoints + ["learner_state.pt"]:
        require((run / name).is_file() and (run / name).stat().st_size > 0, f"Missing checkpoint: {name}")
    require(sha256_file(run / "warm_start_source.pt") == SOURCE_SHA256, "Warm-start checkpoint has wrong SHA256")

    import torch
    torch.set_num_threads(1)
    baseline_checkpoint = torch.load(run / "untrained.pt", map_location="cpu", weights_only=True)
    source_checkpoint = torch.load(run / "warm_start_source.pt", map_location="cpu", weights_only=True)
    initialized = torch.load(run / "initialized.pt", map_location="cpu", weights_only=True)
    trained = torch.load(run / "trained.pt", map_location="cpu", weights_only=True)
    last_trained = torch.load(run / "last_trained.pt", map_location="cpu", weights_only=True)
    selected_checkpoint = torch.load(run / "validation_best.pt", map_location="cpu", weights_only=True)
    keys = baseline_checkpoint["model"].keys()
    for label, checkpoint in (("baseline", baseline_checkpoint), ("source", source_checkpoint),
                              ("initialized", initialized), ("selected", trained),
                              ("last learner", last_trained), ("validation best", selected_checkpoint)):
        require(checkpoint["n_actions"] == 9 and checkpoint["model"].keys() == keys, f"Wrong {label} model architecture")
        require(all(bool(torch.isfinite(value).all()) for value in checkpoint["model"].values()), f"Non-finite {label} model weights")
    require(source_checkpoint["episode"] == SOURCE_BUDGET["completed_episodes"] and
            source_checkpoint["steps"] == SOURCE_BUDGET["decisions"], "Wrong source checkpoint counters")
    require(all(torch.equal(initialized["model"][key], source_checkpoint["model"][key]) for key in keys),
            "Initialized model differs from inherited source weights")
    changed_from_initialized = any(not torch.equal(trained["model"][key], initialized["model"][key]) for key in keys)
    learner_changed_from_initialized = any(not torch.equal(last_trained["model"][key], initialized["model"][key]) for key in keys)
    changed_from_baseline = any(not torch.equal(trained["model"][key], baseline_checkpoint["model"][key]) for key in keys)
    require(any(not torch.equal(initialized["model"][key], baseline_checkpoint["model"][key]) for key in keys),
            "Warm-start weights unexpectedly equal the fresh untrained network")
    require(learner_changed_from_initialized or summary["learning_updates"] == 0,
            "Updates recorded but last learner equals initialized model")
    require(changed_from_baseline, "Final warm-started model equals fresh untrained baseline")
    require(sha256_file(run / "trained.pt") == sha256_file(run / "validation_best.pt"),
            "Classroom checkpoint is not the exact validation-selected checkpoint")
    if winner["reason"] == "inherited starting model":
        candidate_checkpoint = initialized
    elif winner["reason"] == "final candidate":
        candidate_checkpoint = last_trained
    else:
        candidate_checkpoint = torch.load(run / f"episode_{winner['additional_episodes']:04d}.pt",
                                         map_location="cpu", weights_only=True)
    require(all(torch.equal(trained["model"][key], candidate_checkpoint["model"][key]) for key in keys),
            "Selected weights do not match the winning candidate checkpoint")
    expected_metadata = {"untrained.pt": (0, 0), "initialized.pt": (0, 0),
                         "last_trained.pt": (completed, summary["total_decisions"]),
                         "trained.pt": (winner["additional_episodes"], winner["decisions"]),
                         "validation_best.pt": (winner["additional_episodes"], winner["decisions"])}
    expected_metadata.update({f"episode_{episode:04d}.pt": (episode, int(rows[episode - 1]["total_steps"]))
                              for episode in periodic_episodes})
    loaded = {"untrained.pt": baseline_checkpoint, "initialized.pt": initialized, "trained.pt": trained,
              "last_trained.pt": last_trained, "validation_best.pt": selected_checkpoint}
    for name, (episode, steps) in expected_metadata.items():
        checkpoint = loaded.get(name)
        if checkpoint is None:
            checkpoint = torch.load(run / name, map_location="cpu", weights_only=True)
        require(checkpoint["episode"] == episode and checkpoint["steps"] == steps, f"Wrong counters in {name}")
        require(checkpoint["exploration"] == config["exploration"] and checkpoint["learning_rate"] == config["learning_rate"],
                f"Wrong hyperparameters in {name}")
        require(checkpoint["warm_start_run"] == SOURCE_RUN and
                checkpoint["inherited_episodes"] == SOURCE_BUDGET["completed_episodes"] and
                checkpoint["inherited_decisions"] == SOURCE_BUDGET["decisions"] and
                checkpoint["inherited_updates"] == SOURCE_BUDGET["recorded_updates"], f"Wrong inherited metadata in {name}")
        require(checkpoint["n_actions"] == 9 and checkpoint["model"].keys() == keys and
                all(bool(torch.isfinite(value).all()) for value in checkpoint["model"].values()),
                f"Invalid playback model in {name}")

    evidence_names = list(REQUIRED_EVIDENCE)
    if demo_path.exists():
        evidence_names.append("demo_scores.json")
    if warm_path.exists():
        evidence_names.append("warm_start_evaluation.json")
    required_archive = evidence_names + checkpoints + ["learner_state.pt"] + ["demos/" + name for name in expected_gifs]
    print(f"Verifying archive and {len(expected_gifs)} gameplay outputs for {run.name}...", flush=True)
    archive_hashes = verify_zip(archive, run, required_archive)
    archive_hash = sha256_file(archive)
    gif_outputs = []
    for cell_index, cell in enumerate(notebook["cells"]):
        for output_index, output in enumerate(cell.get("outputs", [])):
            if "image/gif" in output.get("data", {}):
                encoded = output["data"]["image/gif"]
                encoded = "".join(encoded) if isinstance(encoded, list) else encoded
                gif_outputs.append((cell_index, output_index, output, base64.b64decode(encoded)))
    require(len(gif_outputs) == len(expected_gifs), "Notebook GIF count differs from evidence inventory")
    gif_matches = []
    for name, (cell_index, output_index, output, content) in zip(expected_gifs, gif_outputs):
        require(content == (run / "demos" / name).read_bytes(), f"Notebook GIF differs from saved file: {name}")
        public_path = "results/demos/" + name
        url = PUBLIC_RAW + quote(public_path, safe="/")
        output["data"]["text/html"] = f'<img src="{html.escape(url, quote=True)}" alt="{html.escape(Path(name).stem)} gameplay" />'
        gif_matches.append({"file": public_path, "sha256": digest(content), "cell_index": cell_index, "output_index": output_index})
    check = copy.deepcopy(notebook)
    for cell_index, output_index, _, _ in gif_outputs:
        old_data = original_notebook["cells"][cell_index]["outputs"][output_index]["data"]
        new_data = check["cells"][cell_index]["outputs"][output_index]["data"]
        if "text/html" in old_data:
            new_data["text/html"] = old_data["text/html"]
        else:
            new_data.pop("text/html", None)
    require(check == original_notebook, "Unexpected source/output change during HTML display update")
    notebook_bytes = (json.dumps(notebook, indent=1, ensure_ascii=False) + "\n").encode()
    before, after = comparison["before"], comparison["after"]
    deltas = [new - old for new, old in zip(after["scores"], before["scores"])]
    best = max(range(5), key=lambda index: after["scores"][index])
    code_hashes = {str(index): digest(source(cell).encode()) for index, cell in enumerate(notebook["cells"]) if cell["cell_type"] == "code"}
    reward = {"mode": "scaled_raw_points_with_life_loss_penalty", "scale": 0.01, "clipping": None,
              "death_penalty": -0.5, "life_loss_ends_episode": False,
              "transform": config["training_reward_transform"], "n_step": config["n_step"]}
    warm = {**config["warm_start"], "source_checkpoint": relative(run / "warm_start_source.pt"),
            "source_weights_sha256_verified": True, "initialized_weights_match_source": True,
            "initialized_checkpoint_sha256": archive_hashes["initialized.pt"], "evaluation": warm_evaluation}
    selected_model = {
        "criterion": selection["criterion"], "validation_seeds": selection["seeds"],
        "selected_candidate_reason": winner["reason"], "additional_episodes": winner["additional_episodes"],
        "decisions": winner["decisions"], "recorded_updates": winner["updates"],
        "validation_metric": winner["selection_metric"], "validation_evaluation": winner,
        "is_inherited_start": winner["reason"] == "inherited starting model",
        "cumulative_completed_episodes": SOURCE_BUDGET["completed_episodes"] + winner["additional_episodes"],
        "cumulative_decisions": SOURCE_BUDGET["decisions"] + winner["decisions"],
        "cumulative_recorded_updates": SOURCE_BUDGET["recorded_updates"] + winner["updates"],
        "checkpoint": relative(run / "trained.pt"), "checkpoint_sha256": archive_hashes["trained.pt"],
        "weights_differ_from_initialized": changed_from_initialized,
        "weights_equal_validation_winner": True,
    }
    report = {
        "schema_version": "advanced_submission_v1", "run_id": run.name, "run_directory": relative(run),
        "archive": relative(archive), "archive_sha256": archive_hash, "config": config, "summary": summary,
        "evaluation": comparison, "warm_start": warm, "training_reward": reward,
        "model_selection": selection, "selected_model": selected_model,
        "last_learner_weights_differ_from_initialized": learner_changed_from_initialized,
        "historical_references": references, "zero_learning_updates": summary["learning_updates"] == 0,
        "zero_new_learning_updates_note": "Even with zero new updates this run starts from inherited trained weights, not the untrained baseline.",
        "ended_before_episode_budget": completed < config["episodes_requested"],
        "partial_episode_decisions": accounting["partial_game_decisions"], "parallel_game_accounting": accounting,
        "update_count_reconciliation": update_reconciliation, "score_changes": deltas,
        "mean_score_change": after["mean"] - before["mean"],
        "mean_score_change_percent": 100 * (after["mean"] - before["mean"]) / before["mean"] if before["mean"] else None,
        "improved_games": sum(value > 0 for value in deltas), "worsened_games": sum(value < 0 for value in deltas),
        "unchanged_games": sum(value == 0 for value in deltas),
        "best_trained_game": {"game": best + 1, "seed": after["seeds"][best], "score": after["scores"][best],
                              "steps": after["steps"][best], "time_limited": after["time_limited"][best]},
        "score_statistics": {"before": score_statistics(before), "after": score_statistics(after),
                             "warm_start": score_statistics(warm_evaluation) if warm_evaluation else None},
        "time_limited_before": sum(before["time_limited"]), "time_limited_after": sum(after["time_limited"]),
        "training_mean_score": statistics.mean(float(row["score"]) for row in rows) if rows else None,
        "training_blocks_of_25": training_blocks(rows, 25), "training_blocks_of_250": training_blocks(rows, 250),
        "mean_loss_scope": "Each game row averages global learner updates that occurred during that game's lifetime; parallel game intervals overlap.",
        "periodic_demo_scores": demos, "gif_count": len(expected_gifs), "periodic_checkpoint_count": len(periodic_episodes),
        "gifs": ["results/demos/" + name for name in expected_gifs],
        "checkpoints_in_local_zip": checkpoints + ["learner_state.pt"],
        "learner_snapshot": {"file": relative(run / "learner_state.pt"), "size_bytes": (run / "learner_state.pt").stat().st_size,
                             "sha256": archive_hashes["learner_state.pt"], "contents_deserialized_by_verifier": False,
                             "emulator_states_saved": False, "model_role": "last learner, which can differ from validation-selected trained.pt"},
        "elapsed_time_scope": "The timer includes initialization and periodic selection evaluation plus periodic gameplay demos. It stops before the final candidate validation, learner snapshot saving, classroom evaluation, and ZIP creation.",
        "source_hashes": {"notebook_before_html_update": input_notebook_hash,
                          "notebook_after_html_update": digest(notebook_bytes), "code_cells_sha256": code_hashes,
                          "all_cell_sources_sha256": all_source_hashes,
                          "pretraining_manifest_sha256": sha256_file(manifest_path)},
    }
    for name, reference in references.items():
        report["comparison_to_" + name] = comparison_to(after, reference)
    provenance = {"run_id": run.name, "algorithm": config["algorithm"], "warm_start": warm,
                  "model_selection": {"criterion": selection["criterion"], "seeds": selection["seeds"],
                                      "selected_candidate": selected_model, "classroom_scores_used_for_selection": False},
                  "source_url": "https://github.com/pepealonso95/pacman-dqn",
                  "evaluation_reference": references["scaled_1069"]["notebook"],
                  "evaluation_reference_sha256": references["scaled_1069"]["notebook_sha256"],
                  "unchanged_evaluation_source_cells": EVAL_CELLS, "code_cells_sha256": code_hashes,
                  "pretraining_manifest": "results/pretraining_manifest.json",
                  "all_notebook_sources_match_pretraining_manifest": True,
                  "experiment_plan_matches_pretraining_manifest": True,
                  "training_extensions": {key: config[key] for key in (
                      "learning_rate", "replay_capacity", "n_step", "num_training_envs", "priority_alpha",
                      "priority_beta_start", "priority_beta_decisions", "learning_starts_decisions",
                      "training_reward_transform", "training_death_penalty", "training_cap_clock")},
                  "causal_limit": "Several training settings changed together and prior training was inherited; this is not an isolated single-setting experiment.",
                  "display_only_postprocessing": "HTML GIF representations reference identical public files; embedded GIFs, other outputs, metadata, and cell sources remain unchanged."}
    evidence_hashes = {name: archive_hashes[name] for name in evidence_names}
    verification = {
        "verified_at_utc": datetime.now(timezone.utc).isoformat(), "run_id": run.name,
        "executed_code_cells": 28, "execution_counts": counts, "notebook_error_outputs": 0,
        "evaluation_settings_unchanged": True, "evaluation_and_preprocessing_source_unchanged": True,
        "unchanged_evaluation_source_cells": EVAL_CELLS, "fresh_baseline_scores": BASELINE_SCORES,
        "warm_start": warm, "update_count_reconciliation": update_reconciliation,
        "model_selection": {"candidate_count": len(selection["candidate_evaluations"]),
                            "all_metrics_and_flags_verified": True, "selected_model": selected_model,
                            "trained_checkpoint_matches_validation_best_bytes": True},
        "parallel_game_accounting": accounting, "training_reward": reward,
        "all_playback_model_weights_finite": True, "all_model_weights_finite": True,
        "trained_weights_differ_from_baseline": changed_from_baseline,
        "trained_weights_differ_from_initialized": changed_from_initialized,
        "last_learner_weights_differ_from_initialized": learner_changed_from_initialized,
        "all_notebook_sources_match_pretraining_manifest": True,
        "experiment_plan_matches_pretraining_manifest": True,
        "pretraining_manifest_sha256": sha256_file(manifest_path),
        "archive": relative(archive), "archive_sha256": archive_hash, "archive_integrity": "passed",
        "archive_crc_check": "Every file member was read to EOF through ZipExtFile with CRC validation.",
        "archive_members_byte_matched_to_local_files": list(archive_hashes), "archive_member_sha256": archive_hashes,
        "large_snapshot_deserialized": False, "expected_periodic_checkpoints_present": periodic_episodes,
        "notebook_sha256": digest(notebook_bytes), "gif_output_count": len(gif_matches), "gif_matches": gif_matches,
        "github_gif_display": "Added HTML image representations; all original embedded GIF bytes and other outputs retained.",
        "evidence_files_sha256": evidence_hashes,
    }
    # Verify JSON serializability and the unchanged input before touching published files.
    for value in (report, provenance, verification):
        json.dumps(value, allow_nan=False)
    require(sha256_file(notebook_path) == input_notebook_hash, "Notebook changed during verification; refusing to overwrite it")
    results = ROOT / "results"
    (results / "demos").mkdir(parents=True, exist_ok=True)
    for name in evidence_names:
        shutil.copy2(run / name, results / name)
    if manifest_path.resolve() != (results / "pretraining_manifest.json").resolve():
        shutil.copy2(manifest_path, results / "pretraining_manifest.json")
    if not demo_path.exists():
        write_json(results / "demo_scores.json", [])
    if not warm_path.exists():
        (results / "warm_start_evaluation.json").unlink(missing_ok=True)
    for path in (results / "demos").glob("*.gif"):
        if path.name not in expected_gifs:
            path.unlink()
    for name in expected_gifs:
        shutil.copy2(run / "demos" / name, results / "demos" / name)
    # Previous supplemental evaluations belong to an archived experiment. New
    # validation, if performed, must be copied after this assembly step.
    for name in ("heldout_validation.json", "heldout_validation.csv"):
        (results / name).unlink(missing_ok=True)
    (results / "environment.txt").write_text(
        f"Python: {config['python']}\nPlatform: {config['platform']}\nTraining device: {config['device']}\n" +
        "".join(f"{name}=={version}\n" for name, version in config["packages"].items()))
    with tempfile.NamedTemporaryFile(mode="wb", suffix=".ipynb.tmp", dir=ROOT, delete=False) as handle:
        temporary = Path(handle.name)
        handle.write(notebook_bytes)
    temporary.replace(notebook_path)
    write_json(results / "report_data.json", report)
    write_json(results / "provenance.json", provenance)
    write_json(results / "verification.json", verification)
    print(f"Verified {run.name}: {completed:,} completed additional games, "
          f"{summary['learning_updates']:,} recorded new updates, {len(expected_gifs)} GIFs.")
    print(f"Raw classroom mean: {before['mean']:g} untrained -> {after['mean']:g} final.")
    print(f"Full checkpoint ZIP retained locally: {archive}")


if __name__ == "__main__":
    main()
