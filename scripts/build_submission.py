#!/usr/bin/env python3
"""Verify a finished run, publish selected evidence, and retain notebook outputs.

Run only AFTER run_notebook.py has finished all cells:
    .venv/bin/python scripts/build_submission.py --run pacman_runs/RUN_ID

This does not train, evaluate, change cell source, write README.md, or rewrite the
full local ZIP. The ZIP keeps all playback checkpoints; results/ keeps evidence.
"""

from __future__ import annotations

import argparse
import ast
import base64
import csv
from datetime import datetime, timezone
import hashlib
import html
import json
import math
from pathlib import Path
import shutil
import statistics
import tempfile
from urllib.parse import quote
import zipfile


ROOT = Path(__file__).resolve().parents[1]
ORIGINAL = ROOT / "experiments" / "original_100"
PUBLIC_RAW = "https://raw.githubusercontent.com/sbardacosta-code/class-3-pacman-dqn/main/"
REQUIRED_FILES = (
    "config.json", "baseline.json", "comparison.json", "training.csv",
    "training_summary.json", "demo_scores.json", "training_dashboard.png",
)
# Exact function/constant cells whose meaning must not change for evaluation.
UNCHANGED_CELLS = {
    12: "preview settings", 15: "environment and preprocessing",
    17: "network architecture", 21: "action selection",
    30: "evaluation function", 45: "untrained evaluation",
    56: "saved-model evaluation", 58: "score comparison and final outputs",
}
UNCHANGED_SETTINGS = (
    "seed", "environment", "frame_skip", "sticky_action_probability",
    "noop_max", "grayscale_size", "stack_size", "terminal_on_life_loss",
    "max_decisions_per_game", "warmup_decisions", "batch_size",
    "train_every_decisions", "target_sync_decisions", "gamma",
    "training_reward_clipping", "eval_exploration", "eval_seeds",
    "preview_seconds", "preview_speed", "preview_plays", "preview_stride",
    "preview_frame_ms",
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def load_json(path: Path):
    return json.loads(path.read_text())


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def source(cell: dict) -> str:
    value = cell.get("source", "")
    return "".join(value) if isinstance(value, list) else value


def relative(path: Path) -> str:
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return str(path)


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def numeric(value):
    number = float(value)
    return number if math.isfinite(number) else None


def blocks(rows: list[dict], size: int) -> list[dict]:
    return [
        {
            "first_episode": int(block[0]["episode"]),
            "last_episode": int(block[-1]["episode"]),
            "count": len(block),
            "mean_score": statistics.mean(float(r["score"]) for r in block),
            "mean_episode_loss": (
                statistics.mean(losses) if (losses := [
                    numeric(r["mean_loss"]) for r in block
                    if numeric(r["mean_loss"]) is not None
                ]) else None
            ),
        }
        for start in range(0, len(rows), size)
        if (block := rows[start:start + size])
    ]


def assignment_values(cell: dict) -> dict:
    values = {}
    for statement in ast.parse(source(cell)).body:
        if isinstance(statement, ast.Assign) and len(statement.targets) == 1:
            target = statement.targets[0]
            if isinstance(target, ast.Name):
                try:
                    values[target.id] = ast.literal_eval(statement.value)
                except (ValueError, TypeError):
                    pass
    return values


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True,
                        help="The completed run directory, not its ZIP.")
    args = parser.parse_args()
    run = args.run.expanduser().resolve()
    require(run.is_dir(), f"Run directory does not exist: {run}")
    archive = run.with_suffix(".zip")
    require(archive.is_file(), f"Final archive is missing: {archive}")
    for filename in REQUIRED_FILES:
        require((run / filename).is_file(), f"Missing run evidence: {filename}")

    config = load_json(run / "config.json")
    baseline = load_json(run / "baseline.json")
    comparison = load_json(run / "comparison.json")
    summary = load_json(run / "training_summary.json")
    demos = load_json(run / "demo_scores.json")
    original_config = load_json(ORIGINAL / "results" / "config.json")
    notebook_path = ROOT / "pacman_dqn.ipynb"
    original_nb = load_json(ORIGINAL / "pacman_dqn.ipynb")
    notebook = load_json(notebook_path)
    original_sources = [source(cell) for cell in original_nb["cells"]]
    current_sources = [source(cell) for cell in notebook["cells"]]
    require(len(original_sources) == len(current_sources), "Unexpected notebook cell count.")

    code_cells = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]
    require(len(code_cells) == 28, f"Expected 28 code cells, got {len(code_cells)}.")
    counts = [cell.get("execution_count") for cell in code_cells]
    require(counts == list(range(1, 29)), f"Cells were not all executed in order: {counts}")
    for index, cell in enumerate(notebook["cells"]):
        require(not any(output.get("output_type") == "error"
                        for output in cell.get("outputs", [])),
                f"Notebook cell {index} contains an error output.")
    for index, label in UNCHANGED_CELLS.items():
        require(current_sources[index] == original_sources[index],
                f"Evaluation invariant changed: {label} (cell {index}).")
    for key in UNCHANGED_SETTINGS:
        require(config[key] == original_config[key], f"Fixed setting changed: {key}")
    choices = assignment_values(notebook["cells"][2])
    fixed = assignment_values(notebook["cells"][10])
    for name, key in (("EXPLORATION", "exploration"),
                      ("EPISODES", "episodes_requested"),
                      ("LEARNING_RATE", "learning_rate")):
        require(choices[name] == config[key], f"Notebook/config mismatch: {name}")
    for name, key in (("EVAL_SEEDS", "eval_seeds"),
                      ("EVAL_EXPLORATION", "eval_exploration"),
                      ("MAX_STEPS", "max_decisions_per_game"),
                      ("REPLAY_CAPACITY", "replay_capacity")):
        require(fixed[name] == config[key], f"Notebook/config mismatch: {name}")
    streams = "\n".join(
        "".join(output.get("text", ""))
        for cell in code_cells for output in cell.get("outputs", [])
        if output.get("output_type") == "stream"
    )
    require(run.name in streams, "Notebook outputs do not name this run directory.")
    require(comparison["baseline_kind"] == "untrained network", "Wrong baseline kind.")
    require(baseline == comparison["before"], "Baseline and comparison disagree.")
    require(comparison["evaluation_exploration"] == config["eval_exploration"],
            "Comparison exploration differs from config.")
    require(comparison["max_decisions_per_game"] == config["max_decisions_per_game"],
            "Comparison time limit differs from config.")
    for phase in ("before", "after"):
        evaluation = comparison[phase]
        require(evaluation["seeds"] == config["eval_seeds"], f"Wrong {phase} seeds.")
        require(len(evaluation["scores"]) == 5, f"Expected five {phase} scores.")
        require(all(math.isfinite(float(x)) for x in evaluation["scores"]),
                f"Non-finite {phase} score.")
        require(math.isclose(statistics.mean(evaluation["scores"]), evaluation["mean"]),
                f"Incorrect {phase} mean.")
        require(len(evaluation["steps"]) == len(evaluation["time_limited"]) == 5,
                f"Incomplete {phase} lengths/time-limit data.")
        require(all(0 < step <= config["max_decisions_per_game"]
                    for step in evaluation["steps"]), f"Wrong {phase} game length.")

    with (run / "training.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    completed = int(summary["completed_episodes"])
    require(len(rows) == completed, "CSV row count differs from completed episodes.")
    require([int(r["episode"]) for r in rows] == list(range(1, completed + 1)),
            "CSV episodes are missing or out of order.")
    running_steps = 0
    for row in rows:
        running_steps += int(row["steps"])
        require(int(row["total_steps"]) == running_steps, "CSV total-step inconsistency.")
    require(summary["total_decisions"] >= running_steps, "Summary undercounts CSV decisions.")
    require(completed <= config["episodes_requested"], "Completed more than requested.")
    require(summary["status"] in ("completed", "interrupted", "failed", "time_budget"),
            f"Unexpected training status: {summary['status']}")
    if summary["status"] == "completed":
        require(completed == config["episodes_requested"], "Early stop falsely marked completed.")
        require(summary["total_decisions"] == running_steps, "Completed run has partial episode.")
    warmup = config["warmup_decisions"]
    frequency = config["train_every_decisions"]
    expected_updates = max(0, summary["total_decisions"] // frequency
                           - math.ceil(warmup / frequency) + 1)
    require(summary["learning_updates"] == expected_updates,
            f"Unexpected update count: {summary['learning_updates']} vs {expected_updates}")

    demo_every = fixed["DEMO_EVERY"]
    periodic_episodes = list(range(demo_every, completed + 1, demo_every))
    expected_gifs = ["episode_0000.gif"] + [
        f"episode_{episode:04d}.gif" for episode in periodic_episodes
    ] + ["final_best.gif"]
    require([d["episode"] for d in demos] == periodic_episodes,
            "Periodic demo records do not cover all expected episodes.")
    gif_paths = [run / "demos" / name for name in expected_gifs]
    require(set(path.name for path in (run / "demos").glob("*.gif")) == set(expected_gifs),
            "Missing or unexpected gameplay GIF files.")
    checkpoint_names = ["untrained.pt", "trained.pt"] + [
        f"episode_{episode:04d}.pt" for episode in periodic_episodes
    ]
    for name in checkpoint_names:
        require((run / name).is_file(), f"Missing playback checkpoint: {name}")

    # The archive is verified, not regenerated; every source file remains local.
    archive_files = []
    with zipfile.ZipFile(archive) as zipped:
        require(zipped.testzip() is None, "ZIP CRC integrity check failed.")
        zipped_names = set(zipped.namelist())
        required_archive_paths = list(REQUIRED_FILES) + checkpoint_names + [
            "demos/" + name for name in expected_gifs
        ]
        for name in required_archive_paths:
            require(name in zipped_names, f"ZIP omits {name}")
            with zipped.open(name) as handle:
                zipped_hash = hashlib.file_digest(handle, "sha256").hexdigest()
            require(zipped_hash == sha256_file(run / name), f"ZIP/file mismatch: {name}")
            archive_files.append(name)

    # Match in actual display order; duplicate GIF bytes across demos are valid.
    gif_outputs = []
    for cell_index, cell in enumerate(notebook["cells"]):
        for output_index, output in enumerate(cell.get("outputs", [])):
            if "image/gif" in output.get("data", {}):
                encoded = output["data"]["image/gif"]
                encoded = "".join(encoded) if isinstance(encoded, list) else encoded
                gif_outputs.append((cell_index, output_index, output,
                                    base64.b64decode(encoded)))
    require(len(gif_outputs) == len(gif_paths),
            f"Notebook has {len(gif_outputs)} GIF outputs; expected {len(gif_paths)}.")
    gif_matches = []
    for (cell_index, output_index, output, data), path in zip(gif_outputs, gif_paths):
        require(data == path.read_bytes(), f"Notebook GIF differs from {path.name}")
        public_path = "results/demos/" + path.name
        url = PUBLIC_RAW + quote(public_path, safe="/")
        output["data"]["text/html"] = (
            f'<img src="{html.escape(url, quote=True)}" '
            f'alt="{html.escape(path.stem)} gameplay" />'
        )
        gif_matches.append({"file": public_path, "sha256": digest(data),
                            "cell_index": cell_index, "output_index": output_index})
    require([source(c) for c in notebook["cells"]] == current_sources,
            "Source changed while adding HTML displays.")

    # Check model evidence with the same PyTorch environment used to execute.
    import torch
    untrained = torch.load(run / "untrained.pt", map_location="cpu", weights_only=True)
    trained = torch.load(run / "trained.pt", map_location="cpu", weights_only=True)
    require(trained["episode"] == completed, "Trained checkpoint episode count disagrees.")
    require(trained["steps"] == summary["total_decisions"], "Checkpoint step count disagrees.")
    require(untrained["model"].keys() == trained["model"].keys(), "Model keys changed.")
    finite = all(bool(torch.isfinite(weight).all()) for weight in trained["model"].values())
    changed = any(not torch.equal(weight, untrained["model"][key])
                  for key, weight in trained["model"].items())
    require(finite, "Trained weights contain non-finite values.")
    require(changed or summary["learning_updates"] == 0, "Updates reported but weights unchanged.")

    before, after = comparison["before"], comparison["after"]
    best_index = max(range(5), key=lambda i: after["scores"][i])
    deltas = [a - b for a, b in zip(after["scores"], before["scores"])]
    report = {
        "run_id": run.name, "run_directory": relative(run),
        "archive": relative(archive), "config": config, "summary": summary,
        "zero_learning_updates": summary["learning_updates"] == 0,
        "ended_before_episode_budget": completed < config["episodes_requested"],
        "partial_episode_decisions": summary["total_decisions"] - running_steps,
        "evaluation": comparison,
        "score_changes": deltas,
        "mean_score_change": after["mean"] - before["mean"],
        "mean_score_change_percent": (
            100 * (after["mean"] - before["mean"]) / before["mean"]
            if before["mean"] else None
        ),
        "improved_games": sum(x > 0 for x in deltas),
        "worsened_games": sum(x < 0 for x in deltas),
        "unchanged_games": sum(x == 0 for x in deltas),
        "best_trained_game": {
            "game": best_index + 1, "seed": after["seeds"][best_index],
            "score": after["scores"][best_index],
            "steps": after["steps"][best_index],
            "time_limited": after["time_limited"][best_index],
        },
        "time_limited_before": sum(before["time_limited"]),
        "time_limited_after": sum(after["time_limited"]),
        "training_mean_score": statistics.mean(float(r["score"]) for r in rows) if rows else None,
        "training_blocks_of_25": blocks(rows, 25),
        "training_blocks_of_250": blocks(rows, 250),
        "periodic_demo_scores": demos,
        "gif_count": len(expected_gifs), "periodic_checkpoint_count": len(periodic_episodes),
        "gifs": ["results/demos/" + name for name in expected_gifs],
        "checkpoints_in_local_zip": checkpoint_names,
    }
    changed_cells = [index for index, (old, new) in enumerate(zip(original_sources, current_sources))
                     if old != new]
    old_provenance = load_json(ORIGINAL / "results" / "provenance.json")
    provenance = {
        "source_url": old_provenance["source_url"],
        "notebook_source_url": old_provenance["notebook_source_url"],
        "downloaded_notebook_sha256": old_provenance["downloaded_notebook_sha256"],
        "original_experiment_notebook": "experiments/original_100/pacman_dqn.ipynb",
        "original_experiment_notebook_sha256": sha256_file(ORIGINAL / "pacman_dqn.ipynb"),
        "notebook_cell_sources_changed": bool(changed_cells),
        "changed_source_cell_indices_zero_based": changed_cells,
        "documented_extensions": {
            "algorithm": config.get("algorithm", "DQN"),
            "replay_capacity": config["replay_capacity"],
            "training_time_limit_seconds": config.get("training_time_limit_seconds"),
        },
        "evaluation_and_preprocessing_cells_unchanged": UNCHANGED_CELLS,
        "execution": "Local Jupyter kernel through nbclient; all 28 code cells executed in order",
        "run_id": run.name,
        "display_only_postprocessing": "Added HTML image representations to byte-matched GIF outputs; all embedded GIF bytes and other outputs retained.",
    }

    # Nothing above mutates files. Only publish evidence after every check passes.
    results = ROOT / "results"
    (results / "demos").mkdir(parents=True, exist_ok=True)
    for filename in REQUIRED_FILES:
        shutil.copy2(run / filename, results / filename)
    for old_gif in (results / "demos").glob("*.gif"):
        if old_gif.name not in expected_gifs:
            old_gif.unlink()
    for path in gif_paths:
        shutil.copy2(path, results / "demos" / path.name)
    (results / "environment.txt").write_text(
        f"Python: {config['python']}\nPlatform: {config['platform']}\n"
        f"Training device: {config['device']}\n" + "".join(
            f"{package}=={version}\n" for package, version in config["packages"].items()
        )
    )
    # Atomic notebook replacement prevents losing the executed result on a short write.
    with tempfile.NamedTemporaryFile(mode="w", suffix=".ipynb.tmp", dir=ROOT,
                                     delete=False) as handle:
        temporary = Path(handle.name)
        json.dump(notebook, handle, indent=1, ensure_ascii=False)
        handle.write("\n")
    temporary.replace(notebook_path)
    verification = {
        "verified_at_utc": datetime.now(timezone.utc).isoformat(),
        "run_id": run.name, "executed_code_cells": len(code_cells),
        "execution_counts": counts, "notebook_error_outputs": 0,
        "evaluation_settings_unchanged": True,
        "evaluation_and_preprocessing_source_unchanged": True,
        "cell_sources_match_original": not changed_cells,
        "all_model_weights_finite": finite,
        "trained_weights_differ_from_baseline": changed,
        "archive": relative(archive), "archive_sha256": sha256_file(archive),
        "archive_integrity": "passed",
        "archive_members_byte_matched_to_local_files": archive_files,
        "expected_periodic_checkpoints_present": periodic_episodes,
        "notebook_sha256": sha256_file(notebook_path),
        "gif_output_count": len(gif_matches), "gif_matches": gif_matches,
        "github_gif_display": "HTML image representations added; original embedded GIF data and all other outputs preserved.",
        "evidence_files_sha256": {
            filename: sha256_file(results / filename) for filename in REQUIRED_FILES
        },
    }
    write_json(results / "report_data.json", report)
    write_json(results / "provenance.json", provenance)
    write_json(results / "verification.json", verification)
    print(f"Verified and assembled run {run.name}: {completed} episodes, "
          f"{summary['learning_updates']:,} updates, {len(gif_matches)} GIFs.")
    print(f"Evaluation mean: {before['mean']:g} -> {after['mean']:g}")
    print(f"README data: {results / 'report_data.json'}")
    print(f"Full checkpoint archive retained unchanged: {archive}")


if __name__ == "__main__":
    main()
