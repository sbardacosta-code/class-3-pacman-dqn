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
import copy
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
CLIPPED_REFERENCE = ROOT / "experiments" / "clipped_2000"
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
    "eval_exploration", "eval_seeds",
    "preview_seconds", "preview_speed", "preview_plays", "preview_stride",
    "preview_frame_ms",
)
REWARD_FIELDS = {
    "training_reward_clipping", "training_reward_scale",
    "training_reward_transform", "training_death_penalty",
}
RUNTIME_FIELDS = {"device", "python", "platform", "packages"}
SCALED_REWARD_DESCRIPTION = "raw_points * 0.01 (no clipping)"


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


def reconcile_learning_updates(summary: dict, config: dict) -> dict:
    """Keep the recorded counter; disclose one possible interrupted update."""
    frequency = config["train_every_decisions"]
    scheduled = max(0, summary["total_decisions"] // frequency
                    - math.ceil(config["warmup_decisions"] / frequency) + 1)
    recorded = summary["learning_updates"]
    gap = scheduled - recorded
    interruption_gap = (summary["status"] == "interrupted" and gap == 1
                        and summary["total_decisions"] % frequency == 0)
    require(gap == 0 or interruption_gap,
            f"Unexpected update count: {recorded} recorded vs {scheduled} scheduled")
    return {
        "scheduled_learning_updates": scheduled,
        "recorded_learning_updates": recorded,
        "scheduled_minus_recorded": gap,
        "interrupted_update_completion_uncertain": interruption_gap,
        "explanation": (
            "The decision counter advances before learn() and the update counter advances "
            "after learn() returns. Interruption at an update boundary left one scheduled "
            "update unrecorded; whether its optimizer step started or completed is unknown. "
            "The recorded update count is preserved, not increased."
            if interruption_gap else "The recorded update count matches scheduled updates."
        ),
    }


def ast_key(node: ast.AST) -> str:
    """Compare code structure while allowing comments and formatting to differ."""
    return ast.dump(node, include_attributes=False)


def cell_code_key(cell: dict) -> str:
    code = source(cell)
    # The unchanged package-install cell contains IPython syntax, not Python.
    return code if code.lstrip().startswith("%") else ast_key(ast.parse(code))


def config_dictionary(tree: ast.Module) -> ast.Dict:
    assignments = [node for node in tree.body if isinstance(node, ast.Assign)
                   and len(node.targets) == 1
                   and isinstance(node.targets[0], ast.Name)
                   and node.targets[0].id == "config"]
    require(len(assignments) == 1 and isinstance(assignments[0].value, ast.Dict),
            "Expected one config dictionary assignment in the settings-record cell.")
    return assignments[0].value


def replay_tuple(tree: ast.Module) -> ast.Tuple:
    calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)
             and isinstance(node.func, ast.Attribute) and node.func.attr == "append"
             and ast_key(node.func.value) == ast_key(ast.parse("self.items", mode="eval").body)]
    require(len(calls) == 1 and len(calls[0].args) == 1
            and isinstance(calls[0].args[0], ast.Tuple)
            and len(calls[0].args[0].elts) == 6,
            "Replay transition storage no longer matches the supplied six-field tuple.")
    return calls[0].args[0]


def verify_reward_experiment(notebook: dict, config: dict) -> tuple[dict, dict | None]:
    """Verify the selected reward rule and isolate scaled-reward code changes.

    Returns reward evidence and, for the scaled experiment, preserved comparator
    evidence. This never executes notebook code or mutates any experiment files.
    """
    replay = ast.parse(source(notebook["cells"][19]))
    stored_reward = replay_tuple(replay).elts[2]
    clipping = config.get("training_reward_clipping")
    if clipping == [-1, 1]:
        require(config.get("training_reward_scale", 1.0) == 1.0,
                "Clipped-mode config unexpectedly scales training rewards.")
        require(config.get("training_death_penalty", 0.0) == 0.0,
                "A death penalty was not authorized for the clipped experiment.")
        require(ast_key(stored_reward) == ast_key(ast.parse(
                    "float(np.clip(reward, -1, 1))", mode="eval").body),
                "Clipped reward config does not match replay storage source.")
        return ({"mode": "clipped", "clipping": [-1, 1], "scale": 1.0,
                 "transform": "clip(raw_points, -1, 1)", "death_penalty": 0.0,
                 "source_and_config_verified": True}, None)

    require("training_reward_clipping" in config and clipping is None,
            "Scaled rewards require an explicit null training_reward_clipping.")
    require(config.get("training_reward_scale") == 0.01,
            "Scaled reward config must record training_reward_scale = 0.01.")
    require(config.get("training_reward_transform") == SCALED_REWARD_DESCRIPTION,
            "Scaled reward config must describe raw_points * 0.01 without clipping.")
    require(config.get("training_death_penalty") == 0.0,
            "Scaled reward config must explicitly record zero added death penalty.")
    require(ast_key(stored_reward) == ast_key(ast.parse(
                "float(reward * TRAINING_REWARD_SCALE)", mode="eval").body),
            "Replay storage is not exactly raw reward times TRAINING_REWARD_SCALE.")
    require(assignment_values(notebook["cells"][10]).get("TRAINING_REWARD_SCALE") == 0.01,
            "The notebook reward-scale constant must be 0.01.")

    reference_nb_path = CLIPPED_REFERENCE / "pacman_dqn.ipynb"
    require(reference_nb_path.is_file(), "Preserve the clipped 2,000-episode experiment first.")
    previous_nb = load_json(reference_nb_path)
    previous_config = load_json(CLIPPED_REFERENCE / "results" / "config.json")
    previous_comparison = load_json(CLIPPED_REFERENCE / "results" / "comparison.json")
    previous_summary = load_json(CLIPPED_REFERENCE / "results" / "training_summary.json")
    previous_verification = load_json(CLIPPED_REFERENCE / "results" / "verification.json")
    require(previous_config["training_reward_clipping"] == [-1, 1],
            "The preserved comparison experiment is not the clipped-reward run.")
    require(len(previous_nb["cells"]) == len(notebook["cells"]),
            "Cell count changed relative to the clipped-reward experiment.")
    previous_settings = {k: v for k, v in previous_config.items()
                         if k not in REWARD_FIELDS | RUNTIME_FIELDS}
    current_settings = {k: v for k, v in config.items()
                        if k not in REWARD_FIELDS | RUNTIME_FIELDS}
    require(current_settings == previous_settings,
            "A non-reward training/evaluation setting changed versus clipped_2000.")

    # Every code cell must have the same AST except the explicit scale constant,
    # replay reward expression, and descriptive reward config entries.
    changed_code_cells = []
    for index, (previous_cell, current_cell) in enumerate(zip(previous_nb["cells"], notebook["cells"])):
        require(previous_cell["cell_type"] == current_cell["cell_type"],
                f"Cell type changed versus clipped_2000 at index {index}.")
        if current_cell["cell_type"] != "code":
            continue
        if cell_code_key(previous_cell) != cell_code_key(current_cell):
            changed_code_cells.append(index)
            require(index in {10, 19, 43},
                    f"Unexpected code change versus clipped_2000 in cell {index}.")
    require(set(changed_code_cells) == {10, 19, 43},
            f"Expected only reward-related cells 10, 19, 43 to change; got {changed_code_cells}.")

    fixed_tree = ast.parse(source(notebook["cells"][10]))
    scale_statements = [node for node in fixed_tree.body if isinstance(node, ast.Assign)
                        and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name)
                        and node.targets[0].id == "TRAINING_REWARD_SCALE"]
    require(len(scale_statements) == 1, "Expected exactly one reward-scale constant.")
    fixed_tree.body.remove(scale_statements[0])
    require(ast_key(fixed_tree) == ast_key(ast.parse(source(previous_nb["cells"][10]))),
            "Fixed settings source changed beyond adding the reward-scale constant.")

    previous_replay = ast.parse(source(previous_nb["cells"][19]))
    normalized_replay = copy.deepcopy(replay)
    replay_tuple(normalized_replay).elts[2] = copy.deepcopy(replay_tuple(previous_replay).elts[2])
    require(ast_key(normalized_replay) == ast_key(previous_replay),
            "Replay source changed beyond the stored reward expression.")

    previous_config_tree = ast.parse(source(previous_nb["cells"][43]))
    current_config_tree = ast.parse(source(notebook["cells"][43]))
    current_dictionary = config_dictionary(current_config_tree)
    key_names = [ast.literal_eval(key) for key in current_dictionary.keys]
    require(len(set(key_names)) == len(key_names), "The recorded config has duplicate keys.")
    config_nodes = dict(zip(key_names, current_dictionary.values))
    expected_nodes = {
        "training_reward_clipping": "None",
        "training_reward_scale": "TRAINING_REWARD_SCALE",
        "training_reward_transform": repr(SCALED_REWARD_DESCRIPTION),
        "training_death_penalty": "0.0",
    }
    for key, expected in expected_nodes.items():
        require(key in config_nodes and ast_key(config_nodes[key]) == ast_key(ast.parse(expected, mode="eval").body),
                f"Notebook config source does not faithfully record {key}.")
    for tree in (previous_config_tree, current_config_tree):
        dictionary = config_dictionary(tree)
        retained = [(key, value) for key, value in zip(dictionary.keys, dictionary.values)
                    if ast.literal_eval(key) not in REWARD_FIELDS]
        dictionary.keys = [key for key, _ in retained]
        dictionary.values = [value for _, value in retained]
    require(ast_key(previous_config_tree) == ast_key(current_config_tree),
            "Settings-record source changed beyond reward metadata.")

    # Confirm preservation before the finished new run replaces root evidence.
    reference_notebook_hash = sha256_file(reference_nb_path)
    require(reference_notebook_hash == previous_verification["notebook_sha256"],
            "The preserved clipped-run notebook no longer matches its saved hash.")
    previous_archive = ROOT / previous_verification["archive"]
    require(previous_archive.is_file(), "The previous full local ZIP is missing.")
    require(sha256_file(previous_archive) == previous_verification["archive_sha256"],
            "The previous full local ZIP no longer matches its saved hash.")
    for filename, expected_hash in previous_verification["evidence_files_sha256"].items():
        require(sha256_file(CLIPPED_REFERENCE / "results" / filename) == expected_hash,
                f"Preserved clipped-run evidence changed: {filename}")
    for key in ("seeds", "scores", "mean"):
        require(previous_comparison["before"][key] is not None,
                f"Preserved comparison lacks its untrained {key}.")

    reward_evidence = {
        "mode": "scaled_raw_points", "clipping": None, "scale": 0.01,
        "transform": SCALED_REWARD_DESCRIPTION, "death_penalty": 0.0,
        "source_and_config_verified": True,
        "only_reward_training_setting_changed_vs_clipped_2000": True,
        "changed_code_cells_vs_clipped_2000_zero_based": changed_code_cells,
    }
    reference = {
        "notebook": relative(reference_nb_path),
        "notebook_sha256": reference_notebook_hash,
        "runtime_field_differences": {
            key: {"previous": previous_config.get(key), "current": config.get(key)}
            for key in sorted(RUNTIME_FIELDS)
            if previous_config.get(key) != config.get(key)
        },
        "config": previous_config,
        "config_sha256": sha256_file(CLIPPED_REFERENCE / "results" / "config.json"),
        "comparison": previous_comparison, "summary": previous_summary,
        "archive": relative(previous_archive),
        "archive_sha256": previous_verification["archive_sha256"],
        "preserved_archive_and_evidence_verified": True,
    }
    return reward_evidence, reference


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
    reward_evidence, clipped_reference = verify_reward_experiment(notebook, config)
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
    update_count_reconciliation = reconcile_learning_updates(summary, config)

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
        "training_reward": reward_evidence,
        "update_count_reconciliation": update_count_reconciliation,
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
    if clipped_reference is not None:
        previous_after = clipped_reference["comparison"]["after"]
        require(previous_after["seeds"] == after["seeds"],
                "Scaled and clipped experiments used different evaluation seeds.")
        previous_mean = previous_after["mean"]
        changes_vs_clipped = [new - previous for new, previous
                              in zip(after["scores"], previous_after["scores"])]
        report["clipped_2000_reference"] = clipped_reference
        report["runtime_field_differences_vs_clipped_2000"] = clipped_reference["runtime_field_differences"]
        report["comparison_to_clipped_2000"] = {
            "comparison_json": "experiments/clipped_2000/results/comparison.json",
            "previous_mean_score": previous_mean,
            "new_mean_score": after["mean"],
            "mean_score_change": after["mean"] - previous_mean,
            "mean_score_change_percent": (
                100 * (after["mean"] - previous_mean) / previous_mean
                if previous_mean else None
            ),
            "mean_score_multiple": after["mean"] / previous_mean if previous_mean else None,
            "seeds": after["seeds"], "previous_scores": previous_after["scores"],
            "new_scores": after["scores"], "score_changes": changes_vs_clipped,
            "improved_games": sum(delta > 0 for delta in changes_vs_clipped),
            "worsened_games": sum(delta < 0 for delta in changes_vs_clipped),
            "unchanged_games": sum(delta == 0 for delta in changes_vs_clipped),
            "previous_training_budget": clipped_reference["summary"],
            "new_training_budget": summary,
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
            "training_reward": reward_evidence,
        },
        "evaluation_and_preprocessing_cells_unchanged": UNCHANGED_CELLS,
        "execution": "Local Jupyter kernel through nbclient; all 28 code cells executed in order",
        "run_id": run.name,
        "display_only_postprocessing": "Added HTML image representations to byte-matched GIF outputs; all embedded GIF bytes and other outputs retained.",
    }
    if clipped_reference is not None:
        provenance["training_comparison_reference"] = {
            key: value for key, value in clipped_reference.items()
            if key not in {"config", "comparison", "summary"}
        }
        provenance["only_reward_training_setting_changed_vs_clipped_2000"] = True
        provenance["runtime_field_differences_vs_clipped_2000"] = clipped_reference["runtime_field_differences"]

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
        "update_count_reconciliation": update_count_reconciliation,
        "training_reward": reward_evidence,
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
