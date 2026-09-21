#!/usr/bin/env python3
"""Prepare, but never execute, a notebook that resumes a finished advanced run.

The source run and its verified reproduction ZIP are read-only. The caller must
wait for phase 1's notebook process to exit and archive that executed notebook
before making this generated notebook the root submission.
"""

from __future__ import annotations

import argparse
import ast
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path, PurePosixPath
import pprint
import stat
import zipfile


ROOT = Path(__file__).resolve().parents[1]
EVALUATION_CELLS = (15, 17, 21, 30)
TOTAL_ADVANCED_STARTS = 6000
STARTUP_RESERVE_SECONDS = 120
FIXED_MEMBERS = (
    "learner_state.pt", "config.json", "training_summary.json", "training.csv",
    "model_selection.json", "demo_scores.json", "validation_best.pt", "last_trained.pt",
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_json(path):
    return json.loads(path.read_text())


def sha256_file(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def source(cell):
    return "".join(cell["source"]) if isinstance(cell["source"], list) else cell["source"]


def set_source(notebook, index, value):
    notebook["cells"][index]["source"] = value.splitlines(keepends=True)


def replace_once(text, old, new):
    require(text.count(old) == 1, f"Expected one source anchor: {old[:90]!r}")
    return text.replace(old, new, 1)


def utc_datetime(value):
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    require(parsed.tzinfo is not None, "UTC timestamps must include Z or a timezone offset.")
    return parsed.astimezone(timezone.utc)


def relative_or_absolute(path):
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path.resolve())


def verify_bundle(bundle, run, completed):
    expected = set(FIXED_MEMBERS)
    for episode in range(25, completed + 1, 25):
        expected.add(f"episode_{episode:04d}.pt")
        expected.add(f"demos/episode_{episode:04d}.gif")
    members = {}
    with zipfile.ZipFile(bundle) as archive:
        infos = archive.infolist()
        names = [info.filename for info in infos if not info.is_dir()]
        require(len(names) == len(set(names)) and set(names) == expected,
                "Resume ZIP must contain exactly the required root-level files and periodic evidence.")
        for info in infos:
            path = PurePosixPath(info.filename)
            require(not path.is_absolute() and ".." not in path.parts and
                    stat.S_IFMT(info.external_attr >> 16) != stat.S_IFLNK,
                    f"Unsafe ZIP member: {info.filename}")
            if info.is_dir():
                require(info.filename == "demos/", "Unexpected ZIP directory.")
                continue
            local = run / info.filename
            require(local.is_file() and local.stat().st_size == info.file_size,
                    f"Missing or size-mismatched phase 1 source: {info.filename}")
            digest = sha256_file(local)
            with archive.open(info) as handle:
                archived_digest = hashlib.file_digest(handle, "sha256").hexdigest()
            require(digest == archived_digest, f"ZIP content mismatch: {info.filename}")
            members[info.filename] = {"sha256": digest, "size": info.file_size}
    return dict(sorted(members.items()))


RESTORE_CELL = r'''if _training_started:
    raise RuntimeError("This continuation already started; rerun from section 5a for a new run.")
_training_started = True

# Count bundle restoration and restored GIF display inside the phase 2 budget.
_phase1 = CONTINUATION["phase1_summary"]
_phase1_elapsed = float(_phase1["elapsed_seconds_including_periodic_demos"])
_phase1_active = float(_phase1.get("active_monotonic_elapsed_seconds", _phase1_elapsed))
started = time.time() - _phase1_elapsed
active_started = time.monotonic() - _phase1_active

import csv
import gc
import stat
import zipfile
from pathlib import PurePosixPath

def _continuation_sha256(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()

_bundle = Path(CONTINUATION["resume_bundle_local_path"])
_cache_root = Path("resume_bundles")
_cache_root.mkdir(exist_ok=True)
if not _bundle.is_file():
    _bundle = _cache_root / Path(CONTINUATION["resume_bundle_local_path"]).name
    if not _bundle.is_file():
        print("Downloading the pinned phase 1 learner-state bundle...", flush=True)
        _temporary = _bundle.with_suffix(".download")
        urllib.request.urlretrieve(CONTINUATION["resume_bundle_url"], _temporary)
        if _continuation_sha256(_temporary) != CONTINUATION["resume_bundle_sha256"]:
            _temporary.unlink()
            raise ValueError("Downloaded resume bundle checksum mismatch.")
        _temporary.replace(_bundle)
if _continuation_sha256(_bundle) != CONTINUATION["resume_bundle_sha256"]:
    raise ValueError("Resume bundle checksum mismatch; no snapshot will be loaded.")
_resume_dir = _cache_root / (CONTINUATION["source_run_id"] + "_" +
                             CONTINUATION["resume_bundle_sha256"][:12])
_resume_dir.mkdir(exist_ok=True)
_expected = CONTINUATION["resume_bundle_members"]
with zipfile.ZipFile(_bundle) as _archive:
    _infos = _archive.infolist()
    _names = [entry.filename for entry in _infos if not entry.is_dir()]
    if len(_names) != len(set(_names)) or set(_names) != set(_expected):
        raise ValueError("Resume bundle has unexpected or duplicate members.")
    for _entry in _infos:
        _member = PurePosixPath(_entry.filename)
        if (_member.is_absolute() or ".." in _member.parts or
                stat.S_IFMT(_entry.external_attr >> 16) == stat.S_IFLNK):
            raise ValueError("Unsafe resume bundle path.")
        if _entry.is_dir():
            if _entry.filename != "demos/":
                raise ValueError("Unexpected bundle directory.")
            continue
        _record = _expected[_entry.filename]
        if _entry.file_size != _record["size"]:
            raise ValueError("Resume bundle member size mismatch.")
        _destination = _resume_dir / _entry.filename
        _destination.parent.mkdir(parents=True, exist_ok=True)
        if (_destination.is_file() and _destination.stat().st_size == _record["size"] and
                _continuation_sha256(_destination) == _record["sha256"]):
            continue
        _temporary = _destination.with_suffix(_destination.suffix + ".extracting")
        with _archive.open(_entry) as _input, _temporary.open("wb") as _output:
            shutil.copyfileobj(_input, _output, length=1024 * 1024)
        if _continuation_sha256(_temporary) != _record["sha256"]:
            _temporary.unlink()
            raise ValueError("Resume bundle member checksum mismatch.")
        _temporary.replace(_destination)

assert json.loads((_resume_dir / "training_summary.json").read_text()) == _phase1
assert json.loads((_resume_dir / "config.json").read_text()) == CONTINUATION["phase1_config"]
_phase1_selection = json.loads((_resume_dir / "model_selection.json").read_text())
assert _phase1_selection == CONTINUATION["phase1_selection"]
_snapshot_path = _resume_dir / "learner_state.pt"
assert _continuation_sha256(_snapshot_path) == CONTINUATION["source_snapshot_sha256"]

print("Restoring the pinned final phase 1 learner, target, Adam optimizer and replay...", flush=True)
# This pickle is accepted only after verifying the entire predeclared bundle and
# member hashes. Never replace it with an untrusted or unverified snapshot.
_snapshot = torch.load(_snapshot_path, map_location="cpu", weights_only=False)
assert _snapshot["summary"] == _phase1 and _snapshot["emulator_states_saved"] is False
_last = torch.load(_resume_dir / "last_trained.pt", map_location="cpu", weights_only=True)
assert _last["n_actions"] == N_ACTIONS and _last["episode"] == _phase1["completed_episodes"]
assert _last["steps"] == _phase1["total_decisions"]
assert _last["model"].keys() == _snapshot["model"].keys()
assert all(torch.equal(_last["model"][key], value) for key, value in _snapshot["model"].items())
del _last
model.load_state_dict(_snapshot["model"], strict=True)
target.load_state_dict(_snapshot["target"], strict=True)
model.train()
target.eval()
optimizer.load_state_dict(_snapshot["optimizer"])
# Adam's non-capturable step tensors may correctly remain on CPU. Its moment
# tensors must follow the parameters onto the current training device.
for _state in optimizer.state.values():
    for _key, _value in _state.items():
        if torch.is_tensor(_value):
            if _key != "step":
                _state[_key] = _value.to(DEVICE)
            assert torch.isfinite(_state[_key]).all().item()
assert all(group["lr"] == LEARNING_RATE for group in optimizer.param_groups)
assert all(torch.isfinite(value).all().item()
           for network in (model, target) for value in network.state_dict().values())
replay.load_state_dict(_snapshot["replay"])
_discarded_pending = sum(len(queue) for queue in replay.pending)
assert _discarded_pending <= NUM_TRAIN_ENVS * (N_STEP - 1)
for _queue in replay.pending:
    _queue.clear()
_restored_rng = {key: _snapshot[key] for key in
                 ("train_rng", "python_rng", "numpy_rng", "torch_rng")}
if "mps_rng" in _snapshot:
    _restored_rng["mps_rng"] = _snapshot["mps_rng"]
del _snapshot
gc.collect()

total_steps = int(_phase1["total_decisions"])
updates = int(_phase1["learning_updates"])
episodes_started = int(_phase1["episodes_started"])
completed_episodes = int(_phase1["completed_episodes"])
training_lives_lost = int(_phase1["training_lives_lost"])
loss_sum = 0.0  # Only new games use this accumulator; old CSV losses stay unchanged.
training_states = []
history = []
with (_resume_dir / "training.csv").open(newline="") as _handle:
    for _row in csv.DictReader(_handle):
        for _key in ("episode", "training_episode_id", "seed", "steps", "total_steps", "lives_lost"):
            _row[_key] = int(_row[_key])
        for _key in ("score", "exploration", "mean_loss", "elapsed_seconds", "shaped_return"):
            _row[_key] = float(_row[_key])
        for _key in ("terminated", "truncated"):
            _row[_key] = {"True": True, "False": False}[_row[_key]]
        history.append(_row)
assert len(history) == completed_episodes
assert total_steps == sum(row["steps"] for row in history) + sum(
    game["decisions"] for game in CONTINUATION["abandoned_partial_games"])
assert episodes_started == completed_episodes + len(CONTINUATION["abandoned_partial_games"])
demo_history = json.loads((_resume_dir / "demo_scores.json").read_text())
selection_history = [dict(row, origin_run_id=CONTINUATION["source_run_id"])
                     for row in _phase1_selection["candidate_evaluations"]]
best_validation_metric = float(_phase1_selection["selected_metric"])
selected_episode = int(_phase1_selection["selected_additional_episodes"])
selected_updates = int(_phase1_selection["selected_updates"])
shutil.copy2(_resume_dir / "validation_best.pt", RUN_DIR / "validation_best.pt")
for _episode in range(DEMO_EVERY, completed_episodes + 1, DEMO_EVERY):
    for _name in (f"episode_{_episode:04d}.pt", f"demos/episode_{_episode:04d}.gif"):
        shutil.copy2(_resume_dir / _name, RUN_DIR / _name)
save_metrics(history, RUN_DIR / "training.csv")
(RUN_DIR / "demo_scores.json").write_text(json.dumps(demo_history, indent=2))
_selection_record = dict(_phase1_selection, candidate_evaluations=selection_history)
(RUN_DIR / "model_selection.json").write_text(json.dumps(_selection_record, indent=2))
config["continuation"]["discarded_pending_transitions"] = _discarded_pending
(RUN_DIR / "config.json").write_text(json.dumps(config, indent=2))
print(f"Restored {completed_episodes} completed advanced games, {episodes_started} starts, "
      f"{total_steps:,} decisions and {updates:,} recorded updates.", flush=True)
print(f"Abandoned {len(CONTINUATION['abandoned_partial_games'])} unfinished phase 1 games; "
      f"discarded {_discarded_pending} pending n-step tails. Emitted replay remains intact.", flush=True)
print(f"The validation incumbent is still the model at {selected_episode} advanced games "
      f"(metric {best_validation_metric:.3f}); no initial validation is repeated.", flush=True)
for _demo in demo_history:
    _episode = int(_demo["episode"])
    print(f"Restored phase 1 demonstration after {_episode} games; "
          f"recorded score {_demo['scores'][0]}", flush=True)
    show_sample(RUN_DIR / "demos" / f"episode_{_episode:04d}.gif",
                f"Restored phase 1 history: after {_episode} training games")

status = "completed"
train_envs = []
try:
    for slot in range(NUM_TRAIN_ENVS):
        check_training_budget()
        train_envs.append(make_env())
        training_states.append(start_training_game(train_envs[-1], slot))
    # Restore global RNGs after model/device/environment initialization. The
    # fresh environments have explicit, unused seeds and their own RNG states.
    train_rng.setstate(_restored_rng["train_rng"])
    random.setstate(_restored_rng["python_rng"])
    np.random.set_state(_restored_rng["numpy_rng"])
    torch.set_rng_state(_restored_rng["torch_rng"].cpu())
    if DEVICE.type == "mps" and "mps_rng" in _restored_rng:
        torch.mps.set_rng_state(_restored_rng["mps_rng"].cpu())
    del _restored_rng
    print("OVERNIGHT_COLLECTION_STARTED", flush=True)
    train_batched_games(train_envs)
except TrainingBudgetReached:
    status = "time_budget"
    print("Fixed continuation time budget reached. Saving and evaluating.", flush=True)
except KeyboardInterrupt:
    status = "interrupted"
    print("Interrupted once. Saving actual completed training.", flush=True)
except Exception:
    status = "failed"
    raise
finally:
    print("OVERNIGHT_COLLECTION_FINISHED", flush=True)
    for env in train_envs:
        env.close()
    save_training_result(status)
print("Saved:", RUN_DIR)
if updates == _phase1["learning_updates"]:
    print("No new learning updates occurred in the continuation.")
'''


SUMMARY_ADDITION = '''    summary["continuation"] = {**config["continuation"],
        "phase2_completed_episodes": completed_episodes - _phase1["completed_episodes"],
        "phase2_episodes_started": episodes_started - _phase1["episodes_started"],
        "phase2_decisions": total_steps - _phase1["total_decisions"],
        "phase2_recorded_updates": updates - _phase1["learning_updates"],
        "phase2_elapsed_seconds": stopped_wall - _phase1_elapsed,
        "phase2_active_monotonic_elapsed_seconds": stopped_active - _phase1_active}
    summary["selected_checkpoint_origin_run_id"] = next(
        row["origin_run_id"] for row in reversed(selection_history) if row["selected_when_evaluated"])
'''


def prepare(args, *, now=None):
    now = now or datetime.now(timezone.utc)
    deadline = utc_datetime(args.deadline_utc)
    phase2_seconds = int((deadline - now).total_seconds()) - STARTUP_RESERVE_SECONDS
    require(phase2_seconds > 0, "No continuation time remains before the requested deadline.")
    run = args.phase1.resolve()
    summary = read_json(run / "training_summary.json")
    config = read_json(run / "config.json")
    selection = read_json(run / "model_selection.json")
    require(summary["status"] in ("completed", "interrupted", "time_budget"),
            "Only a successfully saved phase 1 run may be resumed.")
    require("continuation" not in config, "This generator expects the original advanced phase 1.")
    require(summary["episodes_started"] < TOTAL_ADVANCED_STARTS,
            "The combined started-game budget is already exhausted.")
    require((run / "comparison.json").is_file() and zipfile.is_zipfile(run.with_suffix(".zip")),
            "Wait for phase 1's final class evaluation, ZIP, and notebook process to finish.")
    require(selection["candidate_evaluations"][-1]["reason"] == "final candidate" and
            selection["candidate_evaluations"][-1]["decisions"] == summary["total_decisions"] and
            selection["candidate_evaluations"][-1]["updates"] == summary["learning_updates"],
            "The phase 1 final validation candidate is missing.")
    require(args.resume_url.startswith("https://"), "The resume bundle needs a public HTTPS URL.")
    bundle_members = verify_bundle(args.resume_bundle, run, summary["completed_episodes"])
    notebook = read_json(args.notebook)
    original_manifest = read_json(args.original_manifest)
    for index, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] == "code":
            require(hashlib.sha256(source(cell).encode()).hexdigest() ==
                    original_manifest["cell_source_sha256"].get(str(index)),
                    f"Phase 1 source differs from its frozen manifest in cell {index}.")
    require(sum(cell["cell_type"] == "code" for cell in notebook["cells"]) == 28,
            "Expected the original 28 code cells.")
    phase1_evaluation_sources = {index: source(notebook["cells"][index]) for index in EVALUATION_CELLS}
    spec = {
        "schema_version": 1, "source_run_id": run.name,
        "source_run_path": relative_or_absolute(run),
        "source_snapshot_sha256": bundle_members["learner_state.pt"]["sha256"],
        "phase1_config": config, "phase1_summary": summary, "phase1_selection": selection,
        "resume_bundle_local_path": relative_or_absolute(args.resume_bundle),
        "resume_bundle_url": args.resume_url,
        "resume_bundle_sha256": sha256_file(args.resume_bundle),
        "resume_bundle_members": bundle_members,
        "phase2_time_limit_seconds": phase2_seconds,
        "preparation_utc": now.isoformat(), "deadline_utc": deadline.isoformat(),
        "startup_reserve_seconds": STARTUP_RESERVE_SECONDS,
        "total_advanced_started_game_budget": TOTAL_ADVANCED_STARTS,
        "abandoned_partial_games": summary["partial_games"],
        "discarded_pending_transitions": None,
        "emulator_states_restored": False, "optimizer_and_replay_restored": True,
    }
    set_source(notebook, 2, "# Authorized overnight continuation; EPISODES caps total advanced game starts.\n"
               "# The 1,069 original inherited games remain separate in recorded totals.\n"
               "EXPLORATION = 0.10\nEPISODES = 6000\nLEARNING_RATE = 0.00005\n")
    constants = replace_once(source(notebook["cells"][10]),
                             "TRAINING_TIME_LIMIT_SECONDS = 2 * 60 * 60",
                             "TRAINING_TIME_LIMIT_SECONDS = " + repr(
                                 summary["elapsed_seconds_including_periodic_demos"] + phase2_seconds))
    constants += "\n# Frozen continuation inputs, checksums and reproducible relative budget.\n"
    constants += "CONTINUATION = " + pprint.pformat(spec, width=110, sort_dicts=False) + "\n"
    set_source(notebook, 10, constants)
    config_source = replace_once(source(notebook["cells"][43]),
        '(RUN_DIR / "config.json").write_text(json.dumps(config, indent=2))',
        'config["continuation"] = dict(CONTINUATION)\n'
        '(RUN_DIR / "config.json").write_text(json.dumps(config, indent=2))')
    set_source(notebook, 43, config_source)
    warm_source = replace_once(source(notebook["cells"][45]),
        'print("Additional training starts from these weights with a fresh optimizer and replay.")',
        'print("Original 1,069-game warm start verified. The continuation cell next restores "\n'
        '      "the full final phase 1 learner, optimizer and replay before any new training.")')
    set_source(notebook, 45, warm_source)
    selection_source = replace_once(source(notebook["cells"][49]),
        '"reason": reason, **result,', '"reason": reason, "origin_run_id": RUN_DIR.name, **result,')
    set_source(notebook, 49, selection_source)
    summary_source = replace_once(source(notebook["cells"][51]),
        '    (RUN_DIR / "training_summary.json").write_text(json.dumps(summary, indent=2))',
        SUMMARY_ADDITION + '    (RUN_DIR / "training_summary.json").write_text(json.dumps(summary, indent=2))')
    set_source(notebook, 51, summary_source)
    set_source(notebook, 53, RESTORE_CELL)
    set_source(notebook, 1, source(notebook["cells"][1]) +
        "\n**This saved overnight experiment:** 10% exploration and learning rate 0.00005 continue unchanged. "
        "The 6,000-game ceiling counts all advanced game starts across both phases; "
        "unfinished phase 1 games consume starts but are not counted as completed. "
        "A fixed additional wall-clock budget is recorded below.\n")
    set_source(notebook, 9, "### Continued training, with the same evaluation\n\n"
        "This notebook measures a genuine untrained baseline, verifies the original 1,069-game warm start, "
        "and then restores the final learner from the first advanced training phase. Three-step prioritized "
        "Double DQN, four environments, reward scaling/life penalty, exploration and learning rate remain unchanged. "
        "The replay, optimizer, target network, sampling RNG and decision schedules continue. "
        "The best separate validation checkpoint remains the incumbent.\n\n"
        "At the phase boundary, emulator state was not saved. Unfinished games are recorded as abandoned; "
        "their at-most-eight pending n-step tails are discarded, while emitted replay remains valid. "
        "Fresh games use previously unused training seeds. This is learner-state continuation, "
        "not an exact mid-game emulator resume. The fixed additional budget makes later reruns possible; "
        "the original absolute cutoff is retained as provenance.\n")
    set_source(notebook, 39, "## 5. continue the saved learner\n\n"
        "The next cells create a new results folder, measure the true untrained network and original warm start, "
        "then restore the pinned phase 1 learner bundle. Phase 1 training records, checkpoint files and actual "
        "saved GIFs are carried forward. Global advanced counters keep their values; only newly started games "
        "are played. The best model is selected on the same ten development seeds, never on the class five "
        "or the twenty final held-out seeds. Interrupt once if needed, then finish evaluation and export.\n")
    set_source(notebook, 59, "## 7. save and explain the experiment\n\n"
        "The new ZIP retains the combined training records, all periodic GIFs/checkpoints and the final learner "
        "snapshot. Keep it locally. The executed notebook and README identify phase 1 and phase 2 work "
        "separately, including abandoned games and partial decisions. Run All repeats this continuation from "
        "the checksum-pinned resume bundle, downloading its published release if the local copy is absent. "
        "Only the selected final policy is used for the fixed classroom evaluation.\n")
    for index in EVALUATION_CELLS:
        require(source(notebook["cells"][index]) == phase1_evaluation_sources[index],
                f"Evaluation cell {index} changed.")
    for index in (2, 10, 43, 45, 49, 51, 53):
        ast.parse(source(notebook["cells"][index]))
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            cell["execution_count"] = None
            cell["outputs"] = []
            cell.get("metadata", {}).pop("execution", None)
    output = json.dumps(notebook, indent=1, ensure_ascii=False) + "\n"
    plan_hash = sha256_file(args.plan)
    manifest = {
        "notebook_sha256_before_run": hashlib.sha256(output.encode()).hexdigest(),
        "plan_sha256_before_run": plan_hash,
        "overnight_plan_sha256": plan_hash,
        "inherited_original_plan_sha256": original_manifest["plan_sha256_before_run"],
        "source_phase1_notebook_sha256": sha256_file(args.notebook),
        "source_phase1_manifest_sha256": sha256_file(args.original_manifest),
        "continuation": spec,
        "cell_source_sha256": {str(index): hashlib.sha256(source(cell).encode()).hexdigest()
                               for index, cell in enumerate(notebook["cells"])},
    }
    require(args.output.resolve() != args.manifest_output.resolve(),
            "Notebook and manifest outputs must differ.")
    for path in (args.output, args.manifest_output):
        require(not path.exists(), f"Refusing to replace an existing output: {path}")
        path.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(output)
    args.manifest_output.write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
    print(f"Prepared {args.output}\nFrozen manifest: {args.manifest_output}\n"
          f"Fixed phase 2 budget: {phase2_seconds} seconds; no notebook cells executed.")
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase1", required=True, type=Path)
    parser.add_argument("--deadline-utc", required=True)
    parser.add_argument("--resume-bundle", required=True, type=Path)
    parser.add_argument("--resume-url", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--manifest-output", required=True, type=Path)
    parser.add_argument("--notebook", type=Path, default=ROOT / "pacman_dqn.ipynb")
    parser.add_argument("--plan", type=Path, default=ROOT / "overnight_plan.md")
    parser.add_argument("--original-manifest", type=Path,
                        default=ROOT / ".execution/advanced_pretraining_manifest.json")
    prepare(parser.parse_args(argv))


if __name__ == "__main__":
    main()
