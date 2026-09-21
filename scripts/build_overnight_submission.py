#!/usr/bin/env python3
"""Verify the completed overnight learner continuation and assemble its evidence.

Run only after the continuation notebook has completed all 28 code cells and ZIP:
    .venv/bin/python scripts/build_overnight_submission.py --run pacman_runs/RUN_ID

The advanced verifier's shared helpers perform unchanged evaluation, checkpoint,
ZIP, and display checks. This module adds explicit continuation accounting and
selection-history checks without changing build_advanced_submission.py. No model
is trained/evaluated, and neither large learner snapshot is deserialized here.
"""
from __future__ import annotations

import argparse
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

import build_advanced_submission as advanced
from build_advanced_submission import (
    ROOT, PUBLIC_RAW, SOURCE_RUN, SOURCE_SHA256, SOURCE_BUDGET, EVAL_SEEDS,
    BASELINE_SCORES, EVAL_CELLS, REQUIRED_EVIDENCE, require, load_json,
    sha256_file, digest, source, relative, write_json, integer, finite, csv_bool,
    score_statistics, verify_evaluation, training_blocks, reference_experiment,
    verify_zip, comparison_to,
)


def continuation_records(config, summary):
    declared, actual = config['continuation'], summary['continuation']
    require(declared['schema_version'] == actual['schema_version'] == 1,
            'Unsupported continuation schema')
    for key, value in declared.items():
        if key == 'discarded_pending_transitions' and value is None:
            continue
        require(actual.get(key) == value, f'Config/summary continuation mismatch: {key}')
    require(actual['optimizer_and_replay_restored'] is True and
            actual['emulator_states_restored'] is False,
            'Continuation must restore learner state while resetting emulators')
    require(actual['source_run_id'] != SOURCE_RUN, 'Phase 1 must be the advanced run, not its original warm start')
    prior = actual['phase1_summary']
    require(actual['abandoned_partial_games'] == prior['partial_games'],
            'Abandoned games differ from the unfinished phase-1 games')
    require(config['episodes_requested'] == actual['total_advanced_started_game_budget'] == 6000,
            'The overnight ceiling is 6,000 total advanced game starts')
    discarded = integer(actual['discarded_pending_transitions'], 'discarded n-step tails')
    require(discarded <= config['num_training_envs'] * (config['n_step'] - 1),
            'Discarded n-step tails exceed the per-environment pending capacity')
    for key, global_key in (
        ('phase2_completed_episodes', 'completed_episodes'),
        ('phase2_episodes_started', 'episodes_started'),
        ('phase2_decisions', 'total_decisions'),
        ('phase2_recorded_updates', 'learning_updates'),
    ):
        value = integer(actual[key], key)
        require(summary[global_key] == prior[global_key] + value,
                f'Phase-2 delta does not reconcile: {key}')
    for key, global_key in (
        ('phase2_elapsed_seconds', 'elapsed_seconds_including_periodic_demos'),
        ('phase2_active_monotonic_elapsed_seconds', 'active_monotonic_elapsed_seconds'),
    ):
        value = finite(actual[key], key)
        require(value >= 0 and math.isclose(summary[global_key], prior[global_key] + value,
                                          rel_tol=1e-10, abs_tol=0.05),
                f'Phase-2 timer does not reconcile: {key}')
    require(finite(actual['phase2_time_limit_seconds'], 'phase-2 time cap') > 0,
            'Invalid phase-2 time cap')
    for name in ('preparation_utc', 'deadline_utc'):
        value = datetime.fromisoformat(actual[name].replace('Z', '+00:00'))
        require(value.tzinfo is not None, f'{name} must include a timezone')
    return actual


def verify_settings(notebook, config, reference_notebook):
    fixed = advanced.verify_settings(notebook, config, reference_notebook)
    declared = config['continuation']
    previous = declared['phase1_config']
    permitted = {'episodes_requested', 'training_time_limit_seconds', 'continuation',
                 'device', 'python', 'platform', 'packages'}
    for key, value in previous.items():
        if key not in permitted:
            require(config.get(key) == value, f'Unannounced continuation setting changed: {key}')
    require(config['episodes_requested'] == 6000, 'Wrong global started-game ceiling')
    return fixed


def verify_parallel_history(rows, summary, config):
    continuation = continuation_records(config, summary)
    abandoned = continuation['abandoned_partial_games']
    completed = integer(summary['completed_episodes'], 'completed_episodes')
    started = integer(summary['episodes_started'], 'episodes_started')
    total = integer(summary['total_decisions'], 'total_decisions')
    require(len(rows) == completed and completed <= started <= config['episodes_requested'],
            'Completed CSV rows or global game-start counters are invalid')
    require([int(row['episode']) for row in rows] == list(range(1, completed + 1)),
            'Combined CSV completion numbers have gaps or duplicates')
    current = summary['partial_games']
    require(len(current) <= config['num_training_envs'] and len(abandoned) <= config['num_training_envs'],
            'Too many current or abandoned parallel games')
    ids, seeds = [], []
    completed_steps, completed_lives = 0, 0
    previous_total, previous_elapsed = 0, 0.0
    end_time = finite(summary['elapsed_seconds_including_periodic_demos'], 'combined elapsed time')
    for row in rows:
        episode_id, seed = int(row['training_episode_id']), int(row['seed'])
        steps, aggregate, lives = int(row['steps']), int(row['total_steps']), int(row['lives_lost'])
        require(1 <= steps <= config['max_decisions_per_game'] and lives >= 0,
                'Invalid completed-game decisions or lost lives')
        completed_steps += steps
        completed_lives += lives
        require(previous_total < aggregate <= total and completed_steps <= aggregate,
                'Invalid combined global CSV decision counter')
        previous_total = aggregate
        elapsed = finite(row['elapsed_seconds'], 'CSV elapsed time')
        require(previous_elapsed <= elapsed <= end_time + 0.05, 'Invalid combined CSV time ordering')
        previous_elapsed = elapsed
        ended, truncated = csv_bool(row['terminated'], 'terminated'), csv_bool(row['truncated'], 'truncated')
        require(ended or truncated, 'Completed CSV row lacks an ending flag')
        require(not truncated or steps == config['max_decisions_per_game'], 'Early time-limit flag')
        score, shaped = finite(row['score'], 'training score'), finite(row['shaped_return'], 'shaped return')
        expected_reward = score * config['training_reward_scale'] + lives * config['training_death_penalty']
        require(math.isclose(shaped, expected_reward, rel_tol=1e-9, abs_tol=1e-7), 'Wrong shaped return')
        expected_exploration = 1.0 if aggregate < config['warmup_decisions'] else config['exploration']
        require(finite(row['exploration'], 'exploration') == expected_exploration,
                'Warm-up/exploration schedule was restarted or changed')
        loss = float(row['mean_loss'])
        require(math.isnan(loss) or (math.isfinite(loss) and loss >= 0), 'Invalid episode loss')
        ids.append(episode_id)
        seeds.append(seed)
    incomplete_steps, incomplete_lives = {}, {}
    for label, games in (('abandoned', abandoned), ('current', current)):
        slots = []
        steps_sum = lives_sum = 0
        for game in games:
            slot = integer(game['slot'], label + ' slot')
            require(slot < config['num_training_envs'], 'Invalid environment slot')
            steps = integer(game['decisions'], label + ' decisions')
            require(steps <= config['max_decisions_per_game'], 'Incomplete game exceeds time limit')
            lives = integer(game['lives_lost'], label + ' lives lost')
            finite(game['score'], label + ' score')
            slots.append(slot)
            steps_sum += steps
            lives_sum += lives
            ids.append(integer(game['training_episode_id'], label + ' training ID', 1))
            seeds.append(integer(game['seed'], label + ' seed'))
        require(len(set(slots)) == len(slots), f'Duplicate {label} environment slot')
        incomplete_steps[label], incomplete_lives[label] = steps_sum, lives_sum
    require(sorted(ids) == list(range(1, started + 1)), 'Started game IDs are missing or repeated across phases')
    require(len(set(seeds)) == len(seeds), 'Training reused a seed across the resume boundary')
    require(all(seed == config['seed'] + config['training_seed_offset'] + episode_id
                for seed, episode_id in zip(seeds, ids)), 'Wrong global training seed offset')
    reserved = set(config['eval_seeds'] + config['validation_seeds'] + config['additional_validation_seeds'])
    require(not set(seeds) & reserved, 'Training reused an evaluation seed')
    require(started == completed + len(abandoned) + len(current), 'Started/completed/abandoned/current counts disagree')
    require(completed_steps + sum(incomplete_steps.values()) == total,
            'Completed + abandoned + current decisions do not equal global total')
    require(completed_lives + sum(incomplete_lives.values()) == summary['training_lives_lost'],
            'Lost-life totals do not reconcile across phases')
    status = summary['status']
    require(status in ('completed', 'time_budget', 'interrupted', 'failed'), 'Unknown continuation status')
    if status == 'completed':
        require(started == config['episodes_requested'] and not current,
                'Incomplete total started-game budget falsely marked completed')
    require(summary['training_time_limit_seconds'] == config['training_time_limit_seconds'],
            'Summary/config training cap mismatch')
    replay_size = integer(summary['replay_size'], 'replay_size')
    pending_allowance = config['num_training_envs'] * (config['n_step'] - 1) + continuation['discarded_pending_transitions']
    require(min(config['replay_capacity'], max(0, total - pending_allowance)) <= replay_size <= min(total, config['replay_capacity']),
            'Replay size inconsistent with global decisions, discarded tails, and pending tails')
    for suffix, prior in (('completed_episodes', SOURCE_BUDGET['completed_episodes']),
                           ('decisions', SOURCE_BUDGET['decisions']),
                           ('recorded_updates', SOURCE_BUDGET['recorded_updates'])):
        require(summary['inherited_' + suffix] == prior, 'Wrong original inherited budget')
        current_value = {'completed_episodes': completed, 'decisions': total,
                         'recorded_updates': summary['learning_updates']}[suffix]
        require(summary['cumulative_' + suffix] == prior + current_value, 'Wrong cumulative budget')
    return {'completed_game_decisions': completed_steps,
            'partial_game_decisions': incomplete_steps['current'], 'partial_game_count': len(current),
            'abandoned_game_decisions': incomplete_steps['abandoned'], 'abandoned_game_count': len(abandoned),
            'training_ids_and_seeds_unique': True, 'aggregate_csv_totals_verified': True,
            'explanation': 'Global total_steps includes completed games, phase-1 abandoned partial games, and current partial games; CSV total_steps is the aggregate counter at completion.'}


def reconcile_updates(summary, config):
    prior = config['continuation']['phase1_summary']
    prior_config = config['continuation']['phase1_config']
    prior_reconciliation = advanced.reconcile_updates(prior, prior_config)
    inherited_gap = prior_reconciliation['scheduled_minus_recorded']
    adjusted = dict(summary, learning_updates=summary['learning_updates'] + inherited_gap)
    verified = advanced.reconcile_updates(adjusted, config)
    new_gap = verified['scheduled_minus_recorded']
    return {'scheduled_learning_updates': verified['scheduled_learning_updates'],
            'recorded_learning_updates': summary['learning_updates'],
            'scheduled_minus_recorded': inherited_gap + new_gap,
            'inherited_unrecorded_scheduled_updates': inherited_gap,
            'new_phase_unrecorded_scheduled_updates': new_gap,
            'interrupted_update_completion_uncertain': bool(inherited_gap or new_gap),
            'explanation': ('Global scheduling continues from phase 1 without a second replay-fill delay. '
                            f'{inherited_gap} uncertain scheduled update(s) were inherited and {new_gap} occurred '
                            'at the new interruption boundary. Recorded counters are retained; no extra completed update is inferred.')}


def verify_model_selection(selection, summary, config, rows):
    declared = config['continuation']
    prior_selection, prior_summary = declared['phase1_selection'], declared['phase1_summary']
    advanced.verify_model_selection(prior_selection, prior_summary, declared['phase1_config'],
                                    rows[:prior_summary['completed_episodes']])
    prefix = prior_selection['candidate_evaluations']
    candidates = selection['candidate_evaluations']
    expected_prefix = [dict(candidate, origin_run_id=declared['source_run_id']) for candidate in prefix]
    require(candidates[:len(prefix)] == expected_prefix,
            'Phase-1 selection history changed beyond its explicit origin-run annotation')
    new_periodic = list(range((prior_summary['completed_episodes'] // config['validation_every_episodes'] + 1)
                             * config['validation_every_episodes'], summary['completed_episodes'] + 1,
                             config['validation_every_episodes']))
    appended = candidates[len(prefix):]
    require([candidate['additional_episodes'] for candidate in appended] == [*new_periodic, summary['completed_episodes']],
            'Missing or unexpected phase-2 selection candidates')
    require([candidate['reason'] for candidate in appended] == [*['periodic'] * len(new_periodic), 'final candidate'],
            'Wrong appended selection candidate reasons')
    require(selection['criterion'] == config['selection_metric'] and selection['seeds'] == config['validation_seeds'] and
            selection['eval_exploration'] == config['eval_exploration'] and
            selection['max_decisions_per_game'] == config['max_decisions_per_game'], 'Selection protocol changed')
    best_metric, winner = -math.inf, None
    inherited_gap = advanced.reconcile_updates(prior_summary, declared['phase1_config'])['scheduled_minus_recorded']
    for index, candidate in enumerate(candidates):
        verify_evaluation(candidate, config['validation_seeds'], config['max_decisions_per_game'], f'candidate {index}')
        spread = statistics.pstdev(candidate['scores'])
        metric = statistics.mean(candidate['scores']) - 0.5 * spread
        require(math.isclose(candidate['population_std'], spread, rel_tol=1e-12, abs_tol=1e-8) and
                math.isclose(candidate['selection_metric'], metric, rel_tol=1e-12, abs_tol=1e-8), 'Wrong candidate metric')
        if index >= len(prefix):
            if index == len(candidates) - 1:
                require(candidate['decisions'] == summary['total_decisions'] and candidate['updates'] == summary['learning_updates'],
                        'Final candidate differs from final global learner budget')
            else:
                require(candidate['decisions'] == int(rows[candidate['additional_episodes'] - 1]['total_steps']),
                        'Periodic selection candidate has wrong global decisions')
                scheduled = max(0, candidate['decisions'] // config['train_every_decisions']
                                - math.ceil(config['learning_starts_decisions'] / config['train_every_decisions']) + 1)
                require(candidate['updates'] == scheduled - inherited_gap, 'Periodic candidate updates do not continue global schedule')
        expected_selected = candidate['selection_metric'] > best_metric
        require(type(candidate['selected_when_evaluated']) is bool and candidate['selected_when_evaluated'] == expected_selected,
                'Best validation candidate or earlier tie retention was lost across phases')
        if expected_selected:
            winner, best_metric = candidate, candidate['selection_metric']
    require(winner is not None, 'No selected candidate')
    require(summary['selected_checkpoint_origin_run_id'] == winner['origin_run_id'],
            'Selected checkpoint origin does not match its winning candidate')
    require(selection['selected_additional_episodes'] == summary['selected_additional_episodes'] == winner['additional_episodes'] and
            selection['selected_updates'] == summary['selected_recorded_updates'] == winner['updates'] and
            selection['selected_metric'] == summary['selected_validation_metric'] == best_metric,
            'Global selected model metadata does not match the winning candidate')
    return winner


def verify_continuation_sources(config, summary, rows, manifest, run):
    actual = continuation_records(config, summary)
    source_run = Path(actual['source_run_path']).expanduser().resolve()
    require(source_run.name == actual['source_run_id'] and source_run.is_dir(), 'Missing or wrong phase-1 source directory')
    require(load_json(source_run / 'config.json') == actual['phase1_config'], 'Embedded phase-1 config changed')
    require(load_json(source_run / 'training_summary.json') == actual['phase1_summary'], 'Embedded phase-1 summary changed')
    require(load_json(source_run / 'model_selection.json') == actual['phase1_selection'], 'Embedded phase-1 selection record changed')
    with (source_run / 'training.csv').open(newline='') as handle:
        old_rows = list(csv.DictReader(handle))
    require(rows[:len(old_rows)] == old_rows, 'Phase-1 training CSV prefix changed')
    old_demos = load_json(source_run / 'demo_scores.json')
    current_demos = load_json(run / 'demo_scores.json')
    require(current_demos[:len(old_demos)] == old_demos, 'Phase-1 demonstration score prefix changed')
    imported_evidence = []
    for demo in old_demos:
        episode = demo['episode']
        for name in (f'episode_{episode:04d}.pt', f'demos/episode_{episode:04d}.gif'):
            require(sha256_file(run / name) == sha256_file(source_run / name),
                    f'Imported phase-1 evidence changed: {name}')
            imported_evidence.append(name)
    source_snapshot = source_run / 'learner_state.pt'
    snapshot_hash = sha256_file(source_snapshot)
    require(snapshot_hash == actual['source_snapshot_sha256'], 'Phase-1 learner snapshot SHA256 mismatch')
    bundle_path = Path(actual['resume_bundle_local_path']).expanduser().resolve()
    require(bundle_path.is_file() and sha256_file(bundle_path) == actual['resume_bundle_sha256'], 'Resume bundle SHA256 mismatch')
    members = actual['resume_bundle_members']
    snapshot_entries = [metadata for name, metadata in members.items()
                        if PurePosixPath(name).name == 'learner_state.pt']
    require(len(snapshot_entries) == 1 and snapshot_entries[0]['sha256'] == snapshot_hash and
            snapshot_entries[0]['size'] == source_snapshot.stat().st_size,
            'Resume bundle does not identify the verified phase-1 learner snapshot')
    with zipfile.ZipFile(bundle_path) as bundled:
        files = [info for info in bundled.infolist() if not info.is_dir()]
        require(len(files) == len(members) and {info.filename for info in files} == set(members), 'Resume bundle inventory mismatch')
        for info in files:
            safe_name = PurePosixPath(info.filename)
            require(not safe_name.is_absolute() and '..' not in safe_name.parts, 'Unsafe resume bundle path')
            expected = members[info.filename]
            require(info.file_size == expected['size'], f'Resume bundle member size mismatch: {info.filename}')
            with bundled.open(info) as handle:
                actual_hash = hashlib.file_digest(handle, 'sha256').hexdigest()
            require(actual_hash == expected['sha256'], f'Resume bundle member SHA256 mismatch: {info.filename}')
    require(sha256_file(ROOT / 'overnight_plan.md') == manifest['overnight_plan_sha256'] == manifest['plan_sha256_before_run'],
            'Overnight amendment differs from the frozen manifest')
    require(sha256_file(ROOT / 'experiment_plan.md') == manifest['inherited_original_plan_sha256'],
            'Original experiment plan changed during continuation')
    return {'run_id': None, **actual, 'combined_summary': summary,
            'source_snapshot_sha256_verified': True, 'resume_bundle_sha256_verified': True,
            'resume_bundle_member_crc_and_hashes_verified': True, 'phase1_csv_prefix_preserved': True,
            'phase1_selection_prefix_preserved': True, 'phase1_demo_prefix_preserved': True,
            'imported_phase1_gifs_and_checkpoints_byte_matched': imported_evidence,
            'large_source_snapshot_deserialized': False,
            'original_plan_sha256': manifest['inherited_original_plan_sha256'],
            'overnight_plan_sha256': manifest['overnight_plan_sha256']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True, help="Finished run directory, not its ZIP")
    parser.add_argument("--manifest", type=Path,
                        help="Pretraining source manifest; defaults to .execution/overnight_pretraining_manifest.json or its results/ copy")
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
    manifest_path = args.manifest.expanduser().resolve() if args.manifest else ROOT / ".execution/overnight_pretraining_manifest.json"
    if not manifest_path.is_file() and args.manifest is None:
        manifest_path = ROOT / "results/overnight_pretraining_manifest.json"
    require(manifest_path.is_file(), "Pretraining source manifest is missing")
    manifest = load_json(manifest_path)
    all_source_hashes = {str(index): digest(source(cell).encode()) for index, cell in enumerate(notebook["cells"])}
    require(all_source_hashes == manifest["cell_source_sha256"], "Notebook sources changed after the pretraining manifest")
    require(sha256_file(ROOT / "overnight_plan.md") == manifest["overnight_plan_sha256"] == manifest["plan_sha256_before_run"],
            "Overnight amendment differs from its frozen hash")
    require(sha256_file(ROOT / "experiment_plan.md") == manifest["inherited_original_plan_sha256"],
            "Original experiment plan differs from its inherited hash")
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
    continuation_provenance = verify_continuation_sources(config, summary, rows, manifest, run)
    continuation_provenance["run_id"] = run.name
    accounting = verify_parallel_history(rows, summary, config)
    update_reconciliation = reconcile_updates(summary, config)
    selection = load_json(run / "model_selection.json")
    winner = verify_model_selection(selection, summary, config, rows)
    prior_candidate_count = len(config["continuation"]["phase1_selection"]["candidate_evaluations"])
    require(all(candidate["origin_run_id"] == run.name
                for candidate in selection["candidate_evaluations"][prior_candidate_count:]),
            "A new selection candidate has the wrong continuation run origin")
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
    previous_last_learner = torch.load(Path(config["continuation"]["source_run_path"]).expanduser().resolve() / "last_trained.pt",
                                     map_location="cpu", weights_only=True)
    require(previous_last_learner["model"].keys() == keys and
            all(bool(torch.isfinite(value).all()) for value in previous_last_learner["model"].values()),
            "Invalid phase-1 final learner checkpoint")
    learner_changed_in_phase2 = any(not torch.equal(last_trained["model"][key], previous_last_learner["model"][key])
                                    for key in keys)
    require(learner_changed_in_phase2 or summary["continuation"]["phase2_recorded_updates"] == 0,
            "New updates recorded but phase-2 learner equals phase-1 final learner")
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
        prior_final = dict(config["continuation"]["phase1_selection"]["candidate_evaluations"][-1],
                           origin_run_id=config["continuation"]["source_run_id"])
        if winner == prior_final:
            candidate_checkpoint = previous_last_learner
        else:
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
    selected_index = next(index for index, candidate in enumerate(selection["candidate_evaluations"])
                          if candidate is winner)
    prior_candidate_count = len(config["continuation"]["phase1_selection"]["candidate_evaluations"])
    selected_origin = ("original inherited model" if selected_model["is_inherited_start"] else
                       "phase1" if selected_index < prior_candidate_count else "phase2")
    prior_summary = config["continuation"]["phase1_summary"]
    selected_model.update({
        "selected_origin": selected_origin,
        "retained_phase2_completed_episodes": max(0, winner["additional_episodes"] - prior_summary["completed_episodes"]),
        "retained_phase2_decisions": max(0, winner["decisions"] - prior_summary["total_decisions"]),
        "retained_phase2_recorded_updates": max(0, winner["updates"] - prior_summary["learning_updates"]),
    })
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
    report.update({
        "schema_version": "overnight_submission_v1", "continuation": summary["continuation"],
        "overnight_provenance": "results/overnight_provenance.json",
        "abandoned_partial_game_decisions": accounting["abandoned_game_decisions"],
        "abandoned_partial_game_count": accounting["abandoned_game_count"],
        "ended_before_episode_budget": summary["episodes_started"] < config["episodes_requested"],
        "episode_budget_kind": "total advanced game starts across both phases",
        "phase2_training_budget": {key: value for key, value in summary["continuation"].items() if key.startswith("phase2_")},
        "phase2_zero_learning_updates": summary["continuation"]["phase2_recorded_updates"] == 0,
        "last_learner_weights_changed_in_phase2": learner_changed_in_phase2,
        "elapsed_time_scope": "Combined phase-1 and phase-2 training timers. Phase 2 starts before restoring learner state and redisplaying prior demos. Final candidate validation, final snapshot/ZIP work, and classroom/held-out evaluations are excluded.",
    })
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
    provenance.update({"continuation": summary["continuation"],
                       "overnight_provenance": "results/overnight_provenance.json",
                       "pretraining_manifest": "results/overnight_pretraining_manifest.json",
                       "overnight_plan_matches_pretraining_manifest": True,
                       "original_plan_matches_inherited_manifest": True})
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
    verification.update({"continuation": summary["continuation"],
                         "source_snapshot_sha256_verified": True, "resume_bundle_sha256_verified": True,
                         "phase1_csv_and_selection_prefixes_preserved": True,
                         "phase1_periodic_evidence_preserved": True,
                         "last_learner_weights_changed_in_phase2": learner_changed_in_phase2,
                         "overnight_plan_matches_pretraining_manifest": True,
                         "original_plan_matches_inherited_manifest": True})
    # Verify JSON serializability and the unchanged input before touching published files.
    for value in (report, provenance, verification, continuation_provenance):
        json.dumps(value, allow_nan=False)
    require(sha256_file(notebook_path) == input_notebook_hash, "Notebook changed during verification; refusing to overwrite it")
    results = ROOT / "results"
    (results / "demos").mkdir(parents=True, exist_ok=True)
    for name in evidence_names:
        shutil.copy2(run / name, results / name)
    for manifest_name in ("pretraining_manifest.json", "overnight_pretraining_manifest.json"):
        if manifest_path.resolve() != (results / manifest_name).resolve():
            shutil.copy2(manifest_path, results / manifest_name)
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
    for name in ("heldout_validation.json", "heldout_validation.csv", "validation.json", "validation.csv"):
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
    write_json(results / "overnight_provenance.json", continuation_provenance)
    print(f"Verified {run.name}: {completed:,} completed additional games, "
          f"{summary['learning_updates']:,} recorded new updates, {len(expected_gifs)} GIFs.")
    print(f"Raw classroom mean: {before['mean']:g} untrained -> {after['mean']:g} final.")
    print(f"Full checkpoint ZIP retained locally: {archive}")


if __name__ == "__main__":
    main()
