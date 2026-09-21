#!/usr/bin/env python3
"""Generate the overnight continuation README from finished, verified evidence.

Run after the overnight submission builder and after copying the completed additional
evaluation's validation.json and validation.csv into results/. This generator
does not train, evaluate, change notebook outputs, or modify evidence. It leaves
an explicit VISUAL_REVIEW_REQUIRED passage for a human/assistant who has watched
the actual GIFs to replace before publication.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import statistics


ROOT = Path(__file__).resolve().parents[1]
REPOSITORY = "https://github.com/sbardacosta-code/class-3-pacman-dqn"
CLASS_SEEDS = [101, 202, 303, 404, 505]
SELECTION_SEEDS = list(range(20001, 20011))
ADDITIONAL_SEEDS = list(range(30001, 30021))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def load_json(path):
    return json.loads(path.read_text())


def number(value, places=1):
    value = float(value)
    require(math.isfinite(value), "Cannot report a non-finite number")
    return f"{int(value):,}" if value.is_integer() else f"{value:,.{places}f}"


def signed(value):
    return f"{float(value):+,.1f}"


def duration(seconds):
    seconds = float(seconds)
    hours = int(seconds // 3600)
    minutes = int(seconds % 3600 // 60)
    remaining = seconds % 60
    return f"{hours} h {minutes} min {remaining:.1f} sec ({seconds:,.3f} seconds)"


def score_list(values):
    return ", ".join(str(int(value)) if float(value).is_integer() else str(value)
                     for value in values)


def table(headers, rows):
    lines = ["| " + " | ".join(map(str, headers)) + " |",
             "| " + " | ".join("---" for _ in headers) + " |"]
    lines.extend("| " + " | ".join(map(str, row)) + " |" for row in rows)
    return "\n".join(lines)


def check_scores(record, seeds, label):
    require(record["seeds"] == seeds and len(record["scores"]) == len(seeds),
            f"Wrong or incomplete {label} evaluation")
    scores = [float(value) for value in record["scores"]]
    require(all(math.isfinite(value) for value in scores), f"Non-finite {label} score")
    require(math.isclose(record["mean"], statistics.mean(scores), abs_tol=1e-9),
            f"Incorrect {label} mean")


def check_checkpoint_provenance(validation, report, config):
    selected = validation["models"]["selected"]
    inherited = validation["models"]["inherited"]
    require(selected["run_id"] == report["run_id"] and
            selected["checkpoint_sha256"] == report["selected_model"]["checkpoint_sha256"],
            "Additional evaluation does not match the verified selected checkpoint")
    require(inherited["run_id"] == config["warm_start"]["run_id"] and
            inherited["checkpoint_sha256"] == config["warm_start"]["sha256"],
            "Additional evaluation does not match the inherited source checkpoint")


def training_status_text(status, completed, requested, started=None):
    started = completed if started is None else started
    counts = (f"{number(started)} of {number(requested)} allowed advanced games were started; "
              f"{number(completed)} completed. Abandoned phase-one games are not counted as completed.")
    if status == "completed":
        return "The episode-start budget finished with no active final games. " + counts
    if status == "time_budget":
        return "The authorized training cutoff stopped the run; its recorded status is `time_budget`. " + counts
    if status == "interrupted":
        return "The run was interrupted; its recorded status remains `interrupted`. " + counts
    return ("Training recorded a failure; any saved model and evaluations do not make "
            "that training run completed.")


def check_overnight_provenance(evidence):
    config = evidence["config"]
    summary = evidence["training_summary"]
    continuation = summary["continuation"]
    recorded = evidence["overnight_provenance"]
    require(recorded["run_id"] == evidence["report_data"]["run_id"],
            "Overnight provenance belongs to another run")
    require(all(recorded.get(key) == value for key, value in continuation.items()),
            "Published overnight provenance differs from the final summary")
    phase1 = continuation["phase1_summary"]
    for key in ("source_run_id", "source_run_path", "source_snapshot_sha256", "phase1_summary",
                "resume_bundle_url", "resume_bundle_sha256", "phase2_time_limit_seconds",
                "preparation_utc", "deadline_utc", "total_advanced_started_game_budget"):
        require(continuation[key] == config["continuation"][key],
                f"Continuation configuration and summary disagree: {key}")
    require(continuation["total_advanced_started_game_budget"] == config["episodes_requested"] == 6000,
            "Wrong combined episode-start budget")
    require(continuation["emulator_states_restored"] is False and
            continuation["optimizer_and_replay_restored"] is True,
            "Unexpected restoration semantics")
    require(continuation["abandoned_partial_games"] == phase1["partial_games"],
            "Abandoned games differ from the phase-one partial-game record")
    require(type(continuation["discarded_pending_transitions"]) is int and
            continuation["discarded_pending_transitions"] >= 0,
            "Missing actual count of discarded pending n-step items")
    for total_key, delta_key in (("completed_episodes", "phase2_completed_episodes"),
                                 ("episodes_started", "phase2_episodes_started"),
                                 ("total_decisions", "phase2_decisions"),
                                 ("learning_updates", "phase2_recorded_updates")):
        difference = summary[total_key] - phase1[total_key]
        require(difference >= 0 and difference == continuation[delta_key],
                f"Phase-two delta does not reconcile: {delta_key}")
    require(math.isclose(summary["elapsed_seconds_including_periodic_demos"],
                         phase1["elapsed_seconds_including_periodic_demos"] +
                         continuation["phase2_elapsed_seconds"], abs_tol=1e-6),
            "Combined timer does not equal the two recorded training periods")
    completed_steps = sum(int(row["steps"]) for row in evidence["training_rows"])
    abandoned_steps = sum(game["decisions"] for game in continuation["abandoned_partial_games"])
    active_steps = sum(game["decisions"] for game in summary["partial_games"])
    require(completed_steps + abandoned_steps + active_steps == summary["total_decisions"],
            "Completed, abandoned, and active-game decisions do not reconcile")
    require(summary["episodes_started"] == summary["completed_episodes"] +
            len(continuation["abandoned_partial_games"]) + len(summary["partial_games"]),
            "Started/completed/abandoned/active game counts disagree")
    old_candidates = continuation["phase1_selection"]["candidate_evaluations"]
    inherited_candidates = [dict(row, origin_run_id=continuation["source_run_id"])
                            for row in old_candidates]
    require(evidence["model_selection"]["candidate_evaluations"][:len(old_candidates)] == inherited_candidates,
            "Phase-one model-selection history was not preserved")


def read_evidence(root):
    results = root / "results"
    names = ("config", "training_summary", "comparison", "warm_start_evaluation",
             "model_selection", "demo_scores", "verification", "report_data", "validation",
             "overnight_provenance")
    evidence = {name: load_json(results / f"{name}.json") for name in names}
    with (results / "training.csv").open(newline="") as handle:
        evidence["training_rows"] = list(csv.DictReader(handle))
    with (results / "validation.csv").open(newline="") as handle:
        evidence["validation_rows"] = list(csv.DictReader(handle))
    config, summary = evidence["config"], evidence["training_summary"]
    report, verified = evidence["report_data"], evidence["verification"]
    comparison, warm = evidence["comparison"], evidence["warm_start_evaluation"]
    require(report["run_id"] == verified["run_id"], "Mixed-run verification and report metadata")
    require(report["config"] == config and report["summary"] == summary and
            report["evaluation"] == comparison, "Raw evidence differs from builder metadata")
    require(verified["executed_code_cells"] == 28 and verified["notebook_error_outputs"] == 0,
            "The final notebook must contain all 28 executed code cells without error outputs")
    require(config["exploration"] == 0.10 and config["episodes_requested"] == 6000 and
            config["learning_rate"] == 0.00005, "Unexpected three overnight settings")
    require(config["n_step"] == 3 and config["replay_capacity"] == 50000 and
            config["training_death_penalty"] == -0.5 and config["num_training_envs"] == 4,
            "This generator describes the approved prioritized n-step experiment")
    require(config["eval_seeds"] == CLASS_SEEDS and config["eval_exploration"] == 0.05 and
            config["max_decisions_per_game"] == 3000, "Classroom evaluation settings changed")
    require(comparison["warm_start"] == warm, "Warm-start evaluation records disagree")
    for label, value in (("baseline", comparison["before"]), ("selected", comparison["after"]),
                         ("warm start", warm)):
        check_scores(value, CLASS_SEEDS, label)
    selection = evidence["model_selection"]
    require(selection["seeds"] == SELECTION_SEEDS and selection["eval_exploration"] == 0.05 and
            selection["max_decisions_per_game"] == 3000, "Wrong checkpoint-selection protocol")
    candidates = selection["candidate_evaluations"]
    require(candidates, "Missing selection candidates")
    for candidate in candidates:
        check_scores(candidate, SELECTION_SEEDS, "selection candidate")
        expected = candidate["mean"] - 0.5 * statistics.pstdev(candidate["scores"])
        require(math.isclose(candidate["selection_metric"], expected, abs_tol=1e-8),
                "Incorrect selection metric")
    selected = max(candidates, key=lambda value: value["selection_metric"])
    require(selected["additional_episodes"] == selection["selected_additional_episodes"] ==
            summary["selected_additional_episodes"] and
            selected["updates"] == selection["selected_updates"] == summary["selected_recorded_updates"],
            "Selected checkpoint budgets disagree")
    evidence["selected_candidate"] = selected
    validation = evidence["validation"]
    settings = validation["settings"]
    require(settings["seeds"] == ADDITIONAL_SEEDS and settings["exploration"] == 0.05 and
            settings["max_decisions"] == 3000, "Wrong additional-evaluation protocol")
    require(settings["learning_updates"] == 0, "Additional evaluation must not learn")
    check_checkpoint_provenance(validation, report, config)
    for label in ("inherited", "selected"):
        model = validation["models"][label]
        check_scores(model, ADDITIONAL_SEEDS, label)
        hits = sum(value >= 3000 for value in model["scores"])
        require(model["count_at_least_3000"] == hits and
                math.isclose(model["fraction_at_least_3000"], hits / 20), "Wrong success count")
        checks = model["numerical_checks"]
        require(checks.get("learning_updates") == 0 and
                all(value is True for key, value in checks.items() if key != "learning_updates"),
                "Additional evaluation numerical check failed")
    require(len(evidence["validation_rows"]) == 40, "Additional-evaluation CSV needs 40 scores")
    for label in ("inherited", "selected"):
        model = validation["models"][label]
        rows = [row for row in evidence["validation_rows"] if row["model"] == label]
        require([int(row["seed"]) for row in rows] == model["seeds"] and
                [float(row["score"]) for row in rows] == model["scores"],
                "Additional-evaluation JSON and CSV disagree")
    require(len(evidence["training_rows"]) == summary["completed_episodes"],
            "Training CSV and completed episode count disagree")
    require(len(evidence["demo_scores"]) + 2 == report["gif_count"], "Incomplete gameplay evidence")
    for path in report["gifs"]:
        require((root / path).is_file(), f"Missing gameplay image {path}")
    check_overnight_provenance(evidence)
    return evidence


def render_report(evidence):
    cfg, summary = evidence["config"], evidence["training_summary"]
    report, verified = evidence["report_data"], evidence["verification"]
    comp, warm = evidence["comparison"], evidence["warm_start_evaluation"]
    before, after = comp["before"], comp["after"]
    selected = evidence["selected_candidate"]
    selection = evidence["model_selection"]
    validation = evidence["validation"]
    extra_before = validation["models"]["inherited"]
    extra_after = validation["models"]["selected"]
    inherited = cfg["warm_start"]
    continuation = summary["continuation"]
    phase1 = continuation["phase1_summary"]
    abandoned = continuation["abandoned_partial_games"]
    completed = summary["completed_episodes"]
    initial_won = selected["additional_episodes"] == 0 and selected["updates"] == 0
    class_delta = after["mean"] - warm["mean"]
    extra_delta = extra_after["mean"] - extra_before["mean"]
    target_met = after["mean"] >= 3000 and extra_after["count_at_least_3000"] >= 18
    selected_index = next(i for i, candidate in enumerate(selection["candidate_evaluations"])
                          if candidate is selected)
    beta = cfg["priority_beta_start"] + (1 - cfg["priority_beta_start"]) * min(
        1, summary["total_decisions"] / cfg["priority_beta_decisions"])
    best_index = max(range(5), key=lambda i: after["scores"][i])
    lines = ["# Class 3: Ms. Pac-Man — overnight continuation of prioritized, three-step Double DQN", "",
             f"The validation-selected agent scored **{number(after['mean'])} mean raw points** on the "
             f"five unchanged classroom games. Its inherited starting model scored **{number(warm['mean'])}** "
             f"on the same games; the fresh untrained network scored **{number(before['mean'])}**. "
             f"The combined advanced attempt completed **{number(completed)} games across two phases**, "
             f"with a **{number(cfg['episodes_requested'])}-game start limit**, and recorded "
             f"**{number(summary['learning_updates'])} learning updates beyond the inherited model**.", "",
             f"The selected policy reached at least 3,000 points in **{extra_after['count_at_least_3000']}/20** "
             f"additional games, with mean **{number(extra_after['mean'])}**. The predeclared target "
             f"(classroom mean ≥3,000 **and** at least 18/20 additional games ≥3,000) was "
             f"**{'met on these tests' if target_met else 'not met'}**. These finite evaluations do not "
             "guarantee future game scores.", ""]
    if initial_won:
        lines += ["**The inherited starting model won checkpoint selection.** The selected agent therefore "
                  "contains **zero updates from this new attempt**, even though later candidates were trained. "
                  "The full attempted budget remains reported below; additional training is not presented as "
                  "a successful improvement.", ""]
    elif class_delta < 0:
        lines += [f"**Classroom regression:** the selected policy scored {number(-class_delta)} fewer mean "
                  "points than the inherited policy. Selection used separate validation games, so a validation "
                  "winner can still perform worse on the classroom games.", ""]
    elif class_delta == 0:
        lines += ["The selected policy's classroom mean was unchanged from the inherited model. This does "
                  "not establish a performance gain from the new training.", ""]
    else:
        lines += [f"The observed classroom gain over the inherited model was **{signed(class_delta)} points**. "
                  "Several changes were introduced together, so this result cannot isolate which change helped.", ""]
    lines += ["The [executed notebook](pacman_dqn.ipynb) contains all **28 code cells executed in order**, "
              "with their saved outputs. The supplied project is [pepealonso95/pacman-dqn]"
              "(https://github.com/pepealonso95/pacman-dqn); this experiment's extensions are disclosed below.", "",
              "## Choices and expectation recorded before training", "",
              table(["Setting", "Choice", "Reason"], [
                  ["Training exploration", "0.10", "Try alternatives around the inherited policy; constant after random warm-up."],
                  ["Total advanced episode-start budget", "6,000", "Continue the same learner within the newly authorized overnight window; includes both phases."],
                  ["Learning rate", "0.00005", "Use smaller updates while fine-tuning with prioritized, longer-horizon targets."],
              ]), "",
              "The original [pre-training plan](experiment_plan.md) expected more efficient learning, safer "
              "choices, and higher typical scores. During its unchanged first phase, the student explicitly "
              "authorized approximately **six more hours overnight**, superseding the former three-hour "
              "overall limit. The frozen [overnight amendment](overnight_plan.md) increases the total "
              "advanced episode-start budget from 2,500 to 6,000 while retaining exploration 0.10, "
              "learning rate 0.00005, and the same learning/evaluation methods.", "",
              "The extension was authorized at approximately **08:07 UTC on September 21, 2026**. "
              "Its planned training cutoff is **13:37 UTC**, reserving approximately 30 minutes for "
              "evaluation, saving, and publication before **14:07 UTC**. These are the authorized "
              "schedule boundaries; actual phase timing is reported below.", "",
              "The student authorized the extension **before** the first 250-game development result "
              "arrived. That later candidate averaged approximately 1,194 raw points versus approximately "
              "1,199 for the inherited model on the ten development seeds. Those are means, not the "
              "mean-minus-variability selection metric. This was flat development performance. The "
              "longer budget came from the student's authorization, not a change to the classroom test "
              "or a claim that more training guarantees improvement.", "",
              "The network starts from the previous scaled-reward model, which had already received "
              f"**{number(inherited['completed_episodes'])} completed games, {number(inherited['decisions'])} "
              f"decisions, and {number(inherited['recorded_updates'])} recorded updates**. Its optimizer and "
              "replay were reset only at the beginning of phase one. Phase two restores the actual last "
              "phase-one learner, optimizer, target network, replay, and random states, rather than "
              "reinitializing them or restarting from the best selected checkpoint. The source run's separately "
              "documented interruption uncertainty remains in its archived evidence; inherited updates here "
              "are the recorded count.", "",
              "## Disclosed training extensions", "",
              table(["Extension", "Setting and purpose"], [
                  ["Three-step returns", "n=3, gamma=0.99; use up to three rewards and the actual future four-frame stack."],
                  ["Prioritized replay", "50,000 transitions; alpha=0.5. Sample surprising transitions more often and apply importance correction."],
                  ["Importance correction", f"Beta begins at 0.4 and approaches 1.0 over 2,000,000 new decisions; reached {beta:.4f} in this attempt."],
                  ["Training reward", "raw game points × 0.01 − 0.5 × lives lost; no clipping. Losing a life costs the equivalent of 50 points during training."],
                  ["Replay collection", "First 1,000 new decisions are random; collect 10,000 decisions before learning. Exploration then stays at 10%."],
                  ["Four training games", "Batch action predictions across four environments. One shared learner update per four aggregate decisions; batch size 32."],
                  ["Target network", "Double DQN; synchronize the target every 1,000 new decisions. Network architecture is unchanged."],
                  ["Episode boundaries", "Life loss does not terminate the game or TD target. True game over ends bootstrap; a time limit retains it."],
              ]), "",
              "Four environments improve batching; they do not multiply the reported decision or update "
              "counts. Training seeds continue beyond the source run using `42 + 1069 + additional_game_id`. "
              "Phase two retains the global decision, update, and game-start counters, so random warm-up, "
              "replay filling, beta annealing, target synchronization, and validation milestones do not restart. "
              "The replay stores full current and future image stacks, approximately 2.63 GiB at capacity. "
              "Neither RAM observations nor ghost coordinates are supplied to the policy.", "",
              "The design draws on [Prioritized Experience Replay](https://arxiv.org/abs/1511.05952), "
              "[Rainbow](https://arxiv.org/abs/1710.02298), and [Revisiting the Arcade Learning Environment]"
              "(https://arxiv.org/abs/1709.06009). This is a small adaptation, not a reproduction of their "
              "large-budget results. Bundling replay, return length, reward shaping, and batching prevents "
              "an isolated causal claim about any one change.", "",
              "## Actual training budget and selected model", ""]
    status = summary["status"]
    status_text = training_status_text(status, completed, cfg["episodes_requested"], summary["episodes_started"])
    lines += [f"**Status: `{status}`.** {status_text}", "",
              table(["Budget", "Completed games", "Decisions", "Recorded learning updates"], [
                  ["Inherited training", number(inherited["completed_episodes"]), number(inherited["decisions"]), number(inherited["recorded_updates"])],
                  ["Combined advanced attempt: phases one + two", number(completed), number(summary["total_decisions"]), number(summary["learning_updates"])],
                  ["Original inheritance + combined advanced attempt", number(inherited["completed_episodes"] + completed), number(inherited["decisions"] + summary["total_decisions"]), number(inherited["recorded_updates"] + summary["learning_updates"])],
                  ["New work retained in selected candidate", number(selected["additional_episodes"]), number(selected["decisions"]), number(selected["updates"])],
                  ["Inherited + selected candidate", number(inherited["completed_episodes"] + selected["additional_episodes"]), number(inherited["decisions"] + selected["decisions"]), number(inherited["recorded_updates"] + selected["updates"])],
              ]), "",
              "The combined attempt measures compute spent across both phases without counting carried "
              "counters twice; the selected-candidate rows measure training "
              "retained in the submitted policy. Candidate snapshots can include decisions from parallel "
              "games that had not yet completed. The selected checkpoint can precede the end of the attempt.", ""]
    lines += ["### Two phases and restoration provenance", "",
              table(["Phase", "Games started", "Games completed", "Decisions", "Recorded updates", "Recorded elapsed time"], [
                  [f"Phase one: `{continuation['source_run_id']}`", number(phase1["episodes_started"]),
                   number(phase1["completed_episodes"]), number(phase1["total_decisions"]),
                   number(phase1["learning_updates"]), duration(phase1["elapsed_seconds_including_periodic_demos"])],
                  [f"Phase two only: `{report['run_id']}`", number(continuation["phase2_episodes_started"]),
                   number(continuation["phase2_completed_episodes"]), number(continuation["phase2_decisions"]),
                   number(continuation["phase2_recorded_updates"]), duration(continuation["phase2_elapsed_seconds"])],
                  ["Combined advanced work", number(summary["episodes_started"]), number(completed),
                   number(summary["total_decisions"]), number(summary["learning_updates"]),
                   duration(summary["elapsed_seconds_including_periodic_demos"])],
              ]), "",
              f"Phase one ended with status **`{phase1['status']}`**; phase two/final status is "
              f"**`{summary['status']}`**. The phase-two row contains differences from the carried "
              "phase-one counters, so restoration itself creates no extra episodes or learning updates. "
              "The original 1,069-game model remains separate inheritance above.", "",
              f"Phase-two preparation was recorded at **`{continuation['preparation_utc']}`**. "
              f"The supervisor's training deadline was **`{continuation['deadline_utc']}`**, with "
              f"a relative phase-two allowance of **{duration(continuation['phase2_time_limit_seconds'])}** "
              f"and a **{number(continuation['startup_reserve_seconds'])}-second startup reserve**. "
              "Preparation time and the authorization-to-finalization window are not additional training "
              "periods to add to the table. Phase one began before the six-more-hours authorization.", "",
              "Phase two restored the **last learner** and its Adam optimizer, target, replay, priorities, "
              "and random states. The incumbent selected checkpoint and full ten-seed selection history "
              "were carried separately. The final policy may therefore be inherited, a phase-one candidate, "
              "or a phase-two candidate; selection never forces the latest weights to win.", "",
              f"Because emulator states were unavailable, **{len(abandoned)} phase-one partial games** "
              f"were abandoned and **{number(continuation['discarded_pending_transitions'])} pending "
              "n-step items** were cleared. Already emitted replay transitions remain. Fresh games begin "
              "after the previous `episodes_started` counter, avoiding seed reuse. This restores learning "
              "state, but it is **not exact mid-game resumption**.", "",
              "Source evidence: [phase-one executed notebook](experiments/overnight_phase1/pacman_dqn.ipynb), "
              "[phase-one results](experiments/overnight_phase1/results/), and "
              "[overnight_provenance.json](results/overnight_provenance.json).", ""]
    if continuation["phase2_recorded_updates"] == 0:
        lines += ["**Phase two recorded no additional learning updates.** Any advanced learning in the "
                  "combined budget happened in phase one; restoring its saved state does not count as "
                  "new training.", ""]
    hardware = validation.get("hardware", {})
    chip = hardware.get("cpu_model", hardware.get("processor", "not recorded"))
    memory = hardware.get("physical_memory_bytes")
    hardware_label = chip + (f"; {memory / 2**30:g} GiB memory" if memory else "")
    lines += [table(["Measure", "Recorded value"], [
        ["Run ID", f"`{report['run_id']}`"],
        ["Additional games started / completed", f"{number(summary['episodes_started'])} / {number(completed)}"],
        ["Combined recorded training periods", duration(summary["elapsed_seconds_including_periodic_demos"])],
        ["Combined monotonic elapsed time", duration(summary["active_monotonic_elapsed_seconds"])],
        ["Phase-two monotonic elapsed time", duration(continuation["phase2_active_monotonic_elapsed_seconds"])],
        ["Final summary's recorded training cap", duration(summary["training_time_limit_seconds"])],
        ["Training device", str(cfg["device"]).upper()],
        ["Host hardware recorded by additional evaluation", hardware_label],
        ["Python / PyTorch", f"{cfg['python']} / {cfg['packages']['torch']}"],
        ["Platform", cfg["platform"]],
        ["Replay entries at stop", number(summary["replay_size"])],
        ["Lives lost in new training", number(summary["training_lives_lost"])],
    ]), "",
              "Recorded training periods include collection, updates, periodic GIF/checkpoint work, and "
              "selection evaluations performed inside training. The phase-two timer also includes restoration "
              "and display of copied phase-one GIFs because it starts at the top of the continuation cell. "
              "They exclude each phase's final selection evaluation, final snapshot/ZIP saving, and final "
              "classroom/additional evaluations. Fresh/inherited classroom evaluation before training and "
              "external preparation/finalization are also separate. The six-hour authorization is an overall "
              "wall-clock allowance, not a claim that six hours of optimizer work occurred. "
              "Package details: [config.json](results/config.json) "
              "and [environment.txt](results/environment.txt).", ""]
    partial = summary["partial_games"]
    partial_decisions = sum(game["decisions"] for game in partial)
    abandoned_decisions = sum(game["decisions"] for game in abandoned)
    completed_decisions = sum(int(row["steps"]) for row in evidence["training_rows"])
    lines += [f"Parallel-game accounting: **{number(completed_decisions)} completed-game decisions + "
              f"{number(abandoned_decisions)} decisions in {len(abandoned)} abandoned phase-one games + "
              f"{number(partial_decisions)} decisions in {len(partial)} final active games = "
              f"{number(summary['total_decisions'])} new decisions**. Partial-game scores are not inserted "
              "as completed training CSV rows.", ""]
    if abandoned or partial:
        lines += [table(["State", "Slot", "Training game ID", "Seed", "Partial decisions", "Raw score", "Lives lost"],
                        [[label, game["slot"], game["training_episode_id"], game["seed"], game["decisions"],
                          number(game["score"]), game["lives_lost"]]
                         for label, games in (("Abandoned after phase one", abandoned), ("Active at final stop", partial))
                         for game in games]), ""]
    reconciliation = report.get("update_count_reconciliation", verified["update_count_reconciliation"])
    lines += [f"Update accounting: **{number(reconciliation['recorded_learning_updates'])} recorded** versus "
              f"**{number(reconciliation['scheduled_learning_updates'])} scheduled**. "
              + reconciliation["explanation"], ""]
    if summary["learning_updates"] == 0:
        lines += ["**No new learning updates occurred.** This run still inherits a trained policy; it is "
                  "not equivalent to the fresh untrained baseline.", ""]
    lines += ["Budget evidence: [training_summary.json](results/training_summary.json), "
              "[training.csv](results/training.csv), and [verification.json](results/verification.json).", "",
              "## All five classroom games", "",
              "These scores use the same seeds, **5% exploration**, **3,000-decision limit**, preprocessing, "
              "and sticky-action settings before and after training. Evaluations do not learn. The fresh "
              "baseline is an untrained network, not a random-action agent. The inherited-model column "
              "separates previous learning from this attempt.", "",
              table(["Game", "Seed", "Fresh untrained", "Inherited model", "Selected model", "Selected − inherited"],
                    [[i + 1, seed, number(before["scores"][i]), number(warm["scores"][i]),
                      number(after["scores"][i]), signed(after["scores"][i] - warm["scores"][i])]
                     for i, seed in enumerate(CLASS_SEEDS)] + [
                        ["**Mean**", "", f"**{number(before['mean'])}**", f"**{number(warm['mean'])}**",
                         f"**{number(after['mean'])}**", f"**{signed(class_delta)}**"]]), "",
              f"The class leaderboard value for this selected model is **{number(after['mean'])}**. "
              f"Time-limited games: {sum(before['time_limited'])}/5 fresh, {sum(warm['time_limited'])}/5 "
              f"inherited, {sum(after['time_limited'])}/5 selected. All points are raw game points, "
              "not training rewards. Machine-readable results: [comparison.json](results/comparison.json) "
              "and [warm_start_evaluation.json](results/warm_start_evaluation.json).", "",
              "The earlier [clipped-reward 2,000-game experiment](experiments/clipped_2000/README.md) "
              f"scored **1,582** on these five seeds; this submission differs by **{signed(after['mean'] - 1582)}** "
              "points. That earlier model is not the warm-start source used here.", "",
              "## Checkpoint selection on separate validation games", "",
              "Candidates were evaluated on seeds **20001–20010** before new learning, every 250 completed "
              "advanced games, and at each phase's end. Phase-one history and the incumbent persist into "
              "phase two. The predeclared selection metric is **mean raw score − "
              "0.5 × population standard deviation**. Higher wins; ties retain the earlier candidate. "
              "This variability penalty is a heuristic, not a confidence bound. Classroom scores and "
              "the later 20-game evaluation did not select the checkpoint.", ""]
    candidate_rows = []
    for i, candidate in enumerate(selection["candidate_evaluations"]):
        candidate_rows.append([
            candidate["additional_episodes"], number(candidate["decisions"]), number(candidate["updates"]),
            score_list(candidate["scores"]), number(candidate["mean"]), number(candidate["population_std"]),
            number(candidate["selection_metric"]), "**Selected**" if i == selected_index else candidate["reason"],
        ])
    lines += ["The ten-score column is ordered by seeds 20001, 20002, …, 20010.", "",
              table(["Additional games", "Decisions", "Updates", "All ten scores", "Mean", "Population SD", "Metric", "Outcome / reason"], candidate_rows), "",
              f"Selected: **{number(selected['additional_episodes'])} additional completed games**, "
              f"**{number(selected['updates'])} additional recorded updates**, metric "
              f"**{number(selected['selection_metric'])}**. Full record: "
              "[model_selection.json](results/model_selection.json). The saved `trained.pt` is this "
              "selected model; `last_trained.pt` preserves the final attempted learner separately.", "",
              "## Twenty additional games: consistency check", "",
              "After checkpoint selection, the inherited and selected models were each evaluated on "
              "seeds **30001–30020**, disjoint from training, selection, and classroom seeds. They use "
              "the unchanged 5% exploration and 3,000-decision cap. No weight updates occurred. These "
              "40 scores do not replace the five classroom scores.", "",
              table(["Measure", "Inherited", "Selected"], [
                  ["Mean", number(extra_before["mean"]), number(extra_after["mean"])],
                  ["Median", number(extra_before["median"]), number(extra_after["median"])],
                  ["Minimum", number(extra_before["min"]), number(extra_after["min"])],
                  ["Maximum", number(extra_before["max"]), number(extra_after["max"])],
                  ["Games ≥3,000", f"{extra_before['count_at_least_3000']}/20", f"{extra_after['count_at_least_3000']}/20"],
                  ["Fraction ≥3,000", f"{extra_before['fraction_at_least_3000']:.0%}", f"{extra_after['fraction_at_least_3000']:.0%}"],
                  ["Time-limited games", f"{sum(extra_before['time_limited'])}/20", f"{sum(extra_after['time_limited'])}/20"],
              ]), "",
              table(["Seed", "Inherited raw score", "Selected raw score", "Change"],
                    [[seed, number(old), number(new), signed(new - old)]
                     for seed, old, new in zip(ADDITIONAL_SEEDS, extra_before["scores"], extra_after["scores"])])]
    distribution = []
    for label, lower, upper in (("Below 1,000", -math.inf, 1000), ("1,000–1,999", 1000, 2000),
                                ("2,000–2,999", 2000, 3000), ("3,000 or more", 3000, math.inf)):
        distribution.append([label,
                             sum(lower <= value < upper for value in extra_before["scores"]),
                             sum(lower <= value < upper for value in extra_after["scores"])])
    lines += ["", table(["Score range", "Inherited game count", "Selected game count"], distribution), "",
              f"The additional-game mean changed by **{signed(extra_delta)} points**. "
              f"Operational target: **{'met in these evaluations' if target_met else 'not met'}**. "
              "A high maximum alone does not establish consistency. Data and unchanged-weight checks: "
              "[validation.json](results/validation.json), [validation.csv](results/validation.csv).", "",
              "## Gameplay evidence", "",
              "Each GIF shows at most the first 20 seconds of game time, approximately 4× playback, "
              "and plays twice. Full-game scores below can therefore exceed the score visible in the excerpt. "
              "The final GIF is the best of the selected model's five classroom games, not an average game "
              "or a model chosen using those scores.", "",
              f"**Fresh untrained network — seed 101, full-game score {number(before['scores'][0])}.**", "",
              "![Fresh untrained gameplay](results/demos/episode_0000.gif)", "",
              f"**Selected model — best classroom game, seed {after['seeds'][best_index]}, full-game "
              f"score {number(after['scores'][best_index])}.** This model retains "
              f"{number(selected['additional_episodes'])} additional completed games of training.", "",
              "![Best selected gameplay](results/demos/final_best.gif)", "",
              gameplay_observations(report["run_id"]), "",
              "The periodic GIFs below span both phases and show the current learner every 25 completed "
              "advanced games, using the preserved global completion counter. "
              "They all use seed 101 and are demonstrations only. They may occur after the eventually "
              "selected checkpoint; those later GIFs do not depict the submitted policy. Full scores: "
              "[demo_scores.json](results/demo_scores.json).", "",
              "<details>", "<summary>Expand all intermediate gameplay GIFs (every 25 completed games)</summary>", ""]
    for demo in evidence["demo_scores"]:
        episode = demo["episode"]
        lines += [f"**After {number(episode)} additional games — seed 101, score {number(demo['scores'][0])}.**", "",
                  f"![After {episode} additional games](results/demos/episode_{episode:04d}.gif)", ""]
    lines += ["</details>", "", "## Training curves and what the policy learned", "",
              "![Raw score, shared update loss, and exploration](results/training_dashboard.png)", "",
              "The dashboard shows raw training scores and the rolling average of up to 25 games, mean "
              "learning loss, and exploration. The green dashed marker identifies the validation-selected "
              "checkpoint. Each CSV row is a completed game, ordered by completion across four environments. "
              "Its `mean_loss` averages **shared learner updates during that game's lifetime**; overlapping "
              "games can include the same updates, so it is not loss attributable only to that game. "
              "Exploration records the value when the game completed, which can hide the early random "
              "warm-up within the first few games.", ""]
    rows = evidence["training_rows"]
    if rows:
        blocks = []
        for start in range(0, len(rows), 250):
            block = rows[start:start + 250]
            blocks.append([f"{start + 1}–{start + len(block)}",
                           number(statistics.mean(float(row["score"]) for row in block)),
                           number(statistics.mean(int(row["steps"]) for row in block))])
        first_block = rows[:250]
        last_block = rows[-250:]
        first_mean = statistics.mean(float(row["score"]) for row in first_block)
        last_mean = statistics.mean(float(row["score"]) for row in last_block)
        lines += [table(["Completed additional games", "Mean raw training score", "Mean decisions per game"], blocks), "",
                  f"The first {len(first_block)} completed training games averaged **{number(first_mean)}** "
                  f"points; the last {len(last_block)} averaged **{number(last_mean)}**.", ""]
        if report["run_id"] == "20260921_025203_850824":
            lines += ["The score curve "
                  "remains noisy and the selected checkpoint precedes the final learner. The loss curve "
                  "falls substantially, while typical scores level off well below 3,000. This supports "
                  "improved scoring behavior, but not steadily improving or consistently strong play.", ""]
    lines += ["Training uses different seeds and 10% exploration; its scores are not substitutes for "
              "the fixed evaluations. Priorities, multi-step targets, and importance weighting change loss "
              "magnitudes. Lower loss does not guarantee better play, and loss values are not directly "
              "comparable with older uniform-replay experiments.", ""]
    if initial_won:
        learned = "The selected policy retains the inherited network's behavior. New training changed candidate "
        learned += "weights, but the selection evidence did not justify replacing the initial policy. No new "
        learned += "strategy is credited to the submitted model from this attempt."
        if summary["learning_updates"] == 0:
            learned = "The selected policy retains inherited learning, and no new weight updates were recorded."
    else:
        learned = f"The submitted checkpoint includes {number(selected['updates'])} new recorded updates to its "
        learned += "predicted action values. Its score and consistency comparisons above show the observable "
        learned += "outcome; those updates alone do not demonstrate that it understands ghosts, maps, or digits."
    lines += [learned, "",
              "**Observations:** four recent 84 × 84 grayscale game screens, stacked so movement can be inferred. "
              "The policy receives pixels, not a hand-coded map or game RAM.", "",
              "**Actions:** nine joystick choices: no movement, up, right, left, down, and the four diagonals. "
              "Each decision normally advances four emulator frames; sticky actions may repeat a prior move.", "",
              "**Rewards:** the environment supplies numeric game-point changes directly. Training scales those "
              "points and subtracts the disclosed life-loss penalty; evaluation reports original points. "
              "A 10-point pellet supplies 0.1 learning reward, and a 200-point event supplies 2.0 before any "
              "concurrent life-loss penalty. Each life lost subtracts 0.5. Three-step targets combine these "
              "rewards with discounting, then bootstrap from the future state unless the game truly ended. "
              "The network does not need OCR or an understanding of score digits to receive reward. "
              "The CNN can see the scoreboard as part of its images, but these outputs provide no evidence "
              "that it learned arithmetic or semantic understanding of the score. "
              "Lives are read only to form the training reward, not added as policy observations. "
              "Avoiding death is a proxy for the actual 3,000-point scoring goal: the penalty may encourage "
              "caution, but it can also discourage profitable risks and reduce raw scores.", ""]
    if not target_met:
        limitation = f"The selected policy reached 3,000 in {extra_after['count_at_least_3000']} of 20 additional "
        limitation += f"games, with scores from {number(extra_after['min'])} to {number(extra_after['max'])}, "
        limitation += f"and a classroom mean of {number(after['mean'])}. It did not meet the stated consistency target."
    else:
        limitation = f"Even in this successful finite test, additional-game scores ranged from {number(extra_after['min'])} "
        limitation += f"to {number(extra_after['max'])}. Twenty games and one training attempt do not establish reliability "
        limitation += "over all possible initial states or training seeds."
    if initial_won or class_delta <= 0:
        next_experiment = "Change only the life-loss penalty from 0.5 to 0, starting from the same inherited "
        next_experiment += "checkpoint with the same budget and selection/evaluation protocols. This tests whether "
        next_experiment += "the extra punishment discourages useful risk-taking; the current bundled experiment "
        next_experiment += "does not establish that it caused the outcome."
    else:
        next_experiment = "Change only training exploration from 0.10 to 0.05, keeping the starting checkpoint, "
        next_experiment += "budget, and evaluation protocols unchanged. This tests whether fewer random training "
        next_experiment += "moves improve consistent behavior; lower exploration could also reduce discovery."
    lines += [f"**Observed limitation:** {limitation}", "",
              f"**One next experiment:** {next_experiment} Keep the same six-hour overnight work budget "
              "and reserve finalization time; no further time extension is assumed. This proposed "
              "experiment has not been run.", "",
              "## Open and reproduce", "",
              "Open [pacman_dqn.ipynb](pacman_dqn.ipynb) on GitHub to inspect saved scores, plots, and gameplay "
              "without rerunning. Or [open the notebook in Google Colab]"
              "(https://colab.research.google.com/github/sbardacosta-code/class-3-pacman-dqn/blob/main/pacman_dqn.ipynb), "
              "select a GPU when available, and Run All. The notebook installs packages and tests the available "
              "CUDA, MPS, or CPU device. It contains the replay implementation, so a separate helper download "
              "is not needed. Local Jupyter or VS Code should use Python 3.11–3.13.", "",
              f"If the local source checkpoint is absent, the notebook downloads the [public starting checkpoint]"
              f"({inherited['url']}) and verifies SHA-256 `{inherited['sha256']}` before loading weights. "
              "That checkpoint supports the separately labeled original inherited evaluation. The final "
              "notebook then restores the saved phase-one learner for phase two; it does not repeat "
              "phase-one training from scratch. The [public phase-two resume bundle]"
              f"({continuation['resume_bundle_url']}) has SHA-256 `{continuation['resume_bundle_sha256']}`. "
              "The notebook verifies and restores this bundle, including saved learning state and history, "
              "when its local source is unavailable. Its source learner snapshot SHA-256 is "
              f"`{continuation['source_snapshot_sha256']}`. Downloading/restoring the full replay requires "
              "more storage and memory than loading a playback-only checkpoint. "
              "Keep the three values and disclosed extensions as recorded in [config.json](results/config.json). "
              "Run every cell in order. Each run gets a new `pacman_runs/` folder. Package versions, hardware, "
              "and GPU nondeterminism can affect results even with fixed seeds.", "",
              "To use the same local execution path:", "", "```bash",
              "python3.12 -m venv .venv", "source .venv/bin/activate",
              "python -m pip install -r requirements.txt", "python run_notebook.py", "```", "",
              "After training and the final evaluation, save/download the **executed `.ipynb` with outputs "
              "intact** and the complete results ZIP. In Colab, download both before ending the session. "
              "Supplementary evaluation uses the separate [validation helper](scripts/validate_advanced.py); "
              "its results do not alter the notebook's classroom evaluation.", "",
              "## Evidence archive and checkpoint storage", "",
              f"The full ZIP is kept locally at **`{report['archive']}`**, with the extracted run at "
              f"`{report['run_directory']}`. Its verified SHA-256 is `{report['archive_sha256']}`. "
              "Large checkpoints and the full replay snapshot are excluded from Git; selected evidence is "
              "published under `results/`. The public warm-start checkpoint is separately available in the "
              "[starting-model release](https://github.com/sbardacosta-code/class-3-pacman-dqn/releases/tag/warmstart-scaled-1069). "
              "The final selected playback model is available as "
              "[selected-trained.pt](https://github.com/sbardacosta-code/class-3-pacman-dqn/releases/download/overnight-20260921/selected-trained.pt). "
              "The actual phase-two starting learner/history is available in the "
              f"[resume bundle]({continuation['resume_bundle_url']}); the local bundle is "
              f"`{continuation['resume_bundle_local_path']}`. The source run remains at "
              f"`{continuation['source_run_path']}`. Individual bundle-member hashes and sizes are "
              "recorded in [overnight_provenance.json](results/overnight_provenance.json).", "",
              "The archive retains the fresh network, original inherited source and initialization, carried "
              "phase-one evidence, periodic playback "
              "checkpoints, `validation_best.pt` / `trained.pt` for the selected model, and `last_trained.pt` "
              "for the final attempted learner. `learner_state.pt` saves that last learner's model, target, "
              "optimizer, full replay with pending n-step tails, counters, and random states. It **does not "
              "save emulator states**, so it is not an exact mid-game resume point. The snapshot describes "
              "the last learner, which may differ from the selected model.", ""]
    snapshot = report["learner_snapshot"]
    lines += [f"Learner snapshot: `{snapshot['file']}`; {snapshot['size_bytes'] / 2**30:.3f} GiB; "
              f"SHA-256 `{snapshot['sha256']}`. The verifier checked archive/file bytes without deserializing "
              "this large snapshot. [verification.json](results/verification.json) records the checks and "
              "[provenance.json](results/provenance.json) identifies source and implementation changes.", "",
              f"All **{report['gif_count']} gameplay GIFs** span both phases. Phase-one GIFs are explicitly "
              "labeled as restored evidence in the final notebook; they are not claimed as newly generated "
              "phase-two games. The dashboard and other outputs are preserved. GitHub-friendly HTML "
              "representations point to identical public GIF files; the "
              "original embedded GIF bytes are retained. Supplemental 20-game validation was produced "
              "after the notebook archive and is published separately as JSON/CSV.", "",
              "The unchanged first advanced phase's executed notebook and evidence are archived under "
              "[overnight_phase1](experiments/overnight_phase1/). Earlier evidence remains available: "
              "[original 100-game run](experiments/original_100/README.md), "
              "[clipped 2,000-game run](experiments/clipped_2000/README.md), and "
              "[scaled 1,069-game run](experiments/scaled_1069/README.md). Regressions and failed "
              "improvements are retained rather than replaced with a best-game anecdote.", "",
              "Submission URL: [sbardacosta-code/class-3-pacman-dqn]"
              "(https://github.com/sbardacosta-code/class-3-pacman-dqn).", ""]
    return "\n".join(lines)


def gameplay_observations(run_id):
    if run_id != "20260921_025203_850824":
        return ("**VISUAL_REVIEW_REQUIRED:** Inspect this run's baseline, selected, and intermediate "
                "GIFs before describing their behavior.")
    return ("In the untrained clip, Ms. Pac-Man moves right and upward, then stays near the "
            "upper-left corner in the late sampled frames while many pellets remain. The selected "
            "model's best clip follows corridors through the lower-right, right side, and upper-left "
            "of the maze, removing visible pellets along its route. Its visible score reaches 3,770 "
            "near the end of the excerpt; the complete game scored 4,300. The 4,250-game intermediate "
            "clip reaches 1,790 visible points, whereas the later 5,750-game clip reaches only about "
            "980 in its excerpt. More training did not improve every demonstration. These observations "
            "come from evenly spaced frames across the actual GIFs. The best clip does not show the "
            "lower-scoring games and cannot establish reliable ghost avoidance, fruit seeking, or "
            "maze mastery; the full evaluation scores measure that inconsistency.")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "README.md",
                        help="Draft destination; default is the repository README.md")
    args = parser.parse_args(argv)
    document = render_report(read_evidence(ROOT))
    args.output.write_text(document)
    print(f"Wrote {args.output}. " + ("Gameplay review is required before publication."
          if "VISUAL_REVIEW_REQUIRED" in document else "Includes observations from this run's inspected GIFs."))


if __name__ == "__main__":
    main()
