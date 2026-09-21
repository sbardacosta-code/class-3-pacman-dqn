#!/usr/bin/env python3
"""Generate the advanced experiment's README from finished, verified evidence.

Run after build_advanced_submission.py and after copying the completed additional
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


def training_status_text(status, completed, requested):
    if status == "completed":
        return "The requested additional episode budget completed."
    if status == "time_budget":
        if completed < requested:
            return "The two-hour wall-clock cap stopped training; the requested episode budget did not complete."
        return ("The wall-clock cap was reached after all requested additional games had completed. "
                "The recorded run status remains `time_budget`.")
    if status == "interrupted":
        if completed < requested:
            return "Training was interrupted before its requested episode budget completed; this is an interrupted run."
        return ("The run was interrupted after all requested additional games had completed. "
                "The recorded run status remains `interrupted`.")
    return ("Training recorded a failure; any saved model and evaluations do not make "
            "that training run completed.")


def read_evidence(root):
    results = root / "results"
    names = ("config", "training_summary", "comparison", "warm_start_evaluation",
             "model_selection", "demo_scores", "verification", "report_data", "validation")
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
    require(config["exploration"] == 0.10 and config["episodes_requested"] == 2500 and
            config["learning_rate"] == 0.00005, "Unexpected three starting choices")
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
    lines = ["# Class 3: Ms. Pac-Man — prioritized, three-step Double DQN", "",
             f"The validation-selected agent scored **{number(after['mean'])} mean raw points** on the "
             f"five unchanged classroom games. Its inherited starting model scored **{number(warm['mean'])}** "
             f"on the same games; the fresh untrained network scored **{number(before['mean'])}**. "
             f"The new attempt completed **{number(completed)} of {number(cfg['episodes_requested'])} "
             f"additional games** and recorded **{number(summary['learning_updates'])} new learning updates**.", "",
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
                  ["Additional episode budget", "2,500", "Give existing visual features more practice, subject to the two-hour training cap."],
                  ["Learning rate", "0.00005", "Use smaller updates while fine-tuning with prioritized, longer-horizon targets."],
              ]), "",
              "The [pre-training plan](experiment_plan.md) expected more efficient learning, safer choices, "
              "and higher typical scores. It explicitly allowed a failed improvement, an inherited-model "
              "winner, or a time-budget stop. The three-hour overall work budget reserved two hours for "
              "training and the remaining time for evaluation, saving, and evidence.", "",
              "The network starts from the previous scaled-reward model, which had already received "
              f"**{number(inherited['completed_episodes'])} completed games, {number(inherited['decisions'])} "
              f"decisions, and {number(inherited['recorded_updates'])} recorded updates**. Its optimizer and "
              "replay were reset. This is a warm start, not exact resumption. The source run's separately "
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
              "The replay stores full current and future image stacks, approximately 2.63 GiB at capacity. "
              "Neither RAM observations nor ghost coordinates are supplied to the policy.", "",
              "The design draws on [Prioritized Experience Replay](https://arxiv.org/abs/1511.05952), "
              "[Rainbow](https://arxiv.org/abs/1710.02298), and [Revisiting the Arcade Learning Environment]"
              "(https://arxiv.org/abs/1709.06009). This is a small adaptation, not a reproduction of their "
              "large-budget results. Bundling replay, return length, reward shaping, and batching prevents "
              "an isolated causal claim about any one change.", "",
              "## Actual training budget and selected model", ""]
    status = summary["status"]
    status_text = training_status_text(status, completed, cfg["episodes_requested"])
    lines += [f"**Status: `{status}`.** {status_text}", "",
              table(["Budget", "Completed games", "Decisions", "Recorded learning updates"], [
                  ["Inherited training", number(inherited["completed_episodes"]), number(inherited["decisions"]), number(inherited["recorded_updates"])],
                  ["Entire new attempt", number(completed), number(summary["total_decisions"]), number(summary["learning_updates"])],
                  ["Inherited + entire attempt", number(inherited["completed_episodes"] + completed), number(inherited["decisions"] + summary["total_decisions"]), number(inherited["recorded_updates"] + summary["learning_updates"])],
                  ["New work retained in selected candidate", number(selected["additional_episodes"]), number(selected["decisions"]), number(selected["updates"])],
                  ["Inherited + selected candidate", number(inherited["completed_episodes"] + selected["additional_episodes"]), number(inherited["decisions"] + selected["decisions"]), number(inherited["recorded_updates"] + selected["updates"])],
              ]), "",
              "The full attempt measures compute spent; the selected-candidate rows measure training "
              "retained in the submitted policy. Candidate snapshots can include decisions from parallel "
              "games that had not yet completed. The selected checkpoint can precede the end of the attempt.", ""]
    hardware = validation.get("hardware", {})
    chip = hardware.get("cpu_model", hardware.get("processor", "not recorded"))
    memory = hardware.get("physical_memory_bytes")
    hardware_label = chip + (f"; {memory / 2**30:g} GiB memory" if memory else "")
    lines += [table(["Measure", "Recorded value"], [
        ["Run ID", f"`{report['run_id']}`"],
        ["Additional games started / completed", f"{number(summary['episodes_started'])} / {number(completed)}"],
        ["Training timer", duration(summary["elapsed_seconds_including_periodic_demos"])],
        ["Monotonic elapsed time", duration(summary["active_monotonic_elapsed_seconds"])],
        ["Training cap", duration(summary["training_time_limit_seconds"])],
        ["Training device", str(cfg["device"]).upper()],
        ["Host hardware recorded by additional evaluation", hardware_label],
        ["Python / PyTorch", f"{cfg['python']} / {cfg['packages']['torch']}"],
        ["Platform", cfg["platform"]],
        ["Replay entries at stop", number(summary["replay_size"])],
        ["Lives lost in new training", number(summary["training_lives_lost"])],
    ]), "",
              "The training timer includes collection, updates, periodic GIF/checkpoint work, and selection "
              "evaluations performed inside training. It excludes setup, the fresh/inherited classroom "
              "evaluations before training, the final selection evaluation, and final snapshot/ZIP saving "
              "and classroom/additional evaluations. Package details: [config.json](results/config.json) "
              "and [environment.txt](results/environment.txt).", ""]
    partial = summary["partial_games"]
    partial_decisions = sum(game["decisions"] for game in partial)
    completed_decisions = sum(int(row["steps"]) for row in evidence["training_rows"])
    lines += [f"Parallel-game accounting: **{number(completed_decisions)} completed-game decisions + "
              f"{number(partial_decisions)} decisions in {len(partial)} partial games = "
              f"{number(summary['total_decisions'])} new decisions**. Partial-game scores are not inserted "
              "as completed training CSV rows.", ""]
    if partial:
        lines += [table(["Slot", "Training game ID", "Seed", "Partial decisions", "Raw score", "Lives lost"],
                        [[game["slot"], game["training_episode_id"], game["seed"], game["decisions"],
                          number(game["score"]), game["lives_lost"]] for game in partial]), ""]
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
              "additional games, and at the end. The predeclared selection metric is **mean raw score − "
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
              "**VISUAL_REVIEW_REQUIRED:** Watch the actual baseline, selected, and intermediate GIFs. "
              "Replace this paragraph with concrete observed behavior and a limitation; do not infer ghost "
              "avoidance, planning, or maze mastery from scores alone.", "",
              "The periodic GIFs below show the current learner every 25 completed additional games. "
              "They all use seed 101 and are demonstrations only. They may occur after the eventually "
              "selected checkpoint; those later GIFs do not depict the submitted policy. Full scores: "
              "[demo_scores.json](results/demo_scores.json).", ""]
    for demo in evidence["demo_scores"]:
        episode = demo["episode"]
        lines += [f"**After {number(episode)} additional games — seed 101, score {number(demo['scores'][0])}.**", "",
                  f"![After {episode} additional games](results/demos/episode_{episode:04d}.gif)", ""]
    lines += ["## Training curves and what the policy learned", "",
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
        lines += [table(["Completed additional games", "Mean raw training score", "Mean decisions per game"], blocks), ""]
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
              f"**One next experiment:** {next_experiment} Keep the same two-hour training cap and "
              "three-hour overall budget. This proposed experiment has not been run.", "",
              "## Open and reproduce", "",
              "Open [pacman_dqn.ipynb](pacman_dqn.ipynb) on GitHub to inspect saved scores, plots, and gameplay "
              "without rerunning. Or [open the notebook in Google Colab]"
              "(https://colab.research.google.com/github/sbardacosta-code/class-3-pacman-dqn/blob/main/pacman_dqn.ipynb), "
              "select a GPU when available, and Run All. The notebook installs packages and tests the available "
              "CUDA, MPS, or CPU device. It contains the replay implementation, so a separate helper download "
              "is not needed. Local Jupyter or VS Code should use Python 3.11–3.13.", "",
              f"If the local source checkpoint is absent, the notebook downloads the [public starting checkpoint]"
              f"({inherited['url']}) and verifies SHA-256 `{inherited['sha256']}` before loading weights. "
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
              "[starting-model release](https://github.com/sbardacosta-code/class-3-pacman-dqn/releases/tag/warmstart-scaled-1069).", "",
              "The archive retains the fresh network, inherited source and initialization, periodic playback "
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
              f"All **{report['gif_count']} gameplay GIFs**, the dashboard, and other notebook outputs are "
              "preserved. GitHub-friendly HTML representations point to identical public GIF files; the "
              "original embedded GIF bytes are retained. Supplemental 20-game validation was produced "
              "after the notebook archive and is published separately as JSON/CSV.", "",
              "Earlier evidence remains available: [original 100-game run](experiments/original_100/README.md), "
              "[clipped 2,000-game run](experiments/clipped_2000/README.md), and "
              "[scaled 1,069-game run](experiments/scaled_1069/README.md). Regressions and failed "
              "improvements are retained rather than replaced with a best-game anecdote.", "",
              "Submission URL: [sbardacosta-code/class-3-pacman-dqn]"
              "(https://github.com/sbardacosta-code/class-3-pacman-dqn).", ""]
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "README.md",
                        help="Draft destination; default is the repository README.md")
    args = parser.parse_args(argv)
    document = render_report(read_evidence(ROOT))
    args.output.write_text(document)
    print(f"Wrote {args.output}. Replace VISUAL_REVIEW_REQUIRED after watching actual GIFs before publication.")


if __name__ == "__main__":
    main()
