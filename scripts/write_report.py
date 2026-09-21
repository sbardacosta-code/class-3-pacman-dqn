#!/usr/bin/env python3
"""Draft README.md from verified, completed experiment evidence.

Run after scripts/build_submission.py. The current run includes observations
from visual review; review and update them before publishing a different run.
This script does not train, evaluate, alter results, or change the notebook.
"""

from __future__ import annotations

import json
from pathlib import Path
import statistics


ROOT = Path(__file__).resolve().parents[1]
REPOSITORY = "https://github.com/sbardacosta-code/class-3-pacman-dqn"


def number(value, places=1):
    value = float(value)
    return f"{int(value):,}" if value.is_integer() else f"{value:,.{places}f}"


def signed(value, places=1):
    return f"{float(value):+,.{places}f}"


def elapsed(seconds):
    hours, remainder = divmod(round(float(seconds)), 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours}h {minutes:02d}m {seconds:02d}s"


def heldout_section(report):
    """Include every additional validation game if the separate evaluation exists."""
    path = ROOT / "results" / "heldout_validation.json"
    if not path.exists():
        return []
    evidence = json.loads(path.read_text())
    clipped = evidence["models"]["clipped_reference"]
    scaled = evidence["models"]["scaled_reward"]
    seeds = evidence["settings"]["seeds"]
    settings = evidence["settings"]
    if (settings["exploration"] != report["config"]["eval_exploration"] or
            settings["max_decisions"] != report["config"]["max_decisions_per_game"]):
        raise ValueError("Additional validation changed the evaluation settings.")
    expected_clipped_run = Path(report["clipped_2000_reference"]["archive"]).stem
    if scaled["run_id"] != report["run_id"] or clipped["run_id"] != expected_clipped_run:
        raise ValueError("Held-out evaluation belongs to different training runs.")
    if set(seeds) & set(report["config"]["eval_seeds"]):
        raise ValueError("Additional validation seeds overlap the classroom seeds.")
    if clipped["seeds"] != seeds or scaled["seeds"] != seeds:
        raise ValueError("Held-out model results use different validation seeds.")
    if len(clipped["scores"]) != len(seeds) or len(scaled["scores"]) != len(seeds):
        raise ValueError("Held-out evaluation is missing games.")
    lines = [
        "### Additional validation on separate seeds",
        "",
        f"Both final saved models were evaluated on **{len(seeds)} additional seeds** "
        "under the same exploration and per-game time limit as classroom evaluation. "
        "The models receive no learning updates during these games. These results "
        "are a separate check of consistency; they do not replace the five official "
        "scores or change the classroom leaderboard mean. Every validation game "
        "is reported below, including low scores.",
        "",
        "| Model | Mean | Median | Minimum | Maximum | Games ≥3,000 |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for label, model in [("Previous clipped reward", clipped),
                         ("New scaled reward", scaled)]:
        scores = model["scores"]
        hits = sum(score >= 3000 for score in scores)
        lines.append(f"| {label} | {number(statistics.mean(scores))} | "
                     f"{number(statistics.median(scores))} | {number(min(scores))} | "
                     f"{number(max(scores))} | {hits}/{len(scores)} ({hits / len(scores):.0%}) |")
    lines += [
        "",
        "| Game | Seed | Previous clipped reward | New scaled reward | Change |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for index, seed in enumerate(seeds):
        old, new = clipped["scores"][index], scaled["scores"][index]
        lines.append(f"| {index + 1} | {seed} | {number(old)} | {number(new)} | "
                     f"{signed(new - old, 0)} |")
    scores = scaled["scores"]
    hits = sum(score >= 3000 for score in scores)
    if hits == len(scores):
        interpretation = (
            "All additional validation games reached 3,000. This supports consistency "
            "on these sampled games, while a finite evaluation still cannot guarantee "
            "performance on every future game."
        )
    else:
        interpretation = (
            f"The new agent reached 3,000 in {hits}/{len(scores)} additional games. "
            "The complete score distribution, rather than its best game, shows the "
            "remaining gap to dependable 3,000+ play."
        )
    if report["run_id"] == "20260920_152728_434799":
        interpretation += (
            " The new mean is only 9.5 points (0.63%) higher, while its median is "
            "lower: 1,285 versus 1,425. Its 4,600-point best game helps lift the mean; "
            "both models reached 3,000 only once in 20 games. These results do not "
            "show a convincing improvement in consistency. The models also received "
            "unequal amounts of training, so this is not a controlled estimate of "
            "the effect of reward scaling."
        )
    lines += [
        "",
        interpretation,
        "",
        "Validation settings, all scores and game lengths, time-limit indicators, "
        "and model identities are saved in "
        "[heldout_validation.json](results/heldout_validation.json) and "
        "[heldout_validation.csv](results/heldout_validation.csv). Both models "
        "come from single training runs; these additional games do not measure "
        "variation across retraining seeds.",
        "",
        "To reproduce this supplemental evaluation with the locally retained checkpoints:",
        "",
        "```bash",
        f".venv/bin/python scripts/validate_generalization.py --run {report['run_directory']} "
        f"--reference-run {Path(report['clipped_2000_reference']['archive']).with_suffix('')}",
        "```",
        "",
        "[validate_generalization.py](scripts/validate_generalization.py) writes its "
        f"output under `.execution/heldout_{report['run_id']}/`. The resulting JSON "
        "and CSV are copied to `results/` for publication. They are supplemental "
        "artifacts produced after training, separate from the notebook-generated ZIP.",
        "",
    ]
    return lines


def main():
    data_path = ROOT / "results" / "report_data.json"
    data = json.loads(data_path.read_text())
    original = json.loads((ROOT / "experiments" / "original_100" /
                           "results" / "comparison.json").read_text())
    clipped_path = ROOT / "experiments" / "clipped_2000" / "results"
    clipped = json.loads((clipped_path / "comparison.json").read_text())
    clipped_summary = json.loads((clipped_path / "training_summary.json").read_text())
    config = data["config"]
    if config.get("training_reward_scale") != 0.01 or config.get("training_reward_clipping") is not None:
        raise ValueError("This report requires the verified, unclipped 0.01-scaled reward experiment.")
    summary = data["summary"]
    before = data["evaluation"]["before"]
    after = data["evaluation"]["after"]
    best = data["best_trained_game"]
    actual = summary["completed_episodes"]
    requested = config["episodes_requested"]
    duration = summary["elapsed_seconds_including_periodic_demos"]
    budget_seconds = config["training_time_limit_seconds"]
    rate = config["learning_rate"]
    reviewed_run = data["run_id"] == "20260920_152728_434799"
    original_scores = dict(zip(original["after"]["seeds"], original["after"]["scores"]))
    original_mean = original["after"]["mean"]
    clipped_scores = dict(zip(clipped["after"]["seeds"], clipped["after"]["scores"]))
    clipped_mean = clipped["after"]["mean"]
    score_delta = after["mean"] - before["mean"]
    clipped_delta = after["mean"] - clipped_mean
    clipped_changes = [score - clipped_scores[seed]
                       for seed, score in zip(after["seeds"], after["scores"])]
    improvement = "increased" if score_delta > 0 else "decreased" if score_delta < 0 else "was unchanged"
    zero_updates = summary["learning_updates"] == 0
    interrupted = summary["status"] != "completed"
    cap_note = (
        f"**This run stopped early:** its recorded status is `{summary['status']}`, "
        f"with {number(actual)} of {number(requested)} requested episodes completed. "
        f"The operational training limit was {number(budget_seconds / 3600)} hours. "
        "The completed games and all collected decisions/updates are reported below."
        if interrupted else
        f"The run completed all **{number(actual)} requested episodes** before the "
        f"{number(budget_seconds / 3600)}-hour training cap; it was not interrupted."
    )
    if reviewed_run and interrupted:
        cap_note = (
            f"**This run was manually interrupted once**, after **{number(actual)} of "
            f"{number(requested)} requested episodes**. Its recorded status is "
            f"`{summary['status']}`. The notebook's elapsed training timer was "
            f"**{elapsed(duration)}**, below the configured "
            f"**{number(budget_seconds)}-second cap**; the automatic cap did not stop this run. "
            "Evaluation and saving were then allowed to finish.\n\n"
            "The interruption followed an apparent host or session pause. Monitoring "
            "observations jumped from approximately 22:40 UTC to 02:39 UTC while the "
            "notebook's training timer advanced only from about 57 to 75 minutes. "
            "The cause of that gap was not confirmed. The original three-hour wall-clock "
            "window had already been exceeded when monitoring resumed, so training was "
            "stopped rather than continuing toward 2,000 episodes. The elapsed training "
            "value below must not be read as the full wall-clock duration."
        )
    partial = data["partial_episode_decisions"]
    partial_note = (
        f"The final interrupted game contributed **{number(partial)} decisions** that are "
        "included in the total decision count. The checkpoint preserves the weights at "
        "interruption. That game is not counted as a completed "
        "episode or included as a CSV row."
        if partial else
        "There are no partial-episode decisions: the CSV covers every recorded training decision."
    )
    learning_note = (
        "**No learning updates occurred.** This is an untrained/setup outcome, not evidence "
        "of a learned policy."
        if zero_updates else
        f"The recorded counter reports **{number(summary['learning_updates'])} learning updates**. "
        "The verified trained weights differ from the untrained weights and remain finite. "
        "This establishes parameter learning, not mastery of the game."
    )
    if reviewed_run and not zero_updates:
        learning_note += (
            " The interruption occurred at a scheduled update boundary before its "
            "counter increment, leaving the "
            "recorded counter one below the count implied by the decision schedule. "
            "An extra completed update is not assumed or added to the reported counter."
        )
    if score_delta > 0:
        conclusion = (
            "The final policy earned a higher mean score on this fixed five-game comparison. "
            "That supports improvement under these evaluation conditions; it does not establish "
            "consistent performance across all possible games."
        )
    elif score_delta < 0:
        conclusion = (
            "The final policy earned a lower mean score than its untrained baseline on this "
            "fixed five-game comparison. Its training updates did not produce a higher "
            "evaluation mean; the failed outcome is reported honestly."
        )
    else:
        conclusion = (
            "The final policy matched the untrained mean on this fixed five-game comparison. "
            "The experiment provides no mean-score evidence of improvement."
        )

    if clipped_delta > 0:
        reward_conclusion = (
            "The scaled-reward policy improved the mean relative to the previous clipped-reward "
            "policy on these five games. This is encouraging evidence for this experiment, "
            "not proof that reward scaling always helps or that it produced reliable 3,000-point play."
        )
    elif clipped_delta < 0:
        reward_conclusion = (
            "The scaled-reward policy scored lower than the previous clipped-reward policy. "
            f"Its mean was **{abs(clipped_delta):,.1f} points "
            f"({abs(clipped_delta) / clipped_mean:.1%}) lower** on these five games. "
            "The lower result is preserved rather than reported as an improvement."
        )
    else:
        reward_conclusion = (
            "The scaled-reward policy matched the clipped-reward mean. These five games "
            "provide no mean-score evidence that the reward change improved performance."
        )
    if actual != clipped_summary["completed_episodes"]:
        reward_conclusion += (
            f" This run completed {number(actual)} episodes versus "
            f"{number(clipped_summary['completed_episodes'])} for the clipped agent, with "
            "fewer recorded training decisions and updates. The unequal training budget "
            "is a substantial confounding factor: the score difference cannot be attributed "
            "solely to the reward transformation."
        )
    target_hits = sum(score >= 3000 for score in after["scores"])
    target_note = (
        "All five classroom games reached 3,000 points. The target was met on this small "
        "fixed evaluation set, but consistency on unseen games is not established."
        if target_hits == len(after["scores"]) else
        f"Only {target_hits} of {len(after['scores'])} classroom games reached 3,000 points. "
        "The target of consistently scoring 3,000+ was not met on this evaluation set."
    )
    lines = [
        "# Class 3: Train a Ms. Pac-Man Agent",
        "",
        f"The **scaled-reward {config.get('algorithm', 'DQN')}** experiment completed "
        f"**{number(actual)} / {number(requested)} episodes**. Its mean score on the "
        f"five unchanged evaluation games {improvement} from **{number(before['mean'])} "
        f"to {number(after['mean'])}** ({signed(score_delta)} points; "
        f"{signed(score_delta / before['mean'] * 100)}%). The previous "
        f"2,000-episode clipped-reward agent scored **{number(clipped_mean)}**; this "
        f"experiment's mean differs by **{signed(clipped_delta)} points**. The original "
        f"100-episode DQN scored **{number(original_mean)}**.",
        "",
        f"The leaderboard score for this final saved model is **{number(after['mean'])}**. "
        "The table below reports every game. The submitted model is "
        "the final model from this run, not a checkpoint selected by its best evaluation score.",
        "",
        f"The earlier **{number(clipped_mean)}** classroom mean and its full evidence remain "
        "available in the [preserved clipped-reward experiment](experiments/clipped_2000/README.md). "
        "This report keeps the latest outcome visible even when its mean is lower.",
        "",
        "This repository contains the [final executed notebook](pacman_dqn.ipynb), with "
        "all 28 code cells executed in order and all outputs retained. It extends the "
        "[supplied Pac-Man project](https://github.com/pepealonso95/pacman-dqn). The "
        "[original 100-episode experiment](experiments/original_100/README.md), "
        "[original notebook](experiments/original_100/pacman_dqn.ipynb), "
        "[previous clipped-reward experiment](experiments/clipped_2000/README.md), and "
        "[previous executed notebook](experiments/clipped_2000/pacman_dqn.ipynb) remain "
        "available with their evidence.",
        "",
        "## Choices and expectation recorded before training",
        "",
        "| Setting | Choice | Reason |",
        "| --- | ---: | --- |",
        f"| Exploration | {config['exploration']:.2f} | Retain the previous balance between "
        "trying alternatives and using learned action preferences. |",
        f"| Episode budget | {number(requested)} | Match the previous requested budget "
        "within the same two-hour training cap. |",
        f"| Learning rate | {rate:.4f} | Keep the update size fixed so reward transformation "
        "is the only changed learning setting. |",
        "",
        "The [before-training plan](experiment_plan.md) proposed preserving the relative "
        "value of game points. The hypothesis was that making a 200-point gain more "
        "valuable than a 10-point gain would encourage higher-value play. The previous "
        "reward clipped both to +1; the new reward stores **raw points × 0.01 without "
        "clipping**, producing +2 and +0.1 respectively. Removing clipping can also make "
        "learning less stable, so improvement was not guaranteed. **No life-loss penalty "
        "was added.** Testing that separately avoids changing two reward properties at once.",
        "",
        "Exploration was 100% for the first 1,000 decisions, then stayed constant at "
        f"**{config['exploration']:.0%}**. No exploration-decay schedule was introduced.",
        "",
        "The following extensions from the previous experiment were retained unchanged:",
        "",
        "- **Double DQN:** the online network chooses the next action; the target network "
        "estimates that chosen action's value. Separating selection from valuation aims "
        "to reduce overoptimistic action-value estimates. "
        "[Double DQN paper](https://arxiv.org/abs/1509.06461).",
        f"- **Replay capacity {number(config['replay_capacity'])}**, already increased from 5,000: "
        "retain a wider range of past experiences for random training batches.",
        "",
        f"An operational **{number(budget_seconds / 3600)}-hour training cap** saves the current "
        "model through the notebook's interruption path, then allows final evaluation and "
        "archiving. This cap does not change the per-game evaluation time limit. The network "
        "architecture, observations, actions, game setup, training seed, and evaluation "
        "settings remain unchanged. All non-reward training settings match the previous "
        "2,000-episode run. This run starts with fresh weights, optimizer, and replay "
        "memory; it is not another 2,000 episodes appended to the previous model.",
        "",
        "The assignment allows explained changes to other hyperparameters and optional "
        "custom agents. Replay capacity is an additional hyperparameter; Double DQN is "
        "disclosed as an algorithm extension to the supplied implementation. The new "
        "training-reward transformation is also disclosed. Classroom evaluation still "
        "reports the original game points under the original settings.",
        "",
        "## Actual run",
        "",
        cap_note,
        "",
        "| Measure | Recorded result |",
        "| --- | --- |",
        f"| Run ID | `{data['run_id']}` |",
        f"| Status | `{summary['status']}` |",
        f"| Episodes requested / completed | {number(requested)} / {number(actual)} |",
        f"| Training decisions | {number(summary['total_decisions'])} |",
        f"| Learning updates | {number(summary['learning_updates'])} |",
        f"| Notebook training timer | {duration:,.3f} seconds ({elapsed(duration)}), including periodic demos |",
        f"| Configured training cap | {number(budget_seconds)} seconds |",
        "| Hardware | Apple M4, 16 GiB unified memory |",
        f"| Training device | `{config['device']}` |",
        f"| Runtime | Python {config['python']}; PyTorch {config['packages']['torch']} |",
        f"| Platform | {config['platform']} |",
        "",
        "The elapsed training time includes periodic demonstration evaluation and saving; "
        "it excludes package setup and the separate baseline/final five-game evaluations. "
        "It is not the total wall-clock time spent completing the assignment.",
        "",
        partial_note,
        "",
        learning_note,
        "",
        "For comparison, the previous clipped-reward run completed "
        f"**{number(clipped_summary['completed_episodes'])} episodes**, "
        f"**{number(clipped_summary['total_decisions'])} decisions**, and "
        f"**{number(clipped_summary['learning_updates'])} updates** in "
        f"**{elapsed(clipped_summary['elapsed_seconds_including_periodic_demos'])}**. "
        "Matching the episode request does not match the number of decisions or updates: "
        "different learned policies produce different game lengths. "
        + ("The interruption also left this run with fewer completed episodes, making "
           "the actual training budgets substantially unequal." if interrupted else
           "The actual recorded budgets above are the basis for comparison."),
        "",
        "Evidence: [training_summary.json](results/training_summary.json), "
        "[training.csv](results/training.csv), [config.json](results/config.json), "
        "[environment.txt](results/environment.txt), "
        "[verification.json](results/verification.json), and "
        "[source provenance](results/provenance.json).",
        "",
        "## All five evaluation games",
        "",
        "Both new-run evaluations use the same five seeds, **5% exploration**, and "
        "**3,000-decision limit per game**. Evaluation never updates the model. The baseline "
        "is a fresh **untrained neural network**, not a random-action agent. The original "
        "100-episode and previous 2,000-episode results used these same evaluation settings. "
        "All values in this section are raw game points, not scaled learning rewards.",
        "",
        "| Game | Seed | New untrained baseline | Original 100 episodes | Clipped 2,000 episodes | New scaled-reward agent | Change vs. clipped |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for i, seed in enumerate(before["seeds"]):
        lines.append(f"| {i + 1} | {seed} | {number(before['scores'][i])} | "
                     f"{number(original_scores[seed])} | {number(clipped_scores[seed])} | "
                     f"{number(after['scores'][i])} | {signed(clipped_changes[i], 0)} |")
    lines += [
        f"| **Mean** | | **{number(before['mean'])}** | **{number(original_mean)}** | "
        f"**{number(clipped_mean)}** | **{number(after['mean'])}** | **{signed(clipped_delta)}** |",
        "",
        f"Relative to the new baseline, **{data['improved_games']} games improved, "
        f"{data['worsened_games']} worsened, and {data['unchanged_games']} were unchanged**. "
        f"Relative to the clipped agent, **{sum(x > 0 for x in clipped_changes)} improved, "
        f"{sum(x < 0 for x in clipped_changes)} worsened, and "
        f"{sum(x == 0 for x in clipped_changes)} were unchanged**. "
        f"Time-limited games: **{data['time_limited_before']} before / "
        f"{data['time_limited_after']} after**.",
        "",
        "Machine-readable evidence: [all new comparison scores](results/comparison.json), "
        "[baseline.json](results/baseline.json), and "
        "the preserved [clipped comparison](experiments/clipped_2000/results/comparison.json) "
        "and [original comparison](experiments/original_100/results/comparison.json).",
        "",
        conclusion,
        "",
        reward_conclusion,
        "",
        "### Progress toward consistent 3,000+ scores",
        "",
        "A high mean alone does not establish consistency. The minimum, median, and "
        "number of games reaching 3,000 help show how uneven performance is.",
        "",
        "| Version | Mean | Median | Minimum | Maximum | Games ≥3,000 |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for label, evaluation in [
        ("New untrained baseline", before),
        ("Original 100 episodes", original["after"]),
        ("Clipped 2,000 episodes", clipped["after"]),
        ("New scaled-reward agent", after),
    ]:
        scores = evaluation["scores"]
        hits = sum(score >= 3000 for score in scores)
        lines.append(f"| {label} | {number(evaluation['mean'])} | "
                     f"{number(statistics.median(scores))} | {number(min(scores))} | "
                     f"{number(max(scores))} | {hits}/{len(scores)} ({hits / len(scores):.0%}) |")
    lines += [
        "",
        f"**{target_note}**",
        "",
    ]
    lines += heldout_section(data)
    lines += [
        "## Gameplay evidence",
        "",
        "Each GIF shows at most the first 20 seconds of game time, played approximately "
        "4× faster and looping twice. The associated score covers the entire evaluated "
        "game, not just the excerpt. Open or reload a GIF to replay it.",
        "",
        f"**Untrained network — seed {before['seeds'][0]}, full-game score "
        f"{number(before['scores'][0])}.**",
        "",
        "![Untrained gameplay](results/demos/episode_0000.gif)",
        "",
        f"**Best of the five final trained games — seed {best['seed']}, full-game "
        f"score {number(best['score'])}.** This is the best game of the final saved "
        "model, not a model chosen from all checkpoints.",
        "",
        "![Best trained gameplay](results/demos/final_best.gif)",
        "",
        ("**Gameplay interpretation:** In the best final clip, Ms. Pac-Man reaches a "
         "power pellet on the right side, the ghosts turn blue, and the score reaches "
         "1,110 by approximately 17 seconds. Both reserve-life icons remain visible "
         "through the excerpt. The score then stays at 1,110 for its remaining few "
         "seconds; the complete game eventually scores 1,930. The episode-1,050 "
         "sample reaches 580 within the excerpt and 770 over its complete game. "
         "By contrast, the episode-850 sample loses a life early and scores only 170 "
         "over its complete game. These visible setbacks show that learning is uneven. "
         "The baseline also loses lives and finishes with 350. The final best clip "
         "uses seed 404 while the baseline and intermediate clips use seed 101, so "
         "this is not a matched-seed visual comparison. The best clip is selected "
         "evidence and cannot establish typical play, reliable ghost avoidance, "
         "fruit-seeking, or an understanding of the scoreboard digits."
         if reviewed_run else
         "**Gameplay interpretation:** Review the actual GIFs before publishing a new "
         "run. The best final excerpt is selected from five games and cannot establish "
         "typical play by itself."),
        "",
        "### Every intermediate gameplay sample",
        "",
        "The notebook saved a separate demonstration and playback checkpoint every "
        "25 completed episodes. All intermediate GIFs from this run are embedded below. "
        "These demonstrations use seed 101 and are single-game scores, not official "
        "five-game evaluation means. Raw records: [demo_scores.json](results/demo_scores.json).",
        "",
    ]
    groups = {}
    for demo in data["periodic_demo_scores"]:
        groups.setdefault((demo["episode"] - 1) // 250, []).append(demo)
    for group in sorted(groups):
        items = groups[group]
        lines += ["<details>",
                  f"<summary>Episodes {items[0]['episode']}–{items[-1]['episode']} "
                  f"({len(items)} gameplay samples)</summary>", ""]
        for demo in items:
            episode = demo["episode"]
            lines += [
                f"**Episode {episode} — seed {demo['seeds'][0]}, full-game score "
                f"{number(demo['scores'][0])}.**",
                "",
                f"![Gameplay after episode {episode}](results/demos/episode_{episode:04d}.gif)",
                "",
            ]
        lines += ["</details>", ""]
    if not groups:
        lines += ["No intermediate sample was produced because the run completed fewer "
                  "than 25 episodes.", ""]

    lines += [
        "## Training curves and what the agent learned",
        "",
        "![Training score, mean update loss, and exploration](results/training_dashboard.png)",
        "",
        "The dashboard shows raw training scores with a rolling average of up to "
        "25 games, average update loss per episode, and the episode's final exploration "
        "rate. The warm-up may end partway through an episode. Training uses different "
        f"seeds and {config['exploration']:.0%} exploration, so training scores are not "
        "substitutes for the fixed evaluation scores.",
        "",
        "Non-overlapping training blocks provide another view of the score trend:",
        "",
        "| Completed episodes | Games in block | Mean raw training score | Mean episode loss |",
        "| --- | ---: | ---: | ---: |",
    ]
    for block in data["training_blocks_of_250"]:
        loss = (f"{block['mean_episode_loss']:.5f}"
                if block["mean_episode_loss"] is not None else "No updates")
        lines.append(f"| {block['first_episode']}–{block['last_episode']} | {block['count']} | "
                     f"{number(block['mean_score'])} | {loss} |")
    lines += [
        "",
        ("A partial final block is included as recorded. " if actual % 250 else
         "Every block contains 250 completed episodes. ") + "Detailed 25-episode block "
        "averages and the original measurements are available in "
        "[report_data.json](results/report_data.json) and [training.csv](results/training.csv). "
        "DQN's target values change as learning proceeds, so loss is not a direct "
        "measure of gameplay quality. Lower loss does not guarantee better play. "
        "Loss magnitudes across the clipped and scaled experiments are not directly "
        "comparable because the reward and target scales changed.",
        "",
        ("In this run, the score average rises from roughly 750 in the first "
         "250 games to around 1,100 in later blocks, with substantial fluctuations. "
         "Loss is noisy and generally increases rather than steadily falling; "
         "the dashboard does not show stable convergence. Exploration stays at "
         "10% after warm-up."
         if reviewed_run else ""),
        "",
        "**Observations:** four consecutive grayscale game screens, each 84 × 84 "
        "pixels, provide positions and recent movement. The network receives pixels, "
        "not explicit ghost coordinates or a hand-coded map.",
        "",
        "**Actions:** nine joystick choices: no movement, up, right, left, down, and "
        "the four diagonals. Each decision spans four emulator frames. The unchanged "
        "sticky-action setting can repeat the previous action.",
        "",
        "**Rewards:** game points provide a numeric reward directly from the emulator. "
        "The agent does not need to read scoreboard digits to receive this signal. "
        "This experiment stores **raw points × 0.01 without clipping** for learning. "
        "A decision earning 10, 50, 200, or 400 points therefore gives a training reward "
        "of 0.1, 0.5, 2, or 4. In the previous experiment, each of those positive "
        "decisions gave +1 after clipping. All reported game scores remain unscaled.",
        "",
        "There is **no added survival bonus or penalty for losing a life**. Pellets, "
        "power pellets, fruit, and vulnerable ghosts are valuable through the points "
        "the game awards. Staying alive can help by providing future opportunities "
        "to score, but the training code does not explicitly reward each second of "
        "survival. Losing a single life does not by itself end the training episode.",
        "",
        "**How learning works:** replay memory stores the screen stack, chosen action, "
        "scaled reward, and next screen stack. Each update samples past experiences "
        "and adjusts action-value predictions toward the immediate reward plus "
        "discounted estimated future rewards. Double DQN uses the online network to "
        "select the next action and the target network to estimate its value. The "
        "agent receives no written instructions to seek fruit or avoid ghosts, and "
        "higher scores alone do not prove that it learned a specific visual concept.",
        "",
        "**What the evidence supports:** " + learning_note + " " + reward_conclusion,
        "",
        "**Observed limitation of this experiment:** this is one training run with a "
        "single training seed. The episode-850 GIF demonstrates early life loss, "
        "and the additional validation still has only one 3,000-point game out of 20. "
        "The classroom comparison has only five evaluation "
        "games, and the best short GIF is selected evidence. " + target_note + " "
        "Reward transformation is the only changed learning setting relative to the "
        "clipped 2,000-episode experiment, but different trajectories, actual decisions "
        "and updates, and GPU nondeterminism still limit causal conclusions from one "
        "run per setting. Relative to the original 100-episode experiment, several "
        "settings changed, so that broader comparison cannot isolate their contributions.",
        "",
        "**Proposed next experiment, if life loss remains a visible weakness:** keep "
        "scaled rewards and every other setting fixed, and add **−0.5 training reward "
        "per lost life**. At this scale, that is a cost equivalent to 50 game points. "
        "It would test whether an explicit life-loss signal improves survival enough "
        "to increase raw scores; it could instead discourage worthwhile risks. This "
        "is an experimental starting value, not a proven optimum. Run a no-penalty "
        "control and a penalty treatment with a matched training budget before "
        "attributing any difference to that single changed setting. Report actual "
        "decisions and updates again, and keep evaluation raw scoring and settings "
        "unchanged. **No life-loss penalty experiment has been performed in this run.**",
        "",
        "## Open and reproduce",
        "",
        "Open [pacman_dqn.ipynb](pacman_dqn.ipynb) on GitHub to inspect the saved "
        "scores, plot, and gameplay without rerunning. To execute it, "
        f"[open the notebook in Google Colab](https://colab.research.google.com/github/"
        "sbardacosta-code/class-3-pacman-dqn/blob/main/pacman_dqn.ipynb), select a GPU "
        "if available, and choose Run All. Download both the executed `.ipynb` and "
        "results ZIP before ending the session. Do not clear the cell outputs.",
        "",
        "For local Jupyter or VS Code, use a Python 3.11–3.13 kernel and keep "
        "`pacman_player.py` beside the notebook for optional popup playback. The "
        "notebook installs its packages and tests CUDA, Apple MPS, or CPU automatically. "
        "Read the documented extensions, confirm the three settings, and run every "
        "cell in order. Run All starts a fresh experiment and a new results directory; "
        "it does not resume a saved checkpoint.",
        "",
        "The programmatic execution method used for this repository is:",
        "",
        "```bash",
        "python3.12 -m venv .venv",
        "source .venv/bin/activate",
        "python -m pip install -r requirements.txt",
        "python run_notebook.py",
        "```",
        "",
        "[run_notebook.py](run_notebook.py) executes the notebook sequentially and "
        "saves its real outputs after each code cell. [requirements.txt](requirements.txt) "
        "records the environment used here; the notebook's installer retains its "
        "version ranges. Hardware, package changes, and GPU nondeterminism can change "
        "scores even with the same seeds.",
        "",
        "After all cells finish, evidence can be assembled and checked with:",
        "",
        "```bash",
        f"python scripts/build_submission.py --run {data['run_directory']}",
        "```",
        "",
        "[build_submission.py](scripts/build_submission.py) verifies all 28 ordered "
        "execution counts, evaluation invariants, evaluation means, model weights, "
        "all periodic checkpoints, and the ZIP. It checks that every notebook GIF "
        "matches the corresponding saved file byte for byte. GitHub does not directly "
        "render the notebook's `image/gif` output format, so it adds an HTML image "
        "representation pointing to the identical public GIF. Original embedded "
        "GIF bytes and all other notebook outputs are preserved, and no cell source "
        "is changed by this display step.",
        "",
        "## Full archive and checkpoints",
        "",
        f"The complete final-run archive is retained locally at **`{data['archive']}`**, "
        f"with extracted files in **`{data['run_directory']}/`**. Its absolute location "
        f"on the experiment machine is `{ROOT / data['archive']}`.",
        "",
        f"The ZIP contains `untrained.pt`, `trained.pt`, and all "
        f"**{data['periodic_checkpoint_count']} intermediate playback checkpoints**, "
        "saved every 25 completed episodes, together with the generated evidence. "
        "Large checkpoints and ZIPs are excluded from Git; the required small "
        "artifacts and every GIF are copied into `results/`. The local ZIP's SHA-256, "
        "integrity check, and checkpoint inventory are recorded in "
        "[verification.json](results/verification.json).",
        "",
        "The checkpoints support playback, not exact training resumption: they "
        "contain model weights but omit replay memory, optimizer state, and emulator "
        "state. Both earlier archives also remain local, as documented in the "
        "[preserved first report](experiments/original_100/README.md) and "
        "[preserved clipped-reward report](experiments/clipped_2000/README.md).",
        "",
        f"Public submission repository: [{REPOSITORY}]({REPOSITORY}).",
        "",
    ]
    readme = "\n".join(lines)
    # Prevent silently missing a produced GIF in a generated report.
    for gif_path in data["gifs"]:
        if f"]({gif_path})" not in readme:
            raise ValueError(f"README omits a generated GIF: {gif_path}")
    (ROOT / "README.md").write_text(readme)
    print(f"Wrote {ROOT / 'README.md'} using verified results for {data['run_id']}.")
    if not reviewed_run:
        print("Before publishing a new run, add observations from its actual GIFs.")


if __name__ == "__main__":
    main()
