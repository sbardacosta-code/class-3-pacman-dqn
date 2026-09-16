#!/usr/bin/env python3
"""Draft README.md from verified, completed experiment evidence.

Run after scripts/build_submission.py, then visually review the gameplay and
replace the explicitly marked neutral gameplay interpretation before publishing.
This script does not train, evaluate, alter results, or change the notebook.
"""

from __future__ import annotations

from datetime import timedelta
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REPOSITORY = "https://github.com/sbardacosta-code/class-3-pacman-dqn"


def number(value, places=1):
    value = float(value)
    return f"{int(value):,}" if value.is_integer() else f"{value:,.{places}f}"


def signed(value, places=1):
    return f"{float(value):+,.{places}f}"


def elapsed(seconds):
    return str(timedelta(seconds=round(float(seconds), 1)))


def main():
    data_path = ROOT / "results" / "report_data.json"
    data = json.loads(data_path.read_text())
    original = json.loads((ROOT / "experiments" / "original_100" /
                           "results" / "comparison.json").read_text())
    config = data["config"]
    summary = data["summary"]
    before = data["evaluation"]["before"]
    after = data["evaluation"]["after"]
    best = data["best_trained_game"]
    actual = summary["completed_episodes"]
    requested = config["episodes_requested"]
    duration = summary["elapsed_seconds_including_periodic_demos"]
    budget_seconds = config["training_time_limit_seconds"]
    rate = config["learning_rate"]
    original_scores = dict(zip(original["after"]["seeds"], original["after"]["scores"]))
    original_mean = original["after"]["mean"]
    score_delta = after["mean"] - before["mean"]
    original_delta = after["mean"] - original_mean
    improvement = "increased" if score_delta > 0 else "decreased" if score_delta < 0 else "was unchanged"
    changes = data["score_changes"]
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
    partial = data["partial_episode_decisions"]
    partial_note = (
        f"The final interrupted game contributed **{number(partial)} decisions** that are "
        "included in the total and final saved weights. It is not counted as a completed "
        "episode or included as a CSV row."
        if partial else
        "There are no partial-episode decisions: the CSV covers every recorded training decision."
    )
    learning_note = (
        "**No learning updates occurred.** This is an untrained/setup outcome, not evidence "
        "of a learned policy."
        if zero_updates else
        f"The network received **{number(summary['learning_updates'])} learning updates**. "
        "The verified trained weights differ from the untrained weights and remain finite. "
        "This establishes parameter learning, not mastery of the game."
    )
    if score_delta > 0:
        conclusion = (
            "The final policy earned a higher mean score on this fixed five-game comparison. "
            "That supports improvement under these evaluation conditions; it does not establish "
            "consistent performance across all possible games."
        )
    elif score_delta < 0:
        conclusion = (
            "The final policy earned a lower mean score on this fixed five-game comparison. "
            "This experiment did not improve the evaluation result despite its training updates; "
            "more training and the extensions did not guarantee better play."
        )
    else:
        conclusion = (
            "The final policy matched the untrained mean on this fixed five-game comparison. "
            "The experiment provides no mean-score evidence of improvement."
        )

    lines = [
        "# Class 3: Train a Ms. Pac-Man Agent",
        "",
        f"The extended **{config.get('algorithm', 'DQN')}** experiment completed "
        f"**{number(actual)} / {number(requested)} episodes**. Its mean score on the "
        f"five unchanged evaluation games {improvement} from **{number(before['mean'])} "
        f"to {number(after['mean'])}** ({signed(score_delta)} points). The original "
        f"100-episode DQN achieved **{number(original_mean)}**, so the new final mean is "
        f"**{signed(original_delta)} points** relative to that first experiment.",
        "",
        f"The leaderboard score for this final saved model is **{number(after['mean'])}**. "
        "The table below reports every game. The submitted model is "
        "the final model from this run, not a checkpoint selected by its best evaluation score.",
        "",
        "This repository contains the [final executed notebook](pacman_dqn.ipynb), with "
        "all 28 code cells executed in order and all outputs retained. It extends the "
        "[supplied Pac-Man project](https://github.com/pepealonso95/pacman-dqn). The "
        "[original 100-episode experiment](experiments/original_100/README.md), "
        "[original notebook](experiments/original_100/pacman_dqn.ipynb), and its evidence "
        "remain available for comparison.",
        "",
        "## Choices and expectation recorded before training",
        "",
        "| Setting | Choice | Reason |",
        "| --- | ---: | --- |",
        f"| Exploration | {config['exploration']:.2f} | Keep trying alternatives while reducing "
        "the disruptions from the first run's 20% random moves. |",
        f"| Episode budget | {number(requested)} | Collect substantially more experience, "
        "subject to the two-hour training limit and three-hour overall project budget. |",
        f"| Learning rate | {rate:.4f} | Retain the conservative update size from the first "
        "run while extending training and memory. |",
        "",
        "The [before-training plan](experiment_plan.md) expected a larger experience budget "
        "and replay memory to improve point collection, with no guarantee of a higher score. "
        "Exploration was 100% for the first 1,000 decisions, then stayed constant at "
        f"**{config['exploration']:.0%}**. No exploration-decay schedule was introduced.",
        "",
        "Two extensions to the supplied DQN were explicitly chosen and documented:",
        "",
        "- **Double DQN:** the online network chooses the next action; the target network "
        "estimates that chosen action's value. Separating selection from valuation aims "
        "to reduce overoptimistic action-value estimates. "
        "[Double DQN paper](https://arxiv.org/abs/1509.06461).",
        f"- **Replay capacity {number(config['replay_capacity'])}**, increased from 5,000: "
        "retain a wider range of past experiences for random training batches.",
        "",
        f"An operational **{number(budget_seconds / 3600)}-hour training cap** saves the current "
        "model through the notebook's interruption path, then allows final evaluation and "
        "archiving. This cap does not change the per-game evaluation time limit. The network "
        "architecture, observations, actions, game setup, reward clipping, training seed, "
        "and evaluation settings remain unchanged. This run starts from fresh weights and "
        "fresh replay memory; it does not resume the first experiment.",
        "",
        "The assignment allows explained changes to other hyperparameters and optional "
        "custom agents. Replay capacity is an additional hyperparameter; Double DQN is "
        "disclosed as an algorithm extension to the supplied implementation.",
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
        f"| Training elapsed time | {duration:,.3f} seconds ({elapsed(duration)}), including periodic demos |",
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
        "100-episode result used these same evaluation settings.",
        "",
        "| Game | Seed | New-run untrained baseline | Original 100-episode DQN | New final trained agent | Change from new baseline |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for i, seed in enumerate(before["seeds"]):
        lines.append(f"| {i + 1} | {seed} | {number(before['scores'][i])} | "
                     f"{number(original_scores[seed])} | {number(after['scores'][i])} | "
                     f"{signed(changes[i], 0)} |")
    lines += [
        f"| **Mean** | | **{number(before['mean'])}** | **{number(original_mean)}** | "
        f"**{number(after['mean'])}** | **{signed(score_delta)}** |",
        "",
        f"Relative to the new baseline, **{data['improved_games']} games improved, "
        f"{data['worsened_games']} worsened, and {data['unchanged_games']} were unchanged**. "
        f"Time-limited games: **{data['time_limited_before']} before / "
        f"{data['time_limited_after']} after**.",
        "",
        "Machine-readable evidence: [all new comparison scores](results/comparison.json), "
        "[baseline.json](results/baseline.json), and "
        "[original comparison](experiments/original_100/results/comparison.json).",
        "",
        conclusion,
        "",
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
        "<!-- VISUAL_REVIEW_REQUIRED: Replace the neutral paragraph below after watching "
        "the actual GIFs; make only concrete observations supported by those recordings. -->",
        "**Gameplay interpretation:** The untrained and trained recordings are short "
        "excerpts under the same game conditions. The trained excerpt is deliberately "
        "selected from the highest-scoring final evaluation game, so it cannot establish "
        "typical play by itself. The full five-game scores provide the quantitative "
        "comparison; the clips alone do not establish maze mastery, reliable ghost "
        "avoidance, or an understanding of the scoreboard digits.",
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
        "measure of gameplay quality. Lower loss does not guarantee better play.",
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
        "Training clips each decision's reward to [-1, 1], while all reported scores "
        "retain the original game points. The network learns estimates of immediate "
        "and discounted future reward for each action using replayed experiences.",
        "",
        "**What the evidence supports:** " + learning_note + " " + conclusion,
        "",
        "**Observed limitation of this experiment:** only one training seed and five "
        "evaluation games were used. Individual game outcomes and a selected short "
        "GIF do not establish dependable performance on unseen games. Episode budget, "
        "exploration, replay capacity, and the learning rule all changed together "
        "relative to the first run, so their individual contributions cannot be isolated.",
        "",
        "**Next experiment:** change only the learning rate from **0.0001 to 0.00005**, "
        "keeping Double DQN, replay capacity 20,000, exploration 0.10, requested episode "
        "budget 2,000, the two-hour cap, and all evaluation settings fixed. Smaller "
        "updates may make learned action values less volatile; this is a hypothesis "
        "to test, not an established improvement. Record the actual decisions and "
        "updates again because the time cap may yield different completed budgets. "
        "This next experiment has not been performed.",
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
        "state. The original run's archive also remains local, as documented in the "
        "[preserved first report](experiments/original_100/README.md).",
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
    print("Before publishing: replace VISUAL_REVIEW_REQUIRED with observations from actual GIF review.")


if __name__ == "__main__":
    main()
