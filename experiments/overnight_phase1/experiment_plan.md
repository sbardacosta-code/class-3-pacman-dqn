# Pre-training plan: improve consistent Ms. Pac-Man scoring

The student requested a stronger attempt to reach consistent 3,000+ scores after the scaled-reward model scored 1,450 on the classroom games. This plan is written before the new training and its validation results. The prior three-hour overall budget remains in effect; training has a two-hour wall-clock cap, leaving time for preparation, final evaluation, saving, and publication. A temporary process-level macOS sleep assertion keeps this run active; no permanent power settings are changed.

## Three choices

| Setting | Choice | Reason |
| --- | ---: | --- |
| Exploration | 0.10 | Keep trying alternatives around the inherited policy; retain constant exploration after the first 1,000 random new decisions. |
| Additional episode budget | 2,500 | Give the existing visual features more practice within the two-hour cap. Report actual completed games and any partial games. |
| Learning rate | 0.00005 | Use smaller updates while fine-tuning learned weights with prioritized, longer-horizon targets. |

## Starting model and disclosed extensions

The previous scaled-reward checkpoint is the starting model because its reward units match this experiment. It already received **1,069 completed episodes, 744,500 decisions, and 185,875 recorded updates**. It is publicly available in the [starting-checkpoint release](https://github.com/sbardacosta-code/class-3-pacman-dqn/releases/tag/warmstart-scaled-1069); the notebook checks SHA-256 `b181f817b3f37a254addca3edb2da8ddb68abf41a3fe7fc64014fff4f0d0869c`. The optimizer and replay start fresh. This is warm-start training, not an exact resumption, and inherited training must not be represented as work performed in this new run.

The genuine fresh, untrained baseline is still evaluated first. A separately labeled five-game evaluation then verifies the inherited starting model. Previous classroom scores were 1,430, 1,050, 1,860, 1,930, 980 (mean 1,450); the earlier clipped-reward model scored 1,582. Neither is a random-action baseline.

The supplied convolutional network and Double DQN remain unchanged. This run adds:

- **Three-step returns:** propagate the next three rewards back to earlier decisions. Store the actual four-frame future stack, flush tails at game boundaries, suppress bootstrapping only at true game over, and retain bootstrapping at a time limit.
- **Prioritized replay:** capacity 50,000 transitions, exponent alpha 0.5, importance correction beta increasing from 0.4 to 1.0 over 2,000,000 additional decisions. Full screen stacks use approximately 2.63 GiB. A sum tree avoids scanning the complete memory for each sample.
- **A small life-loss penalty:** training reward is `raw game points / 100 - 0.5 * lives lost`. Thus a death costs the equivalent of 50 raw points. It may encourage safer play but can also discourage worthwhile risks. A life loss does not terminate the training target or the full game. All evaluation scores remain raw game points.
- **Replay fill:** collect 10,000 new decisions before updating the inherited network. Only the first 1,000 decisions are all-random; exploration thereafter remains 10%.
- **Four training environments:** batch action predictions to reduce MPS synchronization overhead. A shared learner still updates once per four aggregate decisions. Each game retains the original 3,000-decision cap. Seeds continue beyond the source run: `42 + 1069 + additional_training_episode_id`.

An actual-device benchmark favored MPS with four environments. The architecture is retained so all inherited weights remain compatible. No larger network, distributional head, or privileged game-state input is added. Lives are read only to calculate the disclosed reward; observations remain four grayscale images.

## Selection and evaluation fixed before results

The class evaluation remains identical: seeds **101, 202, 303, 404, 505**, exploration **0.05**, and **3,000 decisions per game**, with exactly the original environment, actions, sticky-action setting, and preprocessing. The selected checkpoint is evaluated once on these five games after training. All five scores and both fresh-baseline/final means are reported.

For model selection, use **ten separate validation seeds, 20001–20010**, at the start, every 250 completed additional games, and once at the end. Rank candidates by **mean raw score minus 0.5 times population standard deviation**, retaining the earlier candidate on ties. This heuristic penalizes uneven scores; it is not a confidence bound or a guarantee of generalization. Including the inherited model avoids automatically replacing it with a poorer validation candidate. Save every candidate's ten scores, metric, and decision/update budget in `model_selection.json`. Preserve the last learner weights separately from the selected weights. Periodic classroom-seed gameplay samples are illustrations only and do not select a checkpoint.

Finally, evaluate the selected model and the inherited model on **20 new additional seeds, 30001–30020**, disjoint from training, validation, and classroom seeds. These results do not change the leaderboard mean. Report every score, mean, median, minimum, maximum, and the fraction at or above 3,000. A predeclared operational target is **classroom mean at least 3,000 and at least 18/20 additional games at or above 3,000**. Neither a high best-game score nor a single lucky classroom result meets this definition.

## Expectations, evidence, and limitations

The expectation is improved sample efficiency, safer decisions, and higher typical scores. Several changes are bundled to pursue performance within the available time; this is not an isolated causal test of the death penalty, prioritized replay, or extra training. Failure to improve, a selected starting model, a time-limit stop, or inconsistent scores will be reported honestly.

Keep all 28 notebook code cells executed in order and retain their outputs. Save GIFs and playback checkpoints every 25 completed games, the training dashboard, all configuration/evaluation/training records, and a full local ZIP. A separate learner snapshot saves the optimizer, target, replay, and random states, but not emulator state; it is not exact mid-game resumption. With parallel games, completed CSV step counts plus partial-game decisions must reconcile to the aggregate total. The loss curve averages shared optimizer updates during each game's lifetime and is not loss attributable solely to that game.

Preserve [the previous scaled run](../scaled_1069/README.md), [the clipped run](../clipped_2000/README.md), and [the original run](../original_100/README.md). Publish the new executed notebook and evidence without hiding a regression. The actual outcomes determine the report; do not invent successful strategies or claim that the network understands game entities from its score alone.

## Sources informing the design

[Rainbow](https://arxiv.org/abs/1710.02298) identifies multi-step learning and prioritization as useful components in its Atari experiments. [Prioritized Experience Replay](https://arxiv.org/abs/1511.05952) describes error-based sampling and importance correction. Our choices are adaptations, not a reproduction of those large-compute benchmarks. [Revisiting the Arcade Learning Environment](https://arxiv.org/abs/1709.06009) motivates caution around artificial life-loss termination and consistent evaluation. None of these sources establishes that this laptop budget guarantees 3,000-point play.
