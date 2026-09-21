# Before-training plan: preserve reward magnitude

Recorded before the new training run on September 20, 2026. The student approved the proposed first experiment: test scaled point rewards alone before considering a life-loss penalty.

## Hypothesis and the single changed training setting

The previous Double DQN stored `clip(raw_points, -1, 1)` for learning. Both a 10-point gain and a 200-point gain therefore became +1. This experiment instead stores **`raw_points * 0.01`, with no clipping**. Those gains now become 0.1 and 2.0, preserving the game's relative point values. Larger rewards may encourage more valuable point-collection behavior. Removing clipping may also make learning less stable, so improvement is a hypothesis, not a promised outcome.

**No death penalty is added.** It would change a second aspect of the reward and make this comparison harder to interpret. The proposed later experiment would test a small life-loss penalty separately, if the evidence supports trying it.

The agent still receives numeric rewards from the emulator; it does not need to read score pixels. Raw training and evaluation scores remain the original game points. The reward transformation is applied once, when storing experiences for learning.

## Settings retained from the comparison run

| Setting | Choice | Why retain it? |
| --- | ---: | --- |
| Exploration | 0.10 after 1,000 random warm-up decisions | Keep the same balance of alternative moves and learned action preferences. |
| Episode budget | 2,000 | Match the previous requested budget within the same time constraint. |
| Learning rate | 0.0001 | Isolate reward scaling rather than also change update size. |
| Algorithm | Double DQN | Preserve online action selection and target-network valuation. |
| Replay capacity | 20,000 | Keep memory capacity fixed. |
| Training cap | 7,200 seconds | Reuse the two-hour training limit, leaving time within the prior three-hour overall budget for setup, evaluation, and publishing. |

All other training settings, network architecture, observations, actions, environment configuration, and the training seed remain unchanged. Run All starts **fresh weights, optimizer, and replay**, using the same seed; this is not an exact resumption of the previous model. GPU nondeterminism and different trajectories can still affect results. If the cap stops the run early, report completed episodes, decisions, updates, and any partial episode honestly; unequal realized training budgets limit the comparison.

## Evaluation and success criteria

The classroom evaluation stays byte-for-byte unchanged: seeds **101, 202, 303, 404, 505**, exploration **0.05**, and **3,000 decisions per game**, using the same preprocessing and environment. Obtain a new untrained baseline, then evaluate the final saved model once on those five games. Preserve all five raw scores, their mean, minimum, and the fraction reaching 3,000 points. Do not choose a checkpoint using the best of repeated official evaluations.

The previous clipped experiment achieved scores **2,310, 2,530, 1,000, 1,090, 980**, mean **1,582**, with **0/5** games at 3,000. The new run tests progress toward the student's goal of consistent 3,000+ scores. A high average alone does not establish consistency. Additional validation on separate seeds can test generalization after training; it must remain separate from the unchanged classroom evaluation.

Before seeing results, the expectation is better recognition of valuable point gains, potentially improving raw score. Neither reward scaling nor another 2,000 episodes guarantees 3,000+. Preserve and report regressions or instability.

## Evidence and preservation

The previous 2,000-episode clipped experiment, executed notebook, and evidence are preserved under [experiments/clipped_2000](experiments/clipped_2000/README.md); its full local ZIP remains unchanged. The original 100-episode experiment also remains available. Their historical plans record the reasoning used before those runs.

The new notebook will execute all cells in order and retain all outputs. Save each intermediate GIF/checkpoint at 25-episode intervals, final evaluation, dashboard, settings, package/hardware information, CSV/JSON evidence, and full local ZIP. Publish the final executed notebook, report, and selected evidence, keeping every run's outcome visible.

## Research informing the hypothesis

Hado van Hasselt's [Ms. Pac-Man reward-clipping comparison](https://hadovanhasselt.com/2016/08/17/atari-videos/) showed different behavior with unclipped rewards and adaptive target normalization. Our simple fixed scale of 0.01 is a distinct, smaller experiment, not a reproduction of Pop-Art or a proven best setting. The assignment permits explained extensions; this training reward change is explicitly disclosed, while evaluation remains unchanged.

## Additional validation plan (recorded during initial training, before final results)

Evaluate both the new final model and the preserved clipped-reward final model once on **20 separate seeds, 10001 through 10020**, using the same evaluation function, 5% exploration, and 3,000-decision limit. These seeds are separate from the classroom seeds and this run's training-reset seeds. Report every raw score, mean, median, minimum, and fraction at or above 3,000. This comparison is additional evidence about consistency; it does not replace or change the classroom's five-game leaderboard result, and it will not select a checkpoint. Save its JSON/CSV separately from the notebook-generated run ZIP.

## Post-run execution note

An apparent host pause advanced wall-clock time by several hours while the notebook's training timer advanced much less. After detecting that the original three-hour wall-clock window had passed, training was interrupted once through its existing save handler. The configured 7,200-second training cap was **not** reached. The run recorded **1,069 completed episodes, 744,500 decisions, 185,875 learning updates, and 4,587.930 seconds** on its training timer; final evaluation and ZIP creation then finished normally. The interruption occurred at a scheduled update boundary, so the report retains the recorded update counter rather than inferring one extra completed update. This smaller actual training budget prevents a clean causal comparison of reward transformations with the earlier 2,000-episode run.
