# Before-training plan: budgeted Double DQN experiment

The student approved a new experiment with a maximum of three hours total. Training is limited to two hours, reserving one hour for implementation, evaluation, saving, inspection, and publishing. The original 100-episode experiment is preserved under `experiments/original_100/`, and its full original archive remains under `pacman_runs/`.

## Selected settings

- Exploration: **0.10**, constant after the unchanged 1,000-decision random warm-up. This tests whether fewer random disruptions allow useful behavior to persist.
- Episodes: **2,000 maximum**. More experience than the original run, with an operational wall-clock stop after two hours.
- Learning rate: **0.0001**, unchanged to avoid simultaneously changing the update magnitude.

## Disclosed extensions

- **Double DQN:** the online network selects the next action and the target network evaluates that action. This is intended to reduce overoptimistic value estimates. It is an algorithm extension of the supplied DQN, not a hyperparameter or a new architecture.
- **Replay capacity 20,000** rather than 5,000: retain a wider range of recent experiences. Pixel storage is approximately 673 MiB, plus model, batch, and Python overhead.
- **Two-hour training cap:** checked before each training decision; on expiry a single `KeyboardInterrupt` follows the notebook's existing save-and-evaluate path. The final summary records an interrupted run if the cap is reached. The last partial episode can contribute updates but is not counted as completed.

The assignment explicitly allows explained changes to other hyperparameters and describes building an agent as optional. We interpret that optional extension allowance as permitting Double DQN, while clearly disclosing the modification rather than claiming the supplied implementation is unchanged.

## Expectation and comparison

Before training, expect that more experience, reduced random moves, and broader replay may improve consistent point collection and reduce stalls. Improvement is uncertain. Because multiple training factors change together, this experiment cannot identify which factor caused any difference. No promise of a particular score is made.

The evaluation remains exactly the original function and settings: seeds 101, 202, 303, 404, and 505; exploration 0.05; 3,000 decisions per game; unchanged environment, preprocessing, reward reporting, and GIF selection. The new run starts from a fresh untrained network, obtains its own baseline, and evaluates its final saved checkpoint. It does not select a checkpoint using the five official scores.

All periodic GIFs and checkpoints will be retained. The final executed notebook and selected evidence will replace the root submission, while the first experiment remains accessible. Any regression will be reported.

---

## Original 100-episode plan (preserved)

The student selected the suggested settings before execution:

- Exploration: **0.20**. Continue trying alternative moves after the fixed 1,000-decision random warm-up, while using the learned action values most of the time.
- Episode budget: **100**. Allow substantially more experience than a five-game setup check while keeping the experiment manageable locally.
- Learning rate: **0.0001**. Use conservative updates to reduce instability in the network's value estimates.

Expectation recorded before training: modest improvement in collecting points, with uneven results across games. One hundred episodes may still be insufficient for reliable ghost avoidance. This is a hypothesis, not an observed result.

Only the three requested settings are selected; their values already match the supplied notebook. All other notebook code and evaluation settings remain unchanged. Evaluation uses seeds 101, 202, 303, 404, and 505, exploration 0.05, and a 3,000-decision limit both before and after training. The baseline is an untrained network.
