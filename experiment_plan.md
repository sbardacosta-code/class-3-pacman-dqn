# Before-training plan

The student selected the suggested settings before execution:

- Exploration: **0.20**. Continue trying alternative moves after the fixed 1,000-decision random warm-up, while using the learned action values most of the time.
- Episode budget: **100**. Allow substantially more experience than a five-game setup check while keeping the experiment manageable locally.
- Learning rate: **0.0001**. Use conservative updates to reduce instability in the network's value estimates.

Expectation recorded before training: modest improvement in collecting points, with uneven results across games. One hundred episodes may still be insufficient for reliable ghost avoidance. This is a hypothesis, not an observed result.

Only the three requested settings are selected; their values already match the supplied notebook. All other notebook code and evaluation settings remain unchanged. Evaluation uses seeds 101, 202, 303, 404, and 505, exploration 0.05, and a 3,000-decision limit both before and after training. The baseline is an untrained network.
