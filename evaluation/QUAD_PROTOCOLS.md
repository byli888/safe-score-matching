# Quad behavior evaluation

Use `quad2d.yaml` or `quad3d.yaml` with one trusted training checkpoint. Relative data paths resolve from the protocol YAML directory. The checkpoint stores all nine network parameter sets and observation normalization; evaluation restores them using this repository's learner. Optimizer state is not part of these inference checkpoints. The recorded step comes from the checkpoint filename.

The fixed protocol uses 100 Quad2D starts for 360 steps and 500 Quad3D starts for 500 steps, with policy seed 20260504. Episodes are evaluated as one batched JAX rollout. Actions are raw stochastic DDPM samples at the saved temperature. There is no best-of-N selection, learned-safe initial-state filtering or checkpoint search. Crashed episodes remain in the results; after termination, placeholders are not scored.

- Quad2D tracking error: for each episode, the sum of squared error over all six state coordinates, averaged over its scored transitions. The summary averages these episode values.
- Quad3D terminal error: L1 norm over all nine state coordinates relative to the saved target. L2 is reported under a separate name. Failed episodes retain their last state.
- Episode cost: number of scored transitions with h>0. `any_violation` is whether an episode has any such transition. Crash is the environment termination event and is reported separately.
- JSON contains every episode's reward, cost, scored length, crash/violation indicator and task error, plus summary means and episode standard deviations (ddof=0), hashes and actual imported source paths.

These are behavioral metrics. They do not reproduce the paper's safety-classification precision, coverage and false-safe figure, whose original classification population has not been recovered. Episode violation must not be relabelled classifier false-safe rate. No classification diagnostic is included here.

For a fixed-budget comparison, decide reporting checkpoints before inspecting evaluations. This entry accepts one file and provides no ranking or checkpoint-selection option. Cross-runtime numerical differences are possible; compare checkpoints with the same interpreter and protocol.
