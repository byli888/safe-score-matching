# Stabilize-avoid 2D: fixed reproduction results

This is the illustrative stabilize-avoid example, separate from the Quad2D tracking benchmark. All three reproduction training seeds (0, 1, 2) and all fixed A900k, B600k and B1M reporting points are retained; no seed or checkpoint was selected based on these results.

## Training endpoint: B1M

Each entry is the mean ± sample standard deviation (ddof = 1) across three training seeds. Each seed contributes 200 episodes on the specified panel, with a maximum horizon of 400 steps. Rates below are percentages. The SD measures variation across training seeds.

| Panel | Success (%) | Nominal collision (%) | Mean margin cost | Terminal six-state L2 |
|---|---:|---:|---:|---:|
| exact | 99.833 ± 0.289 | 0.000 ± 0.000 | 3.217 ± 1.917 | 0.173 ± 0.030 |
| vel_small | 100.000 ± 0.000 | 0.000 ± 0.000 | 3.748 ± 1.415 | 0.179 ± 0.020 |
| lower_region | 94.500 ± 2.500 | 5.500 ± 2.500 | 2.423 ± 0.905 | 0.366 ± 0.069 |

The [per-seed table](per_seed.csv) contains all 27 fixed point/panel rows, including counts, route counts and checkpoint hashes. The [across-seed table](over_seeds.csv) contains all nine fixed point/panel aggregates. CSV rates are fractions, not percentages.

A and B denote curriculum stages; both use the Stage-B learner mode. Each seed runs A to 1M steps, then starts B from the fixed A900k checkpoint and runs B for a further 1M steps. A900k is the pre-transition comparison, B600k the fixed comparison at the paper checkpoint's training step, and B1M the training endpoint. The full training allocation is 2M steps per seed; B1M inherits 900k A steps plus 1M B steps.

Reproduction seed 0 uses the original training implementation; seeds 1 and 2 use the example entry points provided in this repository.

## Selected paper checkpoint (separate reference)

The following rows re-evaluate the previously selected seed-0 B600k checkpoint. They are excluded from the three-seed reproduction aggregates. The lower-region reference panel was used to select the original checkpoint and is not held out. These are reference diagnostics.

| Panel | Episodes | Success | Nominal collision | Any margin violation | Success with zero margin cost | Mean margin cost | Terminal six-state L2 | Left / right / other |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| exact | 200 | 200/200 | 0/200 | 200/200 | 0/200 | 7.270 | 0.176 | 0 / 200 / 0 |
| vel_small | 200 | 200/200 | 0/200 | 200/200 | 0/200 | 5.995 | 0.177 | 83 / 117 / 0 |
| lower_region | 200 | 194/200 | 5/200 | 80/200 | 119/200 | 3.905 | 0.270 | 99 / 97 / 4 |

Reference checkpoint SHA-256: `0c665ad6d2ac40d04349bd9c39924f126f5f57518b1c68ead0b67f23c9689e6d`.

## Panel and metric interpretation

`exact` repeats the same full six-dimensional initial state [0, 0, -1.08, 0, 0, 0], in [x, vx, z, vz, theta, omega] order, using distinct per-episode policy keys. `vel_small` retains the same position, attitude and angular rate but perturbs vx uniformly in [-0.1, 0.1] and vz in [-0.05, 0.05]. Route diversity on velocity-perturbed starts does not demonstrate stochastic multimodality at an identical full state. `lower_region` uses the frozen wider, safely initialized lower-region reference distribution; see the example evaluation protocol for its exact sampling parameters.

Success requires terminal six-state L2 error < 0.5 and no nominal collision. Mean margin cost counts steps with h_phys > 0, using the safety margin; it differs from nominal collision. Success with zero margin cost and episodes with any margin violation remain separate per-seed counts. Terminal errors include failed episodes. Left/right uses the side of the first z ≥ 0.62 crossing (the center band is |x| < 0.08); other includes center crossings and episodes with no crossing. All 200 episodes remain in every panel's route denominator.

A-to-B changes include continued training, the reset mixture, learning rates, mirror replay and warm-start state. This comparison does not isolate the causal effect of any single component.
