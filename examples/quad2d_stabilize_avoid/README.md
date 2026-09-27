# Planar stabilize-avoid illustration

This is the six-state obstacle-avoidance illustration associated with Figure 1, separate from the `quad2d.yaml` circle-tracking benchmark. The example provides two training configurations using the shared SSM learner.

## Train

From the repository root, using the custom-task environment:

```bash
bash examples/quad2d_stabilize_avoid/train.sh
```

Equivalently:

```bash
python train.py --config configs/quad2d_stabilize_stage_a.yaml
python train.py --config configs/quad2d_stabilize_stage_b.yaml
```

Stage A and Stage B name the two curriculum phases; both use the learner setting `qh_stage_mode=stage_b`. Stage A runs 1,000,000 environment steps. Stage B starts from the fixed Stage-A checkpoint at 900,000 steps and runs another 1,000,000 steps. This consumes 2,000,000 training steps in total. Stage B copies network parameters and observation statistics, and initializes fresh optimizers, replay, and random keys; it is a warm start, not a full-state resume. Its `init_checkpoint` path is relative to the repository root.

The shell wrapper runs the YAML defaults (seed 0) and does not forward override arguments. For another seed, give both stages distinct run names and point Stage B to that seed's Stage-A checkpoint:

```bash
python train.py --config configs/quad2d_stabilize_stage_a.yaml \
  --set seed=1 --set run_name=quad2d_stabilize_stage_a_s1
python train.py --config configs/quad2d_stabilize_stage_b.yaml \
  --set seed=1 --set run_name=quad2d_stabilize_stage_b_s1 \
  --set init_checkpoint=checkpoints/quad2d_stabilize_stage_a_s1/step_900000.pkl
```

Use the matching Stage-B run name in the evaluation checkpoint path and a distinct output name for each seed.

The two reset mixtures use the same `simple_under3` reset implementation. Fractions sum to one:

| Reset component | Stage A | Stage B |
|---|---:|---:|
| Below central bar | 0.15 | 0.24 |
| Below left block | 0.15 | 0.08 |
| Below right block | 0.15 | 0.08 |
| Inner gaps | 0.20 | 0.08 |
| Lower workspace | 0.25 | 0.20 |
| Near goal | 0.10 | 0.12 |
| Side of goal | 0.00 | 0.00 |
| Near-start fork | 0.00 | 0.20 |

The fork component samples a side uniformly. On that side, `|x|` is uniform in `[0.05, 0.35]`, `z` in `[-1.00, -0.30]`, outward `|vx|` in `[0.10, 0.55]`, `vz` in `[-0.05, 0.45]`, angle in `[-0.12, 0.12]`, and angular velocity in `[-0.15, 0.15]`. These are two fixed mixtures, not an annealing schedule within either run. Stage B also uses mirror replay augmentation and learning rates of `1e-4`; Stage A uses the trainer defaults of `3e-4`. The actor's cosine schedule spans 2,000,000 updates in each stage.

Warmup uses an unseeded action-space sampler. A fixed training seed therefore does not specify a bit-for-bit training replay. The evaluation initial states and policy seeds below are explicit.

## Evaluate one checkpoint

```bash
python evaluate.py --config evaluation/quad2d_stabilize.yaml \
  --checkpoint checkpoints/quad2d_stabilize_stage_b_s0/step_1000000.pkl \
  --out results/stabilize_stage_b_1000000.json
```

This reports the Stage-B endpoint. Use Stage-B `step_600000.pkl` to compare at the selected paper checkpoint's training step, and Stage-A `step_900000.pkl` for the state before the curriculum change. The evaluator accepts one locally trained or trusted checkpoint, does not scan or rank checkpoints, and writes JSON plus an adjacent NPZ of all trajectories. It runs on CPU through the common evaluation launcher.

The protocol specifies 200 episodes in each of three panels, with a 400-step horizon:

- `exact`: every episode starts at `[x, vx, z, vz, theta, omega] = [0, 0, -1.08, 0, 0, 0]`, with a distinct policy key.
- `vel_small`: the same position, angle, and angular velocity, with `vx` uniform in `[-0.10, 0.10]` and `vz` in `[-0.05, 0.05]`. The YAML records the initial-state seed and the policy key construction.
- `lower_region`: the retained `lowz055_hrej` sampler draws broad lower-workspace states up to `z = 0.55`, accepts only states satisfying `h_phys < init_h_threshold` (0 in this recipe) and outside the goal region, and uses initial-state seed 20260501 and policy seed 20260502. Its velocity bounds come from the checkpoint configuration. This reference panel was used to select the original checkpoint; it is **not held out**.

The workspace is `x ∈ [-2.78, 2.78]`, `z ∈ [-1.55, 3.10]`, with goal `(0, 2.45)`. Binary cost counts transitions where the signed constraint `h_phys > 0`: obstacle rectangles are inflated by 0.10 and the workspace is inset by 0.10. Actual collision and termination use the nominal geometry. A collision-free trajectory can therefore have positive margin cost. For this recipe, success means terminal six-state Euclidean error below 0.50 **and no nominal collision**; it does not imply zero margin cost. JSON reports success jointly with zero cost and jointly with no collision, separately.

Saved route labels have the form `side_suffix`. The side is determined at the first crossing of `z >= 0.62`: `|x| < 0.08` is `center`, otherwise negative x is `left` and positive x is `right`. The suffix uses the largest `|x|` among trajectory states with `z < 1.12` (or `|x|` at the crossing if there are no such states): `outer` if it exceeds 2.06, `inner` if it exceeds 1.00, and `center` otherwise. For example, `left_center` denotes the left side with maximum `|x| <= 1.00` in that region. Episodes that never cross remain `none`. Route counts retain all episodes, including failures. The exact panel tests action randomness at the same state; the perturbed panel tests nearby initial states. A left/right split in the latter does not establish multimodality conditional on one identical initial state.

## Fixed-step reproduction

Three training seeds (0, 1, 2) completed both 1M-step phases: 6M training steps in total. All fixed A900k, B600k and B1M checkpoints were evaluated on the three 200-episode reference panels. These retained panels are not an unseen test set. The [complete results](results/RESULTS.md) include all 27 per-seed rows, nine across-seed aggregates and a separate selected-checkpoint reference.

At the B1M endpoint, success is 99.8 ± 0.3% on `exact`, 100.0 ± 0.0% on `vel_small`, and 94.5 ± 2.5% on `lower_region` (mean ± sample SD across training seeds). The first two panels have no nominal collisions; lower-region collision rate is 5.5 ± 2.5%. Margin cost remains positive, so these are not zero-cost safety results.

Route diversity depends on the checkpoint and seed. At the fixed B600k comparison, seed 0 takes 127 left and 73 right routes from the identical full initial state, with all 200 episodes successful and zero margin cost; seeds 1 and 2 take only right routes. At B1M, each seed takes only one side on `exact`, while all three split routes on `vel_small`. Continued training therefore does not preserve every earlier behavior or monotonically improve safety. The [reproduction trajectory plot](../../docs/assets/stabilize_avoid.png) ([PDF](../../docs/assets/stabilize_avoid.pdf)) uses the pre-specified seed-0 B1M endpoint. It is separate from the paper's Figure 1 shown in the repository README.

## Plot the reproduction

After evaluating one checkpoint, install the optional plotting dependency in the custom-task environment and redraw its saved trajectories:

```bash
python -m pip install matplotlib
python examples/quad2d_stabilize_avoid/plot.py \
  --result results/stabilize_stage_b_1000000.json \
  --out-prefix figures/stabilize_stage_b_1000000
```

This writes PNG, PDF and a Markdown caption using all 200 episodes for each panel's statistics. The figure always shows episodes 0, 28, 57, 85, 114, 142, 171 and 199 from `vel_small` and `exact`, including failures, without smoothing or selection. The lower-region panel remains in the evaluation output and complete results; it is not displayed in this two-panel figure. Existing files are not overwritten. No checkpoint or training process is needed to redraw the saved JSON and adjacent NPZ.

## Relation to the paper

The repository README displays Figure 1 from the paper. The paper figure's caption defines success as entering the green goal region. The reproduction here uses the terminal six-state error and collision criterion above; the success rates are not interchangeable. In the paper figure, the background and dashed contour show the learned HJ value and its zero level set, whereas the reproduction plot shows the geometric 0.10 m safety buffer.

The Figure-1 checkpoint was selected at Stage B 600,000 steps after warm-starting from Stage A 900,000 steps. The reproduction keeps those reporting steps fixed instead of selecting a checkpoint from its results.

The task uses this repository's SSM learner and accelerated rollout. The repository license and third-party notices apply; pretrained checkpoints and an automatic checkpoint selector are not included. This is an illustration, separate from the paper's quantitative benchmarks.
