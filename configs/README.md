# Training recipes and reset protocols

The task YAML files are executable recipes: `train.py` passes their `arguments`
to the selected trainer. The bundled environment code also defines fixed task
constants. This reference makes the custom tasks' reset distributions and
curricula explicit; it describes the supplied training recipes.

| Recipe | Training reset protocol | Episode limit |
|---|---|---|
| [Quad2D](quad2d.yaml) | Fixed uniform box; no curriculum | 360 steps, dt = 1/60 |
| [Quad3D](quad3d.yaml) | Safe, non-goal reset box with an annealed outer-xy mixture | 500 steps, dt = 0.01 |
| [F16 Stage A](f16.yaml) | Tail25: fixed 25% theta-tail proxy mixture with the `ours` reset box and `v6` geometry | 640 steps, dt = 0.05 |
| [F16 Stage B](f16_stage_b.yaml) | Fixed 80/10/10 box mixture with `efppo_train` resets and `efppo_plus_p` geometry | 640 steps, dt = 0.05 |

Training reset distributions differ from the fixed evaluation protocols in
[the evaluation guide](../EVALUATION.md).

## Quad2D tracking

State order is `[x, vx, z, vz, theta, omega]`. Each reset samples independent
uniform coordinates with the following bounds, without rejecting unsafe states:

```text
low  = [-1.5, -1.0, 0.25, -1.5, -0.2, -0.1]
high = [ 1.5,  1.0, 1.75,  1.5,  0.2,  0.1]
```

There is no curriculum or reset annealing. The reference is a circle centered at
`(0, 1)`, radius `1`, with 360 points. At reset, the reference index is the closest circle point to the sampled
position; each step advances that index by one. The recipe enables reference velocity and uses
`Q = diag(10, 1, 10, 1, 0.2, 0.2)` and `R = 0.001 I`; the action penalty is relative
to the physical normalized hover action `[0.5, 0.5]`.

The active SDF mode is `baseline` (the trainer default):
`h = max(0.5-z, z-1.5, |x|-2, |z|-3)`, with cost `1[h > 0]`.
An episode terminates only when `|x| > 2` or `|z| > 3`, and otherwise truncates at
360 steps. Thus a safety violation is not automatically a termination. Initial
heights outside `[0.5, 1.5]` are deliberately possible under this reset box.

Source: [environment](../custom/ssm/tasks/quad2d_tracking/env.py), especially
`reset`, `_create_waypoints`, `_sdf`, and `step`;
[trainer](../custom/ssm/tasks/quad2d_tracking/train.py).

## Quad3D stabilization

State order is `[px, py, pz, vx, vy, vz, phi, theta, psi]`. The base uniform box is:

```text
low  = [-1.5, -1.5, -1.0, -0.5, -0.5, -0.5, -0.2, -0.2, -0.2]
high = [ 1.5,  1.5, -0.05, 0.5,  0.5,  0.5,  0.2,  0.2,  0.2]
```

Accepted resets satisfy `h < 0` and lie outside the goal set
`||state_9D|| <= 0.3 AND safe`. The `outer_xy` branch additionally requires
`sqrt(px² + py²) >= 1.5`. Its probability is specified in the YAML:

- Through step 200,000: `0.50`.
- From 200,000 to 500,000: linearly anneal from `0.50` to `0.05`.
- From 500,000 onward: `0.05`.

This schedule changes the reset-branch probability, while keeping the base box
and task geometry fixed. Each branch tries at most 1,000 proposals. An exhausted
outer branch falls back to the ordinary safe, non-goal branch; exhaustion there
raises an error.

The active SDF mode is `raw` (the trainer default):
`h = max(pz, ||state_9D|| - 3)`. Termination occurs at `pz >= 0.3` or
`||state_9D|| >= 3.5`; the 500-step limit is a truncation. The norm is over all nine
state coordinates. The YAML uses L1 regulation to the origin with weights
`[15, 15, 15, 6, 6, 6, 0.5, 0.5, 2]` and no action penalty; legacy reward and
axis-split reward overrides are disabled by their trainer defaults.

Both Quad environments update Welford observation statistics online and clip
normalized observations to `[-5, 5]`. Their trainers bootstrap through time-limit
truncations, but mask genuine terminal transitions.

Source: [environment](../custom/ssm/tasks/quad3d/env.py), especially
`current_init_curriculum_frac`, `_sample_safe_non_goal_start`, `reset`, and `step`;
[trainer](../custom/ssm/tasks/quad3d/train.py).

## F16 stabilization: Tail25, Stage-A

The YAML explicitly selects `reset_box_mode: ours`, `region_profile: v6`,
`reset_requires_safe: true`, and `init_curriculum_mode: theta_tail_proxy`.
The `qh_stage_mode: stage_a` setting is fixed throughout this recipe. Its gate
uses one policy action; it does not switch automatically to Stage B.

The ordinary proposal samples each coordinate uniformly from `_BASE_BOX_V5`:

| Coordinates | Bounds (simulator coordinates; angles in radians) |
|---|---|
| VT | `[150, 550]` |
| alpha, beta | `[-0.025, 0.35]`, `[-0.15, 0.15]` |
| phi, theta, psi | `[-pi/3, pi/3]`, `[-0.60, 0.60]`, `[-1e-4, 1e-4]` |
| P, Q, R | `[-1, 1]`, `[-0.5, 0.5]`, `[-2pi, 2pi]` |
| PN, PE, altitude H | `[-1000, 1000]`, `[-100, 100]`, `[150, 600]` |
| POW | `[0, 10]` |
| NZINT, PSINT, NYRINT | each `[-2, 2]` |

Each training reset requests the **theta-tail proxy with probability 0.25** and
the ordinary box with probability 0.75. The proxy starts from
`nominal_state_v5()` and changes only:

- theta: 80% `Uniform(0.4, 1.0)`, 20% `Uniform(-1.2, -0.4)`;
- altitude H: `Uniform(50, 1000)`;
- alpha: `Uniform(-0.02, 0.05)`.

The remaining nominal values are VT `502.089669`, POW `7.60335468`, NZINT
`0.0243004861`, and zero for the other coordinates. Both branches reject invalid
or non-safe initial states. The curriculum branch also excludes the goal altitude
band `[50, 150]`, so accepted proxy altitudes lie in `(150, 1000)`.
After 10,000 rejected curriculum proposals it falls back to the ordinary box;
that sampler also has a 10,000-proposal limit and raises on exhaustion.

The YAML sets both curriculum fractions to `0.25`. Although the generic annealing
fields contain 120,000–400,000, **this recipe's mixture probability and proxy
geometry remain constant**. It is not an easy-to-hard widening schedule.

The active `v6` safety bounds are altitude H `[50, 1000]`, alpha
`[-0.1745329252, 0.7853981634]`, `|beta| <= 0.5235987756`, `|theta| <= 1.4`,
`|PE| <= 200`, and `|P| <= 8`. Reset acceptance is strict (`h < 0`). Crash thresholds
are wider: H at or beyond `[0, 1100]`, `|PE| >= 250`, `|theta| >= pi/2`, or
`|P| >= 10`; alpha outside `[-0.2745329252, 0.8853981634]`, beta outside
`[-0.6235987756, 0.6235987756]`, and nonfinite states also terminate. A safety
violation can occur before a crash. The recipe uses a termination penalty of
100, a 640-step time limit, and 50 consecutive goal steps for a training success diagnostic.
Reaching that dwell threshold records success without ending the episode.

`NaN` in optional YAML fields is an **inherit/no-override sentinel**. For example,
`train_safe_h_min: NaN` inherits the profile's altitude minimum of 50. It does not
randomize the bound or pass a NaN state into the simulator. Generic curriculum
box bounds do not define the dedicated `theta_tail_proxy` geometry.

The F16 learner also inherits fixed defaults from
[its implementation](../custom/ssm/agents/ssm_agent.py): actor/reward-critic/
safety-critic/safety-value learning rates `3e-4`, actor cosine decay over 2M steps,
`time_dim=64`, `clip_sampler=True`, and no weight decay. These are not exposed
as additional trainer CLI flags.

Source: [environment](../custom/ssm/tasks/f16/env.py), especially `_BASE_BOX_V5`,
`resolve_region_geometry`, `_sample_theta_tail_proxy_state_v5`, and
`_sample_train_state_v5`; [trainer](../custom/ssm/tasks/f16/train.py) and
[accelerated entrypoint](../custom/ssm/tasks/f16/train_accelerated.py).

## F16 stabilization: Stage B

[The Stage-B recipe](f16_stage_b.yaml) starts from scratch and trains for 2M
steps with a 512×3 actor and 512×2 critics. Its HJ gate takes the minimum over one
policy action and seven local Gaussian proposals (`K=8`, `sigma=0.3`). Stage A
and Stage B are alternative learner modes here, not sequential curriculum phases.
`resume_from` restores a full training state; it is not an A-to-B warm-start option.

The fixed reset mixture requests the base box with probability 0.80, a positive
pitch box with probability 0.10 and a negative pitch box with probability 0.10.
The base is `_BASE_BOX_EFPPO_TRAIN`: relative to the Stage-A base table above,
alpha spans `[-0.1745329252, 0.7853981634]`, beta spans
`[-0.5235987756, 0.5235987756]`, theta spans `[-1.4, 0.4]`, P spans `[-0.5, 0.5]`,
PE spans `[-210, 210]` and H spans `[-10, 700]`; the other coordinate bounds are
unchanged. The two pitch components replace theta with `[0.4, 1.0]` or
`[-1.0, -0.4]`, respectively, and H with `[0, 1000]`. Proposals must be valid and
physically safe; the pitch components also exclude the goal altitude band. An
exhausted curriculum component falls back to the accepted base sampler. These
mixture probabilities and boxes remain fixed throughout training.

The effective safety geometry is `efppo_plus_p`: H in `[0, 1000]`,
`|theta| <= 0.95*pi/2`, `|PE| <= 200`, `|P| <= 8`, and the alpha/beta bounds above.
The YAML writes these effective bounds explicitly. It uses `discount_h=0.999`,
no extra termination penalty, a band reward of `0.30`, and seeded warmup actions.
The complete reward, termination and dynamics settings are in the YAML and the
shared [F16 environment](../custom/ssm/tasks/f16/env.py).

## Planar stabilize-avoid illustration

The separate [Stage-A](quad2d_stabilize_stage_a.yaml) and
[Stage-B](quad2d_stabilize_stage_b.yaml) recipes use two fixed reset mixtures,
a fixed Stage-A warm-start checkpoint, and mirror augmentation in Stage B.
See the [example](../examples/quad2d_stabilize_avoid/README.md) for mixture
weights, fork-state bounds, learning rates and training-step accounting.
These recipes describe the obstacle-avoidance illustration, not circle tracking.

## Running and changing parameters

```bash
bash scripts/train/quad3d.sh --dry-run
bash scripts/train/quad3d.sh --set seed=0 --set run_name=quad3d_seed0
```

Use `--set` for keys already in the task YAML. The reset-box constants documented
above live in the linked environment source; they are not additional YAML CLI
arguments. Changing those constants or the curriculum defines a different
training recipe and should be recorded as such.
