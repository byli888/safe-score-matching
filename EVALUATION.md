# Evaluation

Evaluate one locally trained or trusted checkpoint using its task protocol:

```bash
python evaluate.py --config evaluation/swimmer.yaml \
  --checkpoint runs/swimmer/checkpoints/step_1000000.pkl --out swimmer_eval.json
```

Use the Python environment for that task. The checkpoint supplies the policy architecture, sampling temperature and observation statistics. The evaluation YAML fixes the initial states or seeds, episode horizon and metrics. Evaluation runs on CPU and writes per-episode results and summary statistics to JSON. It does not search checkpoints, rank runs, or change the policy. Existing outputs are not overwritten.

| Task | Supplied protocol | Main measurements |
|---|---|---|
| HalfCheetah / Swimmer | 50 episodes of 1000 steps; shared reset and policy-seed streams | Episode reward and native binary velocity cost |
| Quad2D tracking | 100 fixed initial states; 360 steps | Six-state squared tracking error, safety classification, violations and crashes |
| Quad3D | 500 fixed physically safe, non-goal initial states; 500 steps | Terminal L1/L2, final-50 L1, safety classification, violations and crashes |
| F16 | Stage-A or Stage-B predicted-safe grid sampling; 512 steps | Conditional safety, final-window and consecutive stabilization, grid acceptance |
| Planar stabilize-avoid illustration | Three panels of 200 episodes; 400 steps | Terminal success, nominal collisions, margin violations and route counts |

The velocity recipes have five-seed results, and the separate stabilize-avoid
example has three-seed fixed-step results. The latter uses a different success
criterion from the paper's Figure 1. Matching the original custom-task paper
metrics requires the corresponding checkpoint, initial-state population and
metric definition; the supplied reference protocols do not yet reconstruct all
of those results.

## Reporting and checkpoint choice

The velocity benchmark results use the 1M-step endpoint for each of five training seeds. Their reported standard deviations are over the five per-seed evaluation means. An evaluation of one checkpoint measures variation across episodes, not across training seeds.

The paper reports the three custom-control tasks at best checkpoints, and the velocity tasks at fixed endpoints. This repository provides evaluation of a specified checkpoint; it does not include the complete checkpoint-selection workflow used for the paper. A supplied checkpoint path does not imply that it is the endpoint or the model behind a paper figure. Report the candidate steps, selection metric and evaluation data used to select a model. The fixed reference protocols below support comparisons without automatically selecting a model. The separate stabilize-avoid reproduction reports fixed A900k, B600k and B1M points, with no best-checkpoint search.

## Quadrotors

Quad2D uses the retained 100-state reference set, not random training resets. Tracking error is the sum of squared errors over six state coordinates at each valid transition, averaged within each episode and then across all episodes. It is not radial distance or a Euclidean norm. Crashed episodes remain in the results.

Quad3D uses the retained 500-state reference set. Terminal L1 and L2 are separate quantities. Final-50 L1 averages the full-state L1 error over the last 50 post-action states of the 500-step horizon. After an early crash, the terminal state is repeated to the end of that horizon; all episodes remain included.

Both protocols also report initial-state safety classification. Quad2D uses its online expectile value; Quad3D uses its online candidate-minimum Qh with the checkpoint's candidate settings. A score strictly below zero predicts safety. The label for actual safety is no `h > 0` among valid post-action transitions of that state's rollout; the initial state is not included. No-crash labels are reported separately. Precision is `TP/(TP+FP)`, false-safe rate is `FP/(TP+FP)`, and coverage is the predicted-safe fraction of this fixed panel. Precision and false-safe rate are undefined when no states are predicted safe. These labels describe one stochastic rollout per initial state; coverage is not state-space volume or recall.

## Planar stabilize-avoid illustration

Use `evaluation/quad2d_stabilize.yaml` for the separate [stabilize-avoid example](examples/quad2d_stabilize_avoid/README.md). The three panels cover an identical full state, small initial-velocity perturbations, and a broader lower-workspace region. The lower-region panel was used to select the original checkpoint; it is not held out. The evaluator writes all trajectories to an adjacent NPZ for plotting.

For this recipe, success requires terminal six-state Euclidean error below 0.50 and no nominal collision. Margin cost counts transitions inside the 0.10 m obstacle buffer or workspace inset; it can be positive without a collision. Report both measurements. Route counts include failures, and a left/right split across different initial velocities does not establish multimodality at one identical full state.

## F16

Choose the protocol for the checkpoint's learner mode:

- [f16.yaml](evaluation/f16.yaml): Stage A, one policy action for the Qh gate,
  with strict invalid-dynamics checks.
- [f16_stage_b.yaml](evaluation/f16_stage_b.yaml): Stage B, the checkpoint's
  candidate-minimum Qh gate, effective safety bounds and dynamics flags.
- [f16_stage_b_region.yaml](evaluation/f16_stage_b_region.yaml): Stage B with
  safety bounds from its region profile. This is for older checkpoints whose
  saved `train_safe_*` overrides were not applied by the original trainer.

The evaluator records the resolved bounds and settings. Grid acceptance is the
fraction of theta/height-grid states predicted safe. Rollouts sample from that
accepted population, so their safety and stabilization rates are conditional on
acceptance. Safety includes the initial state and every transition and requires
no crash. An empty accepted set has zero grid acceptance and undefined
conditional rates.

Stabilization is reported under three separate definitions, using the goal band
`H in [50, 150]` and a 50-transition window:

- `stabilization_rate`: the final 50 transitions are in band, with no crash.
- `consecutive_stabilization_rate`: an in-band run of at least 50 transitions
  occurs anywhere, with no crash during the entire episode.
- `reached_goal_dwell_rate`: such a run occurs, even if the aircraft later crashes.

These rates have the same sampled-start denominator. Keep their labels and the
initial-state protocol when comparing results; “50 consecutive steps” alone does
not identify the final-window metric.

## Output and numerical reproducibility

Outputs record the checkpoint identity, actual protocol, source hashes, numerical environment and per-episode outcomes. Custom checkpoint step numbers may come from filenames; they are labelled as such. Initial-state files or generator settings are recorded. Changing a protocol file defines a different evaluation and should be reported as such. Exact numeric equality is not guaranteed across hardware or dependency versions; compare using the documented environment and protocol.
