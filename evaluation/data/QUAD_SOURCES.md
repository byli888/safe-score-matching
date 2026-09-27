# Fixed Quad initial states

These technical data files are copied unchanged from the existing SSM evaluation artifacts:

| File | Population | Original artifact | SHA-256 |
|---|---|---|---|
| quad2d_initial_states.npz | 100 collaborator-supplied Quad2D trajectory-tracking initial states, 6 physical state coordinates | quad2d_opt_eval_inits.npz | 26896850f21bb9ab610f26d24363315744c8efda009467e87e570f2fe3415348 |
| quad3d_initial_states.npz | 500 initial states drawn with the physical h<0 and non-goal rejection rule; seed 20260506, 9 physical state coordinates | quad3d_heldout_hreject_test500_seed20260506.npz | 2b805159faaa4e5ed72268e663db510919ae7e03095848d33a89bf3619e03a42 |

Each file contains an `init_states` array in the task's physical state coordinates; the Quad3D file also retains the original `init_h` margins. No learned safety predictor filters either evaluation panel. Quad2D starts are the fixed collaborator panel, not an independently generated uniform-state sample. The Quad3D panel is conditional on physical safety and exclusion of the goal.

The files preserve the reference states used by these protocols. The Quad2D collaborator panel and the generated Quad3D panel have different sampling definitions; report that distinction when comparing results.
