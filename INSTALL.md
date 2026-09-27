# Installation

Use Python 3.10 and separate environments for the custom tasks and velocity
benchmarks. Fresh CPU environments were checked with short, 3–4-step runs for each of the
five task entrypoints, exercising environment interaction, learner updates, and
model saving. These checks verify installation and execution, not convergence
or reproduction of the paper results.

## Quadrotors and F16

```bash
python3.10 -m venv .venv-custom
source .venv-custom/bin/activate
python -m pip install -r requirements-custom.txt
```

## HalfCheetah and Swimmer

```bash
python3.10 -m venv .venv-velocity
source .venv-velocity/bin/activate
python -m pip install -r requirements-velocity.txt
python -m pip install --no-deps gymnasium-robotics==1.2.2 safety-gymnasium==1.0.0
```

The two legacy environment packages are installed last because robotics 1.2.2
declares a NumPy upper bound incompatible with JAX 0.6.2. Their runtime
dependencies are listed explicitly in `requirements-velocity.txt`. This is the
combination used by the benchmark training implementation; `pip check` will
still report that robotics/NumPy metadata conflict.

## NVIDIA GPUs

In either environment, install the CUDA-enabled JAX wheel:

```bash
python -m pip install 'jax[cuda12]==0.6.2'
python -c 'import jax; print(jax.devices())'
```

For CPU use, keep the ordinary JAX dependencies without CUDA extras. The
launchers disable JAX memory preallocation by default; an explicitly set
`XLA_PYTHON_CLIENT_PREALLOCATE` takes precedence.

## Optional experiment tracking

Training logs and model files are saved locally. To enable W&B, install
`wandb==0.26.0` for custom tasks or `wandb==0.25.1` for velocity tasks, authenticate
your account, and pass `--set wandb=true` to the launcher. No account or API key
is needed for the default configurations.
