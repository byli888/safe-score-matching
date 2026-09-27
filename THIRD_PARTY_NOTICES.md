# Licensing and third-party acknowledgments

Original contributions to this repository are provided under the
[MIT License](LICENSE). Third-party components retain their own copyright and
license notices, listed below.

The included `jaxrl5` utilities and diffusion-policy components build on
[Q-Score Matching](https://github.com/escontra/score_matching_rl), by Michael
Psenka, Alejandro Escontrela, Pieter Abbeel and Yi Ma, and the
[JAXRL framework](https://github.com/ikostrikov/jaxrl), by Ilya Kostrikov.
Some files have been adapted for this project.

The upstream QSM README declares the MIT License. The available JAXRL MIT
copyright and permission notice is preserved in
[licenses/JAXRL-LICENSE.txt](licenses/JAXRL-LICENSE.txt).

The reversed-expectile safety value formulation follows
[FISOR](https://github.com/ZhengYinan-AIR/FISOR), by Yinan Zheng, Jianxiong Li,
Dongjie Yu, Yujie Yang, Shengbo Eben Li, Xianyuan Zhan and Jingjing Liu.
Source-level attribution is retained in the corresponding implementation.

The Quad3D environment in `custom/ssm/tasks/quad3d/env.py` and the dynamics
used by `custom/ssm/tasks/quad3d/evaluation.py` adapt the quadrotor dynamics
and region definitions from
[neural_clbf](https://github.com/MIT-REALM/neural_clbf). Its BSD 3-Clause
copyright and license notice is retained in
[licenses/neural-clbf-LICENSE.txt](licenses/neural-clbf-LICENSE.txt).

The F16 dynamics are supplied by the separately installed
[jax-f16](https://github.com/mit-realm/jax-f16) package. The velocity environments
are supplied by [Safety-Gymnasium](https://github.com/PKU-Alignment/safety-gymnasium),
[Gymnasium](https://github.com/Farama-Foundation/Gymnasium) and
[MuJoCo](https://github.com/google-deepmind/mujoco).
These dependencies retain their own licenses and are not included as vendored
distributions here.

The project license does not replace the licenses of these third-party
components or separately installed dependencies.
