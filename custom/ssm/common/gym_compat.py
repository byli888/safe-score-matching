"""Small compatibility layer for importing legacy gym without noisy notices."""

from __future__ import annotations

import contextlib
import io


def _import_gym_quietly():
    # gym 0.23 prints the NumPy-2 warning to stderr during import via
    # gym_notices. Suppress just that import-time notice locally.
    with contextlib.redirect_stderr(io.StringIO()):
        import gym as _gym

    return _gym


gym = _import_gym_quietly()
spaces = gym.spaces

