"""Signed-constraint shaping for the Quad3D safety critic.

Positive h denotes violation. The supported transformations preserve its sign
and zero boundary while changing the scale or slope of positive values.
"""


# ============================================================
# SDF shaping
# ============================================================
def shape_sdf(h_raw, mode, unsafe_const=None, unsafe_slope=None,
              global_scale=None):
    """
    Apply SDF shaping to a raw signed-distance value.

    All modes preserve sign(h) and the h=0 boundary, so any downstream logic
    that uses sign(h) (binary cost, termination) is unaffected by the choice
    of mode. The shaped value is what the safety critic Q_h is trained to
    regress.

    Modes:
        "raw"        : pass-through. Returns h_raw unchanged.
        "hard"       : safe side (h_raw <= 0) unchanged; unsafe side
                       (h_raw > 0) hard-replaced with the constant
                       `unsafe_const`. Requires unsafe_const > 0.
                       Unsafe interior has zero gradient (deep-interior
                       flat plateau).
        "hard_slope" : safe side (h_raw <= 0) unchanged; unsafe side
                       (h_raw > 0) becomes `unsafe_const + unsafe_slope *
                       h_raw`. This is a piecewise-linear generalization
                       that allows a discrete jump (unsafe_const) at the
                       boundary and a slope (unsafe_slope * h) in the
                       unsafe interior. A positive slope retains variation
                       in violation magnitude; it does not guarantee a
                       nonzero gradient of the learned critic. Degenerate cases:
                         - unsafe_slope=0 -> equivalent to "hard"
                         - unsafe_const=0, unsafe_slope=1 -> equivalent to "raw"
                       Requires unsafe_const >= 0 and unsafe_slope >= 0,
                       with (unsafe_const + unsafe_slope) > 0 (otherwise
                       the unsafe side collapses to 0, destroying sign(h)).
        "hard_scale" : same as "hard", then the result is multiplied by
                       `global_scale`. Requires unsafe_const > 0 and
                       global_scale > 0. The safe side is also scaled, so
                       the safe-side dynamic range expands by global_scale.

    Args:
        h_raw: raw SDF value (h > 0 means unsafe).
        mode: one of "raw", "hard", "hard_slope", "hard_scale".
        unsafe_const: non-negative float; required for
            "hard" / "hard_slope" / "hard_scale" (strictly positive for
            "hard" / "hard_scale", non-negative for "hard_slope").
        unsafe_slope: non-negative float; required for "hard_slope".
        global_scale: positive float; required for "hard_scale".

    Returns:
        Shaped SDF value (float).
    """
    if mode == "raw":
        return float(h_raw)
    if mode == "hard":
        if unsafe_const is None or unsafe_const <= 0:
            raise ValueError(
                f"shape_sdf: mode='hard' requires unsafe_const > 0, "
                f"got {unsafe_const!r}")
        base = unsafe_const if h_raw > 0 else h_raw
        return float(base)
    if mode == "hard_slope":
        if unsafe_const is None or unsafe_const < 0:
            raise ValueError(
                f"shape_sdf: mode='hard_slope' requires unsafe_const >= 0, "
                f"got {unsafe_const!r}")
        if unsafe_slope is None or unsafe_slope < 0:
            raise ValueError(
                f"shape_sdf: mode='hard_slope' requires unsafe_slope >= 0, "
                f"got {unsafe_slope!r}")
        if (unsafe_const + unsafe_slope) <= 0:
            raise ValueError(
                f"shape_sdf: mode='hard_slope' requires "
                f"unsafe_const + unsafe_slope > 0 (else the unsafe side "
                f"collapses to 0 and sign(h) is destroyed); got "
                f"unsafe_const={unsafe_const!r}, unsafe_slope={unsafe_slope!r}")
        if h_raw > 0:
            return float(unsafe_const + unsafe_slope * h_raw)
        return float(h_raw)
    if mode == "hard_scale":
        if unsafe_const is None or unsafe_const <= 0:
            raise ValueError(
                f"shape_sdf: mode='hard_scale' requires unsafe_const > 0, "
                f"got {unsafe_const!r}")
        if global_scale is None or global_scale <= 0:
            raise ValueError(
                f"shape_sdf: mode='hard_scale' requires global_scale > 0, "
                f"got {global_scale!r}")
        base = unsafe_const if h_raw > 0 else h_raw
        return float(base * global_scale)
    raise ValueError(
        f"shape_sdf: unknown mode={mode!r}. "
        f"Valid: 'raw', 'hard', 'hard_slope', 'hard_scale'.")
