"""Step IV -- conflict target.

From the fused value s0, the node weight w and the block counts (h, p) of the solve region, build the data term of the
regularised solve: a target value s~ and its data strength c (research ``jacobi_t``, first half, unchanged).

    c_hit  = h / H  (nodes with weight),     c_pass = beta * p / H,     c = c_hit + c_pass       (H = persistence blocks)

``mirror`` (the TEF target): where c > 0 and s0 <= 0 (inside the fused surface)
    s~ = -max(|s0|, 0.1 f) * (c_hit - c_pass) / c          -- surface-supporting where h > beta p, free-space-supporting
                                                              where h < beta p; magnitude bounded by max(|s0|, 0.1 f);
                                                              s~ = s0 elsewhere.
``truncation``: s~ = (c_hit s0 + c_pass f) / c            -- pass evidence pulls towards the truncation value f.
``constant``:   s~ = (c_hit s0 + c_pass kappa) / c         -- carving control with a fixed free-space value kappa.
Here f is the truncation in metres (``free_target_m``), the field holds normalised values (the research scale, kept).
"""
from __future__ import annotations

import torch


def conflict_target(s0, w, h, p, *, pass_weight: float, persistence_blocks: int, free_target_m: float,
                    free_target_mode: str = "mirror", constant_target: float = 1.0):
    """Return ``(target, c)``: the conflict target s~ and its data strength c per node."""

    H = float(persistence_blocks)
    positive = w > 0.0
    c_hit = torch.where(positive, h / H, torch.zeros_like(h))
    c_pass = pass_weight * p / H
    c = c_hit + c_pass
    if free_target_mode == "truncation":
        target = torch.where(c > 0.0, (c_hit * s0 + c_pass * free_target_m) / c.clamp_min(1e-12), s0)
    elif free_target_mode == "constant":
        target = torch.where(c > 0.0, (c_hit * s0 + c_pass * float(constant_target)) / c.clamp_min(1e-12), s0)
    elif free_target_mode == "mirror":
        magnitude = torch.clamp(s0.abs(), min=0.1 * free_target_m)
        blended = -magnitude * (c_hit - c_pass) / c.clamp_min(1e-12)
        target = torch.where((c > 0.0) & (s0 <= 0.0), blended, s0)
    else:
        raise ValueError("free_target_mode must be mirror, truncation or constant")
    return target, c
