"""Step V -- regularised solve.

The solve region is every block the finished block touched or gave a new pass vote, dilated by one block for boundary
context; only the interior (the undilated blocks) is updated, the dilation ring stays fixed.  On that region a damped
Jacobi iteration minimises

    sum_i c_i (s_i - s~_i)^2  +  sum_(i,j) lambda_ij (s_i - s_j)^2      (six-neighbour lattice edges)

with lambda_ij = lambda / degree_i (degree-normalised), warm-started from the current field (research ``jacobi_t``,
low-memory path, unchanged: int32 neighbour table, neighbour values gathered in chunks).
"""
from __future__ import annotations

import torch

from model.sparse_field import dilate_blocks, pack_keys_t

_FACE_STEPS = ((1 << 42), -(1 << 42), (1 << 21), -(1 << 21), 1, -1)


def solve_region(block_of, touched_rows, newly_rows, block_voxels: int, margin_blocks: int):
    """Rows of the solve region and the boolean interior mask over all rows."""

    region_blocks = torch.unique(torch.cat([block_of[touched_rows], block_of[newly_rows]]))
    solve_blocks = dilate_blocks(region_blocks, block_voxels, int(margin_blocks))
    interior = torch.isin(block_of, region_blocks)
    subset = torch.isin(block_of, solve_blocks)
    return torch.nonzero(subset).reshape(-1), interior


def face_neighbour_indices_t(keys, out_dtype=None):
    """(N,6) device tensor of face-neighbour rows, -1 when absent; ``keys`` are (N,3) coordinates or packed (N,) keys."""

    out_dtype = torch.int64 if out_dtype is None else out_dtype
    packed = keys if keys.dim() == 1 else pack_keys_t(keys)
    n = len(packed)
    if out_dtype == torch.int32 and n >= 2 ** 31:
        raise ValueError("int32 neighbour table needs fewer than 2**31 rows")
    out = torch.full((n, 6), -1, dtype=out_dtype, device=keys.device)
    if n == 0:
        return out
    order = torch.argsort(packed)
    sorted_packed = packed[order]
    for column, step in enumerate(_FACE_STEPS):
        neighbour = packed + step
        pos = torch.clamp(torch.searchsorted(sorted_packed, neighbour), max=n - 1)
        found = sorted_packed[pos] == neighbour
        rows = torch.where(found, order[pos], torch.full_like(pos, -1))
        out[:, column] = rows if out_dtype == torch.int64 else rows.to(out_dtype)
    return out


def damped_jacobi(keys_t, target, c, s_init, *, lam: float, iterations: int, damping: float = 0.8, update_mask=None,
                  tolerance: float = 1e-6, gather_rows: int = 8_000_000):
    """Return ``(s, iterations_done, max_update)``.  ``keys_t``: packed keys of the region rows; ``target``, ``c``: the
    conflict target and data strength (step IV); ``s_init``: warm start; rows outside ``update_mask`` keep ``s_init`` and
    act as fixed boundary values.  Stops early once the largest update is below ``tolerance``."""

    n = len(target)
    neighbours = face_neighbour_indices_t(keys_t, out_dtype=torch.int32)
    del keys_t
    absent = neighbours < 0
    degree = (~absent).sum(dim=1).to(target.dtype)
    neighbours.clamp_(min=0)          # -1 -> row 0; those entries are zeroed after the gather
    idx_flat = neighbours.view(-1)
    rows = max(1, int(gather_rows))
    smooth_weight = torch.where(degree > 0.0, lam / degree.clamp_min(1.0), torch.zeros_like(degree))   # lambda / degree
    denominator = c + smooth_weight * degree
    has_den = denominator > 0.0
    den_safe = denominator.clamp_min(1e-12)
    del denominator, degree
    s = s_init.clone()
    max_update = 0.0
    done = 0
    for done in range(1, iterations + 1):
        neighbour_sum = torch.empty_like(s)
        for start in range(0, n, rows):
            stop = min(n, start + rows)
            g = torch.index_select(s, 0, idx_flat[start * 6:stop * 6]).view(stop - start, 6)
            g.masked_fill_(absent[start:stop], 0.0)
            neighbour_sum[start:stop] = g.sum(dim=1)
            del g
        proposal = torch.where(has_den, (c * target + smooth_weight * neighbour_sum) / den_safe, s)
        update = proposal - s
        if update_mask is not None:
            update = torch.where(update_mask, update, torch.zeros_like(update))
        s = s + damping * update
        max_update = float(update.abs().max().item()) if n else 0.0
        if max_update < tolerance:
            break
    return s, done, max_update
