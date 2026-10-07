"""Incremental block remeshing and the per-block active region (research ``incremental.py``, unchanged arithmetic).

The map stays global and block-addressable: the sparse field (rows sorted by packed lattice key) and the per-block meshes
of the Mesher (block id = packed ``floor(coords / block_voxels)``, fixed because poses are given in one world frame).

* ``ray_region_blocks``: the blocks the temporal block's rays can reach, dilated by one block.  With ``region_candidates``
  the pass-vote candidates are the interior nodes of these blocks.  A node's pass votes depend only on its own position
  and normal and on the rays, and the vote kernel only finds nodes within one voxel of a ray sample, so a node outside the
  region would get zero votes: the restriction gives the same votes as testing every interior node.
* ``MeshReference``: the value and gate state each node had when the stored meshes that depend on it were extracted.  A
  face owned by node n (the start node of its crossing edge) depends only on nodes within one voxel of n, so a node whose
  state moved dirties the blocks of its eight corner neighbours (``corner_blocks``); those blocks are re-extracted with a
  one-voxel halo of context (``halo_rows``) instead of a one-block ring.

Reuse rule.  A node is dirty when it can enter or leave the extraction (weight gate), its sign flipped, its value moved
by more than ``tol`` (normalised units; 0.0125 x truncation 0.08 m = 1 mm for a unit-gradient field), or its weight
factor ``1 - exp(-w/2)`` of the cube probability moved by more than ``gate_tol``.  Dirty nodes take the current state as
their reference once their blocks are re-extracted; clean nodes keep theirs.  So every stored face is the exact
extraction of a field with the same signs and gates whose values lie within ``2 tol`` (weight factors within
``2 gate_tol``) of the current field.  The final extraction re-extracts every block, so the final mesh is the one a run
that extracts only at the end produces.
"""
from __future__ import annotations

import numpy as np
import torch

from model.sparse_field import block_keys, dilate_blocks, unpack_keys_t

_CORNERS = np.array([(dx, dy, dz) for dx in (-1, 1) for dy in (-1, 1) for dz in (-1, 1)], dtype=np.int64)


def corner_blocks(coords, rows, block_voxels: int, halo: int = 1, batch: int = 2_000_000):
    """Sorted unique block keys of ``coords[rows] + {-halo, +halo}^3``: every block within ``halo`` voxels of the rows."""
    dev = coords.device
    offs = torch.from_numpy(_CORNERS * int(halo)).to(dev)
    parts = [torch.zeros(0, dtype=torch.int64, device=dev)]
    for s in range(0, len(rows), batch):
        c = coords[rows[s:s + batch]]
        parts.append(torch.unique(block_keys((c[:, None, :] + offs[None, :, :]).reshape(-1, 3), block_voxels)))
    return torch.unique(torch.cat(parts))


def halo_rows(coords, block_of, blocks, block_voxels: int, halo: int = 1, batch: int = 2_000_000):
    """Rows of the nodes in ``blocks`` plus every node within ``halo`` voxels (Chebyshev) of them: all the context the
    faces owned by nodes of ``blocks`` depend on."""
    B = int(block_voxels)
    blocks = torch.unique(blocks.to(coords.device))
    sel = torch.isin(block_of, blocks)
    lc = coords - torch.div(coords, B, rounding_mode="floor") * B
    edge = torch.nonzero(((lc < halo) | (lc > B - 1 - halo)).any(dim=1) & ~sel).reshape(-1)
    offs = torch.from_numpy(_CORNERS * int(halo)).to(coords.device)
    for s in range(0, len(edge), batch):
        e = edge[s:s + batch]
        near = torch.isin(block_keys((coords[e][:, None, :] + offs[None, :, :]).reshape(-1, 3), B), blocks).reshape(-1, 8).any(dim=1)
        sel[e[near]] = True
    return torch.nonzero(sel).reshape(-1)


def ray_region_blocks(views, voxel: float, block_voxels: int, clearance_m: float, device):
    """Sorted keys of every block holding a node the pass-vote kernel can reach from ``views``.

    The kernel samples each ray every voxel up to ``range + clearance + voxel`` and looks up nodes within one voxel of
    the sample's voxel.  Samples here are ``block_voxels / 2`` voxels apart (both ends included), so every kernel node
    lies within ``block_voxels / 4 + 2`` voxels of one of them, i.e. in its block or a 26-neighbour: the dilated set is a
    superset.  Every ray is used (the kernel may subsample; a superset is still exact)."""
    B = int(block_voxels)
    if B < 4:
        raise ValueError("ray_region_blocks needs block_voxels >= 4")
    step = 0.5 * B * voxel
    parts = [torch.zeros(0, dtype=torch.int64, device=device)]
    for origin, ends in views:
        o = torch.as_tensor(np.asarray(origin, dtype=np.float64).reshape(3), device=device)
        e = torch.as_tensor(np.asarray(ends, dtype=np.float64).reshape(-1, 3), device=device)
        e = e[torch.isfinite(e).all(dim=1)]
        if not bool(torch.isfinite(o).all()) or len(e) == 0:
            continue
        ray = e - o
        rng = torch.linalg.norm(ray, dim=1)
        d = ray / rng.clamp_min(1e-12)[:, None]
        t_max = rng + clearance_m + voxel
        n = int(torch.ceil(t_max.max() / step).item()) + 1
        t = torch.minimum(torch.arange(n, device=device, dtype=torch.float64)[None, :] * step, t_max[:, None])   # (R,n), last = t_max
        vox = torch.floor((o[None, None, :] + d[:, None, :] * t[:, :, None]) / voxel).to(torch.int64)
        parts.append(torch.unique(block_keys(vox.reshape(-1, 3), B)))
    return dilate_blocks(torch.cat(parts), B, 1) if sum(len(p) for p in parts) else parts[0]


def dirty_mask(sdf, weight, val, g, keep, tol, gate_tol, w_min):
    """The reuse rule on aligned arrays: current (sdf, weight) against the reference (val NaN = never extracted, g, keep)."""
    keep_now = weight >= w_min
    g_now = 1.0 - torch.exp(-weight / 2.0)
    seen = torch.isfinite(val)
    dirty = (keep_now & ~seen) | (seen & (keep_now != keep))
    both = seen & keep_now & keep
    moved = ((sdf - val).abs() > tol) | ((sdf < 0) != (val < 0)) | ((g_now - g).abs() > gate_tol)
    return dirty | (both & moved)


class MeshReference:
    """Row-aligned reference state of the stored block meshes (see the module docstring for the rule)."""

    def __init__(self, tol: float, gate_tol: float, minimum_node_weight: float):
        if not (tol > 0 and gate_tol > 0):
            raise ValueError("remesh_reuse_tol and remesh_gate_tol must be positive")
        self.tol, self.gate_tol, self.w_min = float(tol), float(gate_tol), float(minimum_node_weight)
        self.val = self.g = self.keep = None      # val NaN = never extracted

    def after_merge(self, n: int, old_inv, device, dtype):
        """Carry the reference through a merge (old row i moved to ``old_inv[i]``; new rows have none)."""
        val = torch.full((n,), float("nan"), dtype=dtype, device=device)
        g = torch.full((n,), float("nan"), dtype=dtype, device=device)
        keep = torch.zeros(n, dtype=torch.bool, device=device)
        if self.val is not None and len(self.val):
            val[old_inv], g[old_inv], keep[old_inv] = self.val, self.g, self.keep
        self.val, self.g, self.keep = val, g, keep

    def evict(self, far):
        if self.val is not None and len(self.val) == len(far):
            self.val, self.g, self.keep = self.val[~far], self.g[~far], self.keep[~far]

    def dirty_rows(self, sdf, weight):
        return torch.nonzero(dirty_mask(sdf, weight, self.val, self.g, self.keep, self.tol, self.gate_tol, self.w_min)).reshape(-1)

    def commit(self, rows, sdf, weight):
        self.val[rows] = sdf[rows]
        self.g[rows] = 1.0 - torch.exp(-weight[rows] / 2.0)
        self.keep[rows] = weight[rows] >= self.w_min


def store_block_faces(blocks: dict, v, f, conf, owner, keep_blocks, block_voxels: int, device) -> int:
    """Vectorised ``Mesher._store_block_faces`` (same stored arrays): split the faces owned by ``keep_blocks`` by owner block
    with one sort on the device, gather each block's vertices there and copy them to the host once.  Takes numpy arrays
    or device tensors (``extract_surface_nets_t(return_tensors=True)``).  Returns the stored face count."""
    if len(f) == 0:
        return 0
    dev = torch.device(device)
    t_ = lambda x: x.to(dev) if isinstance(x, torch.Tensor) else torch.from_numpy(np.asarray(x)).to(dev)   # noqa: E731
    ob = block_keys(unpack_keys_t(t_(owner)), block_voxels)
    keep = torch.isin(ob, keep_blocks.to(dev))
    ob, order = torch.sort(ob[keep], stable=True)
    F = t_(f).to(torch.int64)[keep][order]
    ub, binv, nf = torch.unique_consecutive(ob, return_inverse=True, return_counts=True)
    nv_all = len(v)
    pair, inv = torch.unique((binv[:, None] * nv_all + F).reshape(-1), return_inverse=True)   # sorted by (block, vertex)
    nv = torch.bincount(torch.div(pair, nv_all, rounding_mode="floor"), minlength=len(ub))
    vstart = torch.cumsum(nv, 0) - nv
    local = (inv - vstart[binv.repeat_interleave(3)]).reshape(-1, 3).to(torch.int32).cpu().numpy()
    used = pair % nv_all
    vs = t_(v).to(torch.float64)[used].cpu().numpy()
    cs = t_(conf).to(torch.float32)[used].cpu().numpy()
    fo = np.concatenate([[0], np.cumsum(nf.cpu().numpy())])
    vo = np.concatenate([[0], np.cumsum(nv.cpu().numpy())])
    for i, b in enumerate(ub.cpu().numpy().tolist()):
        a, e = vo[i], vo[i + 1]
        blocks[int(b)] = (vs[a:e].copy(), local[fo[i]:fo[i + 1]].copy(), cs[a:e].copy())
    return int(fo[-1])


def split_block_faces(v, f, owner, block_voxels: int, device):
    """``{block: (vertices, faces)}`` of an extraction, split by owner block (for audits and tests)."""
    ob = block_keys(unpack_keys_t(torch.from_numpy(owner).to(device)), block_voxels).cpu().numpy()
    order = np.argsort(ob, kind="stable")
    f, ob = f[order], ob[order]
    out = {}
    uniq, starts = np.unique(ob, return_index=True)
    for b, lo, hi in zip(uniq, starts, np.append(starts[1:], len(ob))):
        used, local = np.unique(f[lo:hi].reshape(-1), return_inverse=True)
        out[int(b)] = (v[used], local.reshape(-1, 3).astype(np.int32))
    return out
