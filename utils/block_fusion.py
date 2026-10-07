"""Step III -- temporal-block evidence.

Per scan, the samples S_t are splatted into the block-local field B_b (one frame bit per scan).  When a 1 s block ends:
  III.1  the block field is merged into the persistent field.  T2 bounds each node's block weight (min of its largest
         single sample weight and its block weight sum), so dense repeated sampling within a block does not add weight;
         every node the block touched receives one hit vote (h += 1, at most one per block);
  III.2  field normals are refreshed around the touched nodes, and every node inside the fused surface (s0 <= 0) is
         tested against the block's rays: a node traversed as free space by at least one ray of a scan of the block --
         and not hit by that scan -- receives one pass vote (p += 1, at most one per block).
The hit / pass counts (h, p) and the fused value s0 feed the conflict target of step IV.
Optional counting units (frame / ray instead of block) reproduce the counting-unit control of the paper.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np
import torch

from model.sparse_field import (LatticeTable, SparseField, block_keys, mask_bit, dilated_block_keys, pack_keys_t, popcount,
                                splat_trilinear_t, unpack_keys_t, _NEIGHBOUR_OFFSETS)
from utils.sampler import deterministic_subsample_indices


# ----------------------------------------------------------------------------------------------------------------------
# free-space pass votes (research tef/evidence.py)
# ----------------------------------------------------------------------------------------------------------------------
_RAYS_PER_BATCH = 4096
_BLOCK_PREFILTER = 8       # pre-filter blocks (voxels) of the occupied-block test


def estimate_candidate_free_space_frames_gpu(
    candidate_coordinates,
    voxel_size_m: float,
    ray_views: Sequence[tuple[np.ndarray, np.ndarray]],
    *,
    candidate_normals,
    maximum_rays_per_view: int = 12_000,
    ray_radius_m: float = 0.06,
    endpoint_clearance_m: float = 0.10,
    minimum_free_rays_per_frame: int = 1,
    device: str = "cuda",
    occupied_blocks=None,
    count_rays: bool = False,
):
    """Number of scans of the block that count as free space for each candidate node.

    Every ray is sampled once per voxel up to its endpoint (+ clearance); the 27 lattice neighbours of each sample are
    looked up among the candidates.  A ray passes a node with a field normal when it crosses the node's tangent plane
    within ``ray_radius_m`` of the node, at least ``endpoint_clearance_m`` before its endpoint (a node without a normal:
    the ray passes within the radius, the clearance before its endpoint).  A ray hits a node whose distance along the
    ray is within the clearance of its endpoint.  A scan counts as free for a node when at least
    ``minimum_free_rays_per_frame`` of its rays pass the node and none hits it.

    Returns ``frame_counts`` (N,) int32, aligned with ``candidate_coordinates``; with ``count_rays`` also the number of
    distinct passing rays summed over the scans that count as free (counting unit ``ray``)."""

    coords_t = candidate_coordinates.to(device, torch.int64).reshape(-1, 3)
    n_nodes = int(len(coords_t))
    voxel = float(voxel_size_m)
    if not (voxel > 0 and ray_radius_m > 0 and endpoint_clearance_m > 0):
        raise ValueError("free-space metric scales must be finite and positive")
    if maximum_rays_per_view <= 0 or minimum_free_rays_per_frame <= 0:
        raise ValueError("maximum_rays_per_view and minimum_free_rays_per_frame must be positive")
    if n_nodes == 0:
        frame_counts = np.zeros(0, dtype=np.int32)
        return (frame_counts, np.zeros(0, dtype=np.int64)) if count_rays else frame_counts

    table = LatticeTable(coords_t, device)
    centres = (coords_t.to(torch.float64) * voxel).to(torch.float32)
    # Occupied-block pre-filter: most samples along a ray lie in empty air; a sample whose 8-voxel block -- dilated by one
    # voxel -- holds no candidate cannot find any of its 27 neighbours in the table, so it is dropped before the lookups.
    # The mapper passes a superset it maintains (dilated blocks of every node ever touched).
    B = _BLOCK_PREFILTER
    occupied = occupied_blocks if occupied_blocks is not None else dilated_block_keys(coords_t, B)
    nrm_t = candidate_normals.to(device, torch.float64).reshape(-1, 3)
    if len(nrm_t) != n_nodes:
        raise ValueError("candidate_normals must have one row per candidate")
    if not bool(torch.isfinite(nrm_t).all()):
        raise ValueError("candidate_normals must be finite")
    length = torch.linalg.norm(nrm_t, dim=1)
    has_normal = length > 1e-9
    normals = torch.where(has_normal[:, None], nrm_t / length.clamp_min(1e-12)[:, None], torch.zeros_like(nrm_t)).to(torch.float32)
    offsets = torch.from_numpy(_NEIGHBOUR_OFFSETS).to(device)  # (27,3)
    n_off = int(len(offsets))
    counts_dev = torch.zeros(n_nodes, dtype=torch.int32, device=device)
    rays_dev = torch.zeros(n_nodes, dtype=torch.int64, device=device) if count_rays else None
    step = voxel                      # one sample per voxel along each ray
    r2 = float(ray_radius_m) ** 2

    for origin_value, endpoints_value in ray_views:
        origin_np = np.asarray(origin_value, dtype=np.float64).reshape(3)
        endpoints_np = np.asarray(endpoints_value, dtype=np.float64)
        if endpoints_np.ndim != 2 or endpoints_np.shape[1] != 3:
            raise ValueError("each free-space endpoint array must have shape (N, 3)")
        if not np.isfinite(origin_np).all():
            continue
        idx = deterministic_subsample_indices(len(endpoints_np), maximum_rays_per_view)
        endpoints_np = endpoints_np[idx]
        finite = np.isfinite(endpoints_np).all(axis=1)
        endpoints_np = endpoints_np[finite]
        if len(endpoints_np) == 0:
            continue
        origin = torch.from_numpy(origin_np).to(device=device, dtype=torch.float32)
        ends = torch.from_numpy(endpoints_np).to(device=device, dtype=torch.float32)
        ray = ends - origin
        rng = torch.linalg.norm(ray, dim=1)
        keep = rng > endpoint_clearance_m
        ray, rng = ray[keep], rng[keep]
        if len(rng) == 0:
            continue
        direction = ray / rng.clamp_min(1e-12)[:, None]
        free_votes = torch.zeros(n_nodes, dtype=torch.int32, device=device)
        free_rays = torch.zeros(n_nodes, dtype=torch.int64, device=device) if count_rays else None
        surface_hit = torch.zeros(n_nodes, dtype=torch.bool, device=device)
        for start in range(0, len(rng), _RAYS_PER_BATCH):
            d_b = direction[start:start + _RAYS_PER_BATCH]
            r_b = rng[start:start + _RAYS_PER_BATCH]
            n_ray = len(r_b)
            # samples along each ray up to range + clearance (+1 voxel margin) so the
            # surface guard around the endpoint is seen by the same traversal
            t_max = r_b + endpoint_clearance_m + voxel
            n_samp = int(torch.ceil(t_max.max() / step).item()) + 1
            t_grid = torch.arange(n_samp, device=device, dtype=torch.float32) * step  # (S,)
            valid_s = t_grid[None, :] <= t_max[:, None]  # (R,S)
            pts = origin[None, None, :] + d_b[:, None, :] * t_grid[None, :, None]  # (R,S,3)
            vox = torch.round(pts / voxel).to(torch.int64)  # (R,S,3)
            bkey = pack_keys_t(torch.div(vox, B, rounding_mode="floor").reshape(-1, 3)).reshape(vox.shape[:2])
            pos = torch.clamp(torch.searchsorted(occupied, bkey), max=len(occupied) - 1)
            valid_s = valid_s & (occupied[pos] == bkey)
            ray_id, samp_id = torch.nonzero(valid_s, as_tuple=True)
            vox = vox[ray_id, samp_id]  # (P,3)
            # dedupe consecutive identical voxels per ray cheaply: unique over (ray, voxel)
            cand = vox[:, None, :] + offsets[None, :, :]  # (P,n_off,3)
            ray_rep = ray_id[:, None].expand(-1, n_off)
            packed = pack_keys_t(cand.reshape(-1, 3))
            node = table.lookup(packed)
            ray_rep = ray_rep.reshape(-1)
            hit = node >= 0
            node, ray_rep = node[hit], ray_rep[hit]
            if len(node) == 0:
                continue
            if minimum_free_rays_per_frame == 1:
                # duplicates (a node reached from two adjacent samples of the same ray) cannot
                # change an "at least one free ray / any surface ray" decision: skip the sort
                ray_local = ray_rep
            else:
                pair = torch.unique(node * n_ray + ray_rep)
                node = pair // n_ray
                ray_local = pair % n_ray
            p = centres[node] - origin  # candidate ray (node relative to sensor)
            u = d_b[ray_local]
            R = r_b[ray_local]
            p_range = torch.linalg.norm(p, dim=1)
            valid_c = p_range > endpoint_clearance_m
            along = (p * u).sum(dim=1)
            lateral2 = (p_range * p_range - along * along).clamp_min(0.0)
            close = (along > 0) & (lateral2 <= r2)
            free = close & (R >= along + endpoint_clearance_m)
            nb = normals[node]
            present = has_normal[node]
            approach = (nb * u).sum(dim=1)
            approaching = approach.abs() > 1e-6
            plane_offset = (p * nb).sum(dim=1)
            t_cross = torch.where(approaching, plane_offset / torch.where(approaching, approach, torch.ones_like(approach)), torch.full_like(approach, -1.0))
            x = t_cross[:, None] * u
            in_plane2 = ((x - p) ** 2).sum(dim=1)
            crossing = approaching & (t_cross > 0) & (in_plane2 <= r2) & (R >= t_cross + endpoint_clearance_m)
            free = torch.where(present, crossing, free)
            surface = close & ((R - along).abs() < endpoint_clearance_m)
            free = free & valid_c
            surface = surface & valid_c
            free_votes.index_add_(0, node[free], torch.ones(int(free.sum()), dtype=torch.int32, device=device))
            if count_rays and bool(free.any()):
                # a node reached from several samples of one ray counts that ray once
                pair = torch.unique(node[free] * n_ray + ray_local[free])
                free_rays.index_add_(0, pair // n_ray, torch.ones(len(pair), dtype=torch.int64, device=device))
            surface_hit[node[surface]] = True
        free_in_frame = (free_votes >= minimum_free_rays_per_frame) & ~surface_hit
        counts_dev += free_in_frame.to(torch.int32)
        if count_rays:
            rays_dev += torch.where(free_in_frame, free_rays, torch.zeros_like(free_rays))

    frame_counts = counts_dev.cpu().numpy().astype(np.int32)
    if count_rays:
        return frame_counts, rays_dev.cpu().numpy()
    return frame_counts


# ----------------------------------------------------------------------------------------------------------------------
# frame / ray counting units (research tef/evidence.py, --evidence-count-unit)
# ----------------------------------------------------------------------------------------------------------------------
class EvidenceUnits:
    """Counting-unit control (``count_unit: frame | ray``): count hit / pass evidence per scan or per ray instead of once
    per temporal block.

    Hits add, per block and node, the number of the block's scans that splatted the node (``frame``) or the node's summed
    sample weight before the block weight is bounded (``ray``).  Passes add the number of scans that count as free for the
    node (``frame``) or the distinct free rays of those scans (``ray``); the per-scan free / surface rule is unchanged.  The
    solver reads each total rescaled by the running ratio of block-unit to unit mass (hits and passes separately), so the
    evidence keeps on average the size of the block counts and only its distribution over nodes and time changes."""

    def __init__(self, unit: str):
        if unit not in ("frame", "ray"):
            raise ValueError("evidence count unit must be frame or ray (block is the default)")
        self.unit = unit
        self.mass = {"hit_block": 0.0, "hit_unit": 0.0, "pass_block": 0.0, "pass_unit": 0.0}

    def block_hits(self, block_field):
        """Hit units of a block field, aligned with its keys; call before the block weights are bounded."""
        if self.unit == "frame":
            return popcount(block_field.mask).to(block_field.dtype)
        return block_field.weight_sum.clone()

    def after_merge(self, field, old_inv, new_inv, hits) -> None:
        """Carry the unit totals through the merge's row remapping and add this block's hits."""
        m = len(field)
        for name in ("hit_units", "pass_units"):
            old = getattr(field, name, None)
            new = torch.zeros(m, dtype=field.dtype, device=field.device)
            if old is not None and len(old) == len(old_inv):
                new[old_inv] = old
            setattr(field, name, new)
        field.hit_units.index_add_(0, new_inv, hits.to(field.dtype))
        self.mass["hit_block"] += float(len(new_inv))
        self.mass["hit_unit"] += float(hits.sum())

    def add_passes(self, field, rows, units) -> None:
        """``rows``: nodes whose block pass count was incremented in this merge; ``units``: their scan or ray counts."""
        if len(rows) == 0:
            return
        field.pass_units.index_add_(0, rows, units.to(field.dtype))
        self.mass["pass_block"] += float(len(rows))
        self.mass["pass_unit"] += float(units.sum())

    def scales(self):
        m = self.mass
        return (m["hit_block"] / m["hit_unit"] if m["hit_unit"] > 0 else 1.0,
                m["pass_block"] / m["pass_unit"] if m["pass_unit"] > 0 else 1.0)

    def counts(self, field, rows):
        """Rescaled (hit, pass) evidence of ``rows`` for the solver."""
        sh, sp = self.scales()
        return field.hit_units[rows] * sh, field.pass_units[rows] * sp

    def summary(self) -> dict:
        sh, sp = self.scales()
        return {"unit": self.unit, **self.mass, "hit_scale": sh, "pass_scale": sp}


# ----------------------------------------------------------------------------------------------------------------------
# field normals for the crossing test (research tef/solver.py)
# ----------------------------------------------------------------------------------------------------------------------
def _gradient_normals_t(table, sdf_rows, coords):
    def lookup(q):
        row = table.lookup(pack_keys_t(q))
        out = torch.full((len(q),), float("nan"), dtype=sdf_rows.dtype, device=q.device)
        ok = row >= 0
        out[ok] = sdf_rows[row[ok]]
        return out

    centre = lookup(coords)
    grad = torch.zeros((len(coords), 3), dtype=sdf_rows.dtype, device=coords.device)
    for axis in range(3):
        step = torch.zeros(3, dtype=torch.int64, device=coords.device); step[axis] = 1
        plus = lookup(coords + step); minus = lookup(coords - step)
        both = torch.isfinite(plus) & torch.isfinite(minus)
        only_plus = torch.isfinite(plus) & ~both & torch.isfinite(centre)
        only_minus = torch.isfinite(minus) & ~both & torch.isfinite(centre)
        grad[:, axis] = torch.where(both, 0.5 * (plus - minus), grad[:, axis])
        grad[:, axis] = torch.where(only_plus, plus - centre, grad[:, axis])
        grad[:, axis] = torch.where(only_minus, centre - minus, grad[:, axis])
    length = grad.norm(dim=1, keepdim=True)
    return torch.where(length > 1e-12, grad / length.clamp_min(1e-12), torch.zeros_like(grad))


# ----------------------------------------------------------------------------------------------------------------------
# step III as used by the mapper
# ----------------------------------------------------------------------------------------------------------------------
class BlockFusion:
    """Block-local accumulation (per scan) and the block merge with hit / pass evidence (per block)."""

    def __init__(self, cfg, device: str):
        self.cfg, self.device, self.dev = cfg, device, torch.device(device)
        self.voxel, self.truncation = cfg.voxel_m, cfg.truncation_m
        self.alpha_radius = cfg.free_space_ray_radius_factor * self.voxel
        self.alpha_clearance = cfg.free_space_endpoint_clearance_factor * self.voxel
        self.units = None if cfg.count_unit == "block" else EvidenceUnits(cfg.count_unit)
        self.block_field, self.block_id, self.block_local, self.block_views = None, None, 0, []
        self.block_last_stamp_ns, self.block_last_index = 0, None
        self.normals = None        # persistent-field normals, row-aligned (refreshed around touched nodes)
        self.occupied = None       # dilated blocks of every node ever touched (pre-filter of the crossing test)

    # -- per scan -------------------------------------------------------------------------------------------------------
    def begin_block(self, block_id: int):
        self.block_id, self.block_local, self.block_views = block_id, 0, []
        self.block_field = SparseField(self.voxel, self.truncation, device=self.device)

    def add_rays(self, origin_world, endpoints_world):
        """Keep the scan's rays (origin, selected endpoints in the world frame) for the block's crossing test."""
        self.block_views.append((np.asarray(origin_world, dtype=np.float64).copy(), np.ascontiguousarray(endpoints_world)))

    def next_frame_bit(self) -> int:
        bit = mask_bit(self.block_local)
        self.block_local += 1
        return bit

    def accumulate(self, positions, values, weights, frame_bit: int):
        """Splat one scan's samples and add them to the block field B_b."""
        keys, value_sum, weight_sum, weight_max = splat_trilinear_t(positions, values, weights, self.voxel,
                                                                    track_weight_max=self.cfg.block_weight == "max")
        self.block_field.merge(keys, value_sum, weight_sum, frame_bit, weight_max=weight_max)

    # -- per block --------------------------------------------------------------------------------------------------------
    def merge_block(self, field: SparseField, c: int):
        """III.1: bounded block weight, merge into the persistent field, one hit vote per touched node.
        Returns ``(n_new, old_inv, new_inv)``; ``new_inv`` are the persistent rows the block touched."""
        bf = self.block_field
        unit_hits = self.units.block_hits(bf) if self.units is not None else None
        if self.cfg.block_weight != "sum":
            bf.bound_block_weights(self.cfg.block_weight)
        n_new = field.merge(bf.keys, bf.value_sum, bf.weight_sum, mask_bit(c), block_id=c)
        old_inv, new_inv = field.last_merge_inverse
        if unit_hits is not None:
            self.units.after_merge(field, old_inv, new_inv, unit_hits)
        return n_new, old_inv, new_inv

    def count_passes(self, field: SparseField, c: int, old_inv, touched_rows, views, retry: bool = False, region_blocks=None):
        """III.2: refresh normals near the touched nodes, then give one pass vote per block to every node inside the fused
        surface that the block's rays traverse as free space.  Returns ``(coords, raw, block_of, newly, counts)``.
        ``retry``: the same block again after an out-of-memory failure of the final flush (normals are already row-aligned,
        no evidence unit is added twice).  ``region_blocks`` (online extension): only interior nodes of these blocks are
        candidates -- the blocks the rays can reach, so the votes are the same."""
        dev = self.dev
        coords = unpack_keys_t(field.keys)
        raw = field.raw_sdf()
        sdf_nan = torch.where(field.weight_sum > 0, raw, torch.full_like(raw, float('nan')))
        table = LatticeTable.from_sorted_packed(field.keys, self.device)
        normals_all = torch.full((len(field), 3), float('nan'), dtype=raw.dtype, device=dev)
        if retry and self.normals is not None and len(self.normals) == len(field):
            normals_all = self.normals.clone()
        elif self.normals is not None:
            normals_all[old_inv] = self.normals
        step = torch.eye(3, dtype=torch.int64, device=dev)
        nb = torch.cat([coords[touched_rows] + step[i] for i in range(3)] + [coords[touched_rows] - step[i] for i in range(3)])
        nb_rows = table.lookup(pack_keys_t(nb))
        nb_rows = nb_rows[nb_rows >= 0]
        stale = torch.unique(torch.cat([touched_rows, nb_rows, torch.nonzero(~torch.isfinite(normals_all[:, 0])).reshape(-1)]))
        normals_all[stale] = _gradient_normals_t(table, sdf_nan, coords[stale])
        self.normals = normals_all
        cand_mask = (field.weight_sum > 0) & (field.value_sum <= 0)
        block_of = block_keys(coords, self.cfg.block_voxels)
        n_interior = int(cand_mask.sum()) if region_blocks is not None else None
        if region_blocks is not None:
            cand_mask &= torch.isin(block_of, region_blocks)
        cand = coords[cand_mask]
        normals = normals_all[cand_mask]
        new_blocks = dilated_block_keys(coords[touched_rows], 8)
        self.occupied = new_blocks if self.occupied is None else torch.unique(torch.cat([self.occupied, new_blocks]))
        units = self.units
        count_rays = units is not None and units.unit == 'ray'
        votes = estimate_candidate_free_space_frames_gpu(
            cand, self.voxel, views, candidate_normals=normals, count_rays=count_rays,
            maximum_rays_per_view=self.cfg.max_rays_per_frame, ray_radius_m=self.alpha_radius,
            endpoint_clearance_m=self.alpha_clearance, minimum_free_rays_per_frame=self.cfg.free_space_min_rays_per_frame,
            device=self.device, occupied_blocks=self.occupied)
        counts, ray_counts = votes if count_rays else (votes, None)
        voted = torch.from_numpy(counts >= 1).to(dev)
        cand_rows = torch.nonzero(cand_mask).reshape(-1)
        fresh_sel = voted & (field.last_pass_block[cand_rows] != c)      # at most one pass vote per block
        newly = cand_rows[fresh_sel]
        if units is not None and not retry:
            units.add_passes(field, newly, torch.from_numpy(ray_counts if count_rays else counts).to(dev)[fresh_sel])
        field.pass_count[newly] += 1
        field.last_pass_block[newly] = c
        stats = {"candidates": int(len(cand)), "voted": int(voted.sum()), "newly_voted": int(len(newly))}
        if n_interior is not None:
            stats["interior_nodes"] = n_interior
        return coords, raw, block_of, newly, stats

    def after_evict(self, evicted_mask):
        """Keep the normals row-aligned after the persistent field dropped the evicted rows."""
        if self.normals is not None and len(self.normals) == len(evicted_mask):
            self.normals = self.normals[~evicted_mask]

    def release(self):
        """Free the crossing-test state before the final extraction."""
        self.normals = None
        self.occupied = None

    def evidence(self, field: SparseField, rows):
        """(h, p) of the given persistent rows for step IV: block counts, or the rescaled frame / ray totals."""
        if self.units is not None:
            return self.units.counts(field, rows)
        return field.hit_count[rows], field.pass_count[rows]
