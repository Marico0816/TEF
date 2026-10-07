"""Sparse signed-distance field on a regular lattice (the map representation).

TEF keeps two fields of this type:
  * the block-local field B_b, which accumulates the samples of one temporal block (step III, per scan), and
  * the persistent field, into which every finished block is merged (step III, per block) and which the
    regularised solve (step V) and the extraction (step VI) read.

Nodes are stored as sorted packed int64 keys with row-aligned accumulators on the device:
    value_sum, weight_sum        weighted sum of normalised signed distances and its weight
    mask                         per-frame bits (block field) or per-block bits (persistent field)
    hit_count, pass_count        exact numbers of blocks with surface / free-space evidence (h, p)
    last_pass_block              block id of the last pass vote (at most one pass vote per block)
    weight_max                   largest single sample weight per node (block field, bounded block weight)
    solved                       regularised value (NaN = never solved)
    hit_units, pass_units        evidence totals for the frame / ray counting units (empty by default)
"""
from __future__ import annotations

import numpy as np
import torch

_CUBE_CORNERS = np.array(
    [[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0], [0, 0, 1], [1, 0, 1], [0, 1, 1], [1, 1, 1]], dtype=np.int64
)
_PACK_OFFSET = 1 << 20
_NEIGHBOUR_OFFSETS = np.array([(x, y, z) for x in (-1, 0, 1) for y in (-1, 0, 1) for z in (-1, 0, 1)], dtype=np.int64)
_BLOCK_OFFSETS = torch.tensor([(x, y, z) for x in (-1, 0, 1) for y in (-1, 0, 1) for z in (-1, 0, 1)], dtype=torch.int64)


# ----------------------------------------------------------------------------------------------------------------------
# keys, bits and lattice lookup
# ----------------------------------------------------------------------------------------------------------------------
def mask_bit(index: int) -> int:
    """Signed int64 mask bit for a block (or frame) index: bit ``index % 64`` (bit 63 is the sign bit, returned as -2**63)."""

    b = int(index) % 64
    return (1 << b) if b < 63 else -(1 << 63)


def pack_keys_t(keys):
    """Pack signed lattice coordinates (N,3) into int64 keys, 21 bits per axis."""

    keys = keys.to(torch.int64)
    return (keys[:, 0] + _PACK_OFFSET) * (1 << 42) + (keys[:, 1] + _PACK_OFFSET) * (1 << 21) + (keys[:, 2] + _PACK_OFFSET)


def unpack_keys_t(packed):
    """Inverse of :func:`pack_keys_t`: (N,) int64 -> (N,3) int64 lattice coordinates."""

    offset = 1 << 20
    z = packed % (1 << 21) - offset
    rest = packed // (1 << 21)
    y = rest % (1 << 21) - offset
    x = rest // (1 << 21) - offset
    return torch.stack([x, y, z], dim=1)


def popcount(x):
    """Population count of int64 tensors (masks use at most 64 bits; bit 63 handled as a logical shift)."""

    x = x.clone()
    count = torch.zeros_like(x)
    for _ in range(64):
        count += x & 1
        x = x >> 1
        x = x & 0x7FFFFFFFFFFFFFFF
    return count


class LatticeTable:
    """Sorted packed-key table with O(log N) membership lookup on the device."""

    def __init__(self, keys, device: str):
        if isinstance(keys, torch.Tensor):
            packed = pack_keys_t(keys.to(device, torch.int64).reshape(-1, 3))
        else:
            keys = np.asarray(keys, dtype=np.int64).reshape(-1, 3)
            packed = pack_keys_t(torch.from_numpy(keys).to(device))
        self.order = torch.argsort(packed)
        self.sorted_packed = packed[self.order]
        self.size = int(len(packed))
        self.device = device

    @classmethod
    def from_sorted_packed(cls, sorted_packed, device: str):
        """Wrap packed keys that are already sorted ascending (e.g. ``SparseField.keys``): no argsort, no copy."""

        self = cls.__new__(cls)
        self.sorted_packed = sorted_packed.to(device, torch.int64).reshape(-1)
        self.order = None  # identity: row index == sorted position
        self.size = int(len(self.sorted_packed))
        self.device = device
        return self

    def lookup(self, packed_query):
        """Row index per query (-1 when absent)."""

        if self.size == 0:
            return torch.full(packed_query.shape, -1, dtype=torch.int64, device=packed_query.device)
        pos = torch.searchsorted(self.sorted_packed, packed_query)
        pos = torch.clamp(pos, max=self.size - 1)
        found = self.sorted_packed[pos] == packed_query
        rows = pos if self.order is None else self.order[pos]
        return torch.where(found, rows, torch.full_like(pos, -1))


# ----------------------------------------------------------------------------------------------------------------------
# blocks (64-voxel cubes) used for solve regions, extraction and eviction
# ----------------------------------------------------------------------------------------------------------------------
def block_keys(coords, block_voxels: int):
    """Packed key of the block that contains each lattice coordinate."""

    return pack_keys_t(torch.div(coords, block_voxels, rounding_mode="floor"))


def dilate_blocks(block_keys_t, block_voxels: int, rings: int):
    """Packed keys of the blocks within ``rings`` blocks (26-neighbourhood) of the given ones."""

    cur = torch.unique(block_keys_t)
    offs = _BLOCK_OFFSETS.to(cur.device)
    for _ in range(rings):
        xyz = unpack_keys_t(cur)
        cur = torch.unique(pack_keys_t((xyz[:, None, :] + offs[None, :, :]).reshape(-1, 3)))
    return cur


def dilated_block_keys(coords_t, block_voxels: int, batch: int = 2_000_000):
    """Sorted unique packed keys of the ``block_voxels``-blocks containing ``coords`` dilated by one voxel."""

    offs = torch.from_numpy(_NEIGHBOUR_OFFSETS).to(coords_t.device)
    parts = []
    for start in range(0, len(coords_t), batch):
        c = coords_t[start:start + batch]
        dil = (c[:, None, :] + offs[None, :, :]).reshape(-1, 3)
        parts.append(torch.unique(pack_keys_t(torch.div(dil, block_voxels, rounding_mode="floor"))))
    if not parts:
        return torch.zeros(0, dtype=torch.int64, device=coords_t.device)
    return torch.unique(torch.cat(parts))


# ----------------------------------------------------------------------------------------------------------------------
# splatting samples onto the lattice
# ----------------------------------------------------------------------------------------------------------------------
def splat_trilinear_t(samples, values, weights, voxel_size_m: float, *, track_weight_max: bool = False, minimum_weight: float = 1e-5):
    """Trilinear splat of weighted samples onto the lattice, summed per unique node.

    Each sample contributes ``weight * corner_weight`` to the eight corners of its cell; corners below ``minimum_weight``
    are dropped.  Returns ``(packed_keys, value_sum, weight_sum, weight_max)``; ``weight_max`` (the largest single
    corner-weighted sample weight per node, used by the bounded block weight) is None unless ``track_weight_max``.
    """

    device = samples.device
    scaled = samples / float(voxel_size_m)
    base = torch.floor(scaled).to(torch.int64)
    fraction = (scaled - base.to(scaled.dtype))
    corners = torch.from_numpy(_CUBE_CORNERS).to(device)
    key_parts, val_parts, w_parts = [], [], []
    for corner in corners:
        corner_weight = torch.where(corner[None, :] == 1, fraction, 1.0 - fraction).prod(dim=1)
        w = weights * corner_weight
        keep = w > minimum_weight
        key_parts.append(pack_keys_t(base[keep] + corner[None, :]))
        val_parts.append(values[keep] * w[keep])
        w_parts.append(w[keep])
    keys = torch.cat(key_parts)
    unique, inverse = torch.unique(keys, return_inverse=True)
    m = len(unique)
    value_sum = torch.zeros(m, dtype=values.dtype, device=device).index_add_(0, inverse, torch.cat(val_parts))
    weight_sum = torch.zeros(m, dtype=values.dtype, device=device).index_add_(0, inverse, torch.cat(w_parts))
    weight_max = None
    if track_weight_max:
        weight_max = torch.zeros(m, dtype=values.dtype, device=device).scatter_reduce_(0, inverse, torch.cat(w_parts), reduce="amax", include_self=True)
    return unique, value_sum, weight_sum, weight_max


# ----------------------------------------------------------------------------------------------------------------------
# the field
# ----------------------------------------------------------------------------------------------------------------------
class SparseField:
    """Sorted packed keys + row-aligned accumulators on a device; ``merge`` unions a splat (or a block field) into it."""

    _ROW_ALIGNED = ("keys", "value_sum", "weight_sum", "mask", "hit_count", "pass_count", "last_pass_block", "solved",
                    "hit_units", "pass_units", "weight_max")

    def __init__(self, voxel_size_m: float, truncation_m: float, device: str = "cuda", dtype=None):
        if voxel_size_m <= 0.0 or truncation_m <= 0.0:
            raise ValueError("voxel size and truncation must be positive")
        self.voxel_size = float(voxel_size_m)
        self.truncation = float(truncation_m)
        self.device = torch.device(device)
        self.dtype = dtype or torch.float64
        z = lambda dt=None: torch.zeros(0, dtype=self.dtype if dt is None else dt, device=self.device)   # noqa: E731
        self.keys = z(torch.int64)
        self.value_sum, self.weight_sum = z(), z()
        self.mask = z(torch.int64)
        self.hit_count, self.pass_count = z(torch.int32), z(torch.int32)
        self.last_pass_block = z(torch.int32)
        self.hit_units = self.pass_units = z()
        self.weight_max = z()
        self.solved = z()
        self.last_merge_inverse = (z(torch.int64), z(torch.int64))

    def __len__(self) -> int:
        return int(len(self.keys))

    def raw_sdf(self):
        """Fused (weighted mean) value s0; 0 where the node has no weight."""

        return torch.where(self.weight_sum > 0, self.value_sum / self.weight_sum.clamp_min(1e-12), torch.zeros_like(self.value_sum))

    def current_sdf(self):
        """Regularised value where solved, fused value elsewhere."""

        raw = self.raw_sdf()
        return torch.where(torch.isfinite(self.solved), self.solved, raw)

    def extraction_weight(self):
        """Node weight w read by the extraction gates."""

        return self.weight_sum

    def merge(self, packed_keys, value_sum, weight_sum, frame_bit: int, *, block_id: int | None = None, weight_max=None) -> int:
        """Add a splat or a block field (unique packed keys) into this field; returns the number of new nodes.

        ``block_id`` given (one merge per temporal block): the mask bit is that of the block, and ``hit_count`` of every
        incoming node is incremented by one -- at most one hit vote per block.  Otherwise ``frame_bit`` is used verbatim
        (per-scan accumulation into the block field)."""

        before = len(self.keys)
        all_keys = torch.cat([self.keys, packed_keys])
        unique, inverse = torch.unique(all_keys, return_inverse=True)
        m = len(unique)
        old_inv = inverse[:before]
        new_inv = inverse[before:]

        def _merged(old, new):
            out = torch.zeros(m, dtype=self.dtype, device=self.device)
            out.index_add_(0, old_inv, old)
            out.index_add_(0, new_inv, new.to(self.dtype))
            return out

        self.value_sum = _merged(self.value_sum, value_sum)
        self.weight_sum = _merged(self.weight_sum, weight_sum)
        mask = torch.zeros(m, dtype=torch.int64, device=self.device)
        mask[old_inv] = self.mask
        bit_value = int(frame_bit) if block_id is None else mask_bit(block_id)
        bit = torch.full((len(packed_keys),), bit_value, dtype=torch.int64, device=self.device)
        mask[new_inv] = torch.bitwise_or(mask[new_inv], bit)
        self.mask = mask
        hit_count = torch.zeros(m, dtype=torch.int32, device=self.device); hit_count[old_inv] = self.hit_count
        if block_id is not None:
            hit_count[new_inv] += 1
        self.hit_count = hit_count
        pass_count = torch.zeros(m, dtype=torch.int32, device=self.device); pass_count[old_inv] = self.pass_count; self.pass_count = pass_count
        last_pass = torch.full((m,), -1, dtype=torch.int32, device=self.device); last_pass[old_inv] = self.last_pass_block; self.last_pass_block = last_pass
        wmax = torch.zeros(m, dtype=self.dtype, device=self.device)
        if len(self.weight_max) == before:
            wmax[old_inv] = self.weight_max
        if weight_max is not None:
            wmax[new_inv] = torch.maximum(wmax[new_inv], weight_max.to(self.dtype))
        self.weight_max = wmax
        solved = torch.full((m,), float("nan"), dtype=self.dtype, device=self.device)
        solved[old_inv] = self.solved
        self.solved = solved
        self.last_merge_inverse = (old_inv, new_inv)
        self.keys = unique
        return m - before

    def bound_block_weights(self, mode: str = "sum"):
        """Bounded block weight, applied to a block field before it is merged into the persistent field.

        ``sum``: no change (sample-weighted fusion, P2).  ``max``: each node keeps its block mean but its block weight
        becomes min(largest single sample weight, block weight sum), so dense repeated sampling within one block does
        not add weight (T2).  Returns the per-node factor applied."""

        w = self.weight_sum
        if mode == "sum" or len(w) == 0:
            return torch.ones_like(w)
        if mode != "max":
            raise ValueError("block weight mode must be sum or max")
        if len(self.weight_max) != len(w):
            raise ValueError("max mode needs the per-node weight_max accumulator (splat with track_weight_max)")
        target = torch.minimum(self.weight_max, w)  # never above the sum (a single sample)
        factor = torch.where(w > 0, target / w.clamp_min(1e-12), torch.zeros_like(w))
        self.value_sum = self.value_sum * factor
        self.weight_sum = torch.where(w > 0, target, torch.zeros_like(w))
        return factor

    def evict(self, evict_mask):
        """Remove the selected rows from every row-aligned accumulator (keys stay sorted) and return what a later surface
        extraction needs of them on the CPU: packed ``keys``, current ``sdf``, ``weight`` and ``mask``."""

        n = len(self.keys)
        evict_mask = evict_mask.to(self.device, torch.bool)
        if len(evict_mask) != n:
            raise ValueError("evict_mask must have one entry per node")
        out = {"keys": self.keys[evict_mask].cpu(), "sdf": self.current_sdf()[evict_mask].cpu(),
               "weight": self.extraction_weight()[evict_mask].cpu(), "mask": self.mask[evict_mask].cpu()}
        keep = ~evict_mask
        for name in self._ROW_ALIGNED:
            t = getattr(self, name, None)
            if t is not None and len(t) == n:
                setattr(self, name, t[keep])
        return out
