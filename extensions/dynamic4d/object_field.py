"""Object-frame TSDF of the dynamic extension (vendored from ``gubmap.online_object_field`` in the mode the extension uses,
``tsdf``: no pass evidence, no regularisation; unchanged arithmetic).

Every observation (the target's returns in the LiDAR frame + the tracked object_from_lidar pose) is integrated into the
field of its 0.5 s chunk: 5 along-ray samples per return within +-truncation, value = -normalised offset, weight =
1 - 0.35 |value|, trilinear splat onto a 3 cm lattice (``NodeTSDF``, the plain path of ``SparseContinuousTSDF``).  On
demand the chunk fields are summed (``merge_chunk_fields``) and Surface Nets extracts the object mesh -- the same
extractor as the map (``utils/mesher.py``) with a node-weight gate of 0.2.
"""
from __future__ import annotations

import time

import numpy as np
import torch

from utils.mesher import extract_surface_nets_t
from utils.sampler import deterministic_subsample_indices

CUBE_CORNERS = np.asarray([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0], [0, 0, 1], [1, 0, 1], [0, 1, 1], [1, 1, 1]], dtype=np.int64)
_PACK_OFFSET = 1 << 20


def pack_keys(keys: np.ndarray) -> np.ndarray:
    """Pack integer (N,3) lattice keys into sortable int64 scalars."""
    keys = np.asarray(keys, dtype=np.int64).reshape(-1, 3)
    return (keys[:, 0] + _PACK_OFFSET) * (1 << 42) + (keys[:, 1] + _PACK_OFFSET) * (1 << 21) + (keys[:, 2] + _PACK_OFFSET)


class NodeTSDF:
    """Sparse TSDF as a dict: lattice key -> [weighted value sum, weight sum, observed-frame mask]."""

    def __init__(self, voxel_size_m: float, truncation_m: float):
        if voxel_size_m <= 0.0 or truncation_m <= 0.0:
            raise ValueError("voxel size and truncation must be positive")
        self.voxel_size = float(voxel_size_m)
        self.truncation = float(truncation_m)
        self.nodes: dict[tuple[int, int, int], list[float | int]] = {}

    def integrate_lidar(self, frame_number: int, lidar_points: np.ndarray, world_from_lidar: np.ndarray, *,
                        max_rays: int = 12_000, samples_per_ray: int = 5) -> None:
        """Transform one scan with its pose and integrate it (frame bit ``1 << frame_number``)."""
        transform = np.asarray(world_from_lidar, dtype=np.float64)
        if transform.shape != (4, 4):
            raise ValueError("world_from_lidar must have shape (4, 4)")
        source = np.asarray(lidar_points, dtype=np.float64)
        if source.ndim != 2 or source.shape[1] != 3:
            raise ValueError("lidar_points must have shape (N, 3)")
        if frame_number < 0:
            raise ValueError("frame_number must be non-negative")
        if samples_per_ray < 2:
            raise ValueError("samples_per_ray must be at least two")
        indices = deterministic_subsample_indices(len(source), max_rays)
        points = np.ascontiguousarray(source[indices])
        truncation = np.full(len(source), self.truncation, dtype=np.float64)[indices]
        ray_weight = np.full(len(source), 1.0, dtype=np.float64)[indices]
        world = points @ transform[:3, :3].T + transform[:3, 3]
        origin = transform[:3, 3].reshape(3)
        ray = world - origin
        ranges = np.linalg.norm(ray, axis=1)
        valid = np.isfinite(ray).all(axis=1) & (ranges > truncation + 1e-6)
        ray, ranges, truncation, ray_weight = ray[valid], ranges[valid], truncation[valid], ray_weight[valid]
        directions = ray / np.maximum(ranges[:, None], 1e-12)
        if len(ranges) == 0:
            return
        half_steps = np.full(len(ranges), max(1, int(np.ceil((samples_per_ray - 1) / 2.0))), dtype=np.int64)
        step_ids = np.arange(-int(np.max(half_steps)), int(np.max(half_steps)) + 1)
        keep = np.abs(step_ids)[None, :] <= half_steps[:, None]
        normalized = step_ids[None, :] / half_steps[:, None]
        sample_ranges = ranges[:, None] + truncation[:, None] * normalized
        base_samples = origin[None, None, :] + directions[:, None, :] * sample_ranges[:, :, None]
        base_tsdf = -normalized
        base_weight = (1.0 - 0.35 * np.abs(base_tsdf)) * ray_weight[:, None]
        samples, tsdf, sample_weight = base_samples[keep], base_tsdf[keep], base_weight[keep]
        scaled = samples / self.voxel_size
        base = np.floor(scaled).astype(np.int64)
        fraction = scaled - base
        coordinate_chunks, value_chunks, weight_chunks = [], [], []
        for corner in CUBE_CORNERS:
            corner_weight = np.prod(np.where(corner[None, :] == 1, fraction, 1.0 - fraction), axis=1)
            weight = sample_weight * corner_weight
            kept = weight > 1e-5
            coordinate_chunks.append(base[kept] + corner)
            value_chunks.append(tsdf[kept] * weight[kept])
            weight_chunks.append(weight[kept])
        coordinates = np.concatenate(coordinate_chunks, axis=0)
        unique, inverse = np.unique(coordinates, axis=0, return_inverse=True)
        summed_values = np.bincount(inverse, weights=np.concatenate(value_chunks, axis=0))
        summed_weights = np.bincount(inverse, weights=np.concatenate(weight_chunks, axis=0))
        bit = 1 << int(frame_number)
        for coordinate, value, weight in zip(unique, summed_values, summed_weights, strict=True):
            key = tuple(int(item) for item in coordinate)
            node = self.nodes.get(key)
            if node is None:
                self.nodes[key] = [float(value), float(weight), bit]
            else:
                node[0] = float(node[0]) + float(value)
                node[1] = float(node[1]) + float(weight)
                node[2] = int(node[2]) | bit


def merge_chunk_fields(chunk_fields: list, voxel: float, truncation: float) -> NodeTSDF:
    """Plain union of the per-chunk fields: summed value / weight, mask bit c = observed in chunk c."""
    coords, values, weights, chunk_ids = [], [], [], []
    for c, f in enumerate(chunk_fields):
        if f is None or not f.nodes:
            continue
        keys = np.fromiter((x for k in f.nodes for x in k), dtype=np.int64, count=3 * len(f.nodes)).reshape(-1, 3)
        coords.append(keys)
        values.append(np.fromiter((float(n[0]) for n in f.nodes.values()), dtype=np.float64, count=len(f.nodes)))
        weights.append(np.fromiter((float(n[1]) for n in f.nodes.values()), dtype=np.float64, count=len(f.nodes)))
        chunk_ids.append(np.full(len(f.nodes), c, dtype=np.int64))
    merged = NodeTSDF(voxel, truncation)
    if not coords:
        return merged
    coords = np.concatenate(coords); values = np.concatenate(values); weights = np.concatenate(weights); chunk_ids = np.concatenate(chunk_ids)
    uniq, inverse = np.unique(pack_keys(coords), return_inverse=True)
    inverse = inverse.reshape(-1)
    v = np.bincount(inverse, weights=values, minlength=len(uniq))
    w = np.bincount(inverse, weights=weights, minlength=len(uniq))
    masks = np.zeros(len(uniq), dtype=np.uint64)
    np.bitwise_or.at(masks, inverse, (np.uint64(1) << chunk_ids.astype(np.uint64)))
    first = np.zeros(len(uniq), dtype=np.int64)
    first[inverse[::-1]] = np.arange(len(inverse))[::-1]
    rep = coords[first]
    for row in range(len(uniq)):
        merged.nodes[(int(rep[row, 0]), int(rep[row, 1]), int(rep[row, 2]))] = [float(v[row]), float(w[row]), int(masks[row])]
    return merged


class ObjectField:
    """Chunked object-frame TSDF with on-demand mesh extraction (``field.mesh`` = (vertices, faces), object frame)."""

    def __init__(self, *, voxel_size_m: float = 0.03, truncation_m: float = 0.06, chunk_seconds: float = 0.5,
                 max_rays_per_frame: int = 20_000, samples_per_ray: int = 5, minimum_node_weight: float = 0.2,
                 max_chunks: int = 64, device: str = "cpu"):
        self.device = device
        self.voxel, self.truncation = float(voxel_size_m), float(truncation_m)
        self.chunk_ns = int(float(chunk_seconds) * 1e9)
        self.max_rays, self.samples = int(max_rays_per_frame), int(samples_per_ray)
        self.min_weight, self.max_chunks = float(minimum_node_weight), int(max_chunks)
        self.t0_ns = None
        self.chunks: dict[int, NodeTSDF] = {}
        self.local: dict[int, int] = {}
        self.n_obs = self.n_obs_at_mesh = 0
        self.last_stamp_ns = None
        self.mesh = None
        self.mesh_stamp_ns, self.mesh_count = None, 0
        self.integrate_s = 0.0
        self.history: list[dict] = []

    def add_observation(self, stamp_ns: int, object_from_lidar: np.ndarray, points_lidar: np.ndarray) -> int:
        """Integrate one observation; returns its chunk."""
        t = time.time()
        stamp_ns = int(stamp_ns)
        if self.t0_ns is None:
            self.t0_ns = stamp_ns
        c = min(max(int((stamp_ns - self.t0_ns) // self.chunk_ns), 0), self.max_chunks - 1)
        f = self.chunks.get(c)
        if f is None:
            f = self.chunks[c] = NodeTSDF(self.voxel, self.truncation)
            self.local[c] = 0
        f.integrate_lidar(self.local[c], np.asarray(points_lidar, dtype=np.float64), np.asarray(object_from_lidar, dtype=np.float64),
                          max_rays=self.max_rays, samples_per_ray=self.samples)
        self.local[c] += 1
        self.n_obs += 1
        self.last_stamp_ns = max(stamp_ns, self.last_stamp_ns or stamp_ns)
        self.integrate_s += time.time() - t
        return c

    @property
    def dirty(self) -> bool:
        return self.n_obs > self.n_obs_at_mesh

    @property
    def current_chunk(self) -> int:
        return max(self.chunks) if self.chunks else -1

    def extract(self, *, final: bool = False) -> dict:
        """Merge the chunk fields and extract the current surface."""
        t0 = time.time()
        field = merge_chunk_fields([self.chunks.get(c) for c in range(self.current_chunk + 1)], self.voxel, self.truncation)
        keys_list = list(field.nodes.keys())
        n = len(keys_list)
        keys = np.fromiter((x for k in keys_list for x in k), dtype=np.int64, count=3 * n).reshape(-1, 3)
        values = np.fromiter((float(field.nodes[k][0]) for k in keys_list), dtype=np.float64, count=n)
        weights = np.fromiter((float(field.nodes[k][1]) for k in keys_list), dtype=np.float64, count=n)
        sdf = values / np.maximum(weights, 1e-12)
        dev = torch.device(self.device)
        verts, faces = extract_surface_nets_t(torch.from_numpy(keys).to(dev), torch.from_numpy(sdf).to(dev), torch.from_numpy(weights).to(dev),
                                              self.voxel, minimum_node_weight=self.min_weight, device=self.device)[:2]
        self.mesh = (np.asarray(verts, dtype=np.float64).reshape(-1, 3), np.asarray(faces, dtype=np.int64).reshape(-1, 3))
        self.mesh_stamp_ns, self.n_obs_at_mesh = self.last_stamp_ns, self.n_obs
        self.mesh_count += 1
        row = {"mode": "tsdf", "observations": self.n_obs, "chunks": self.current_chunk + 1, "nodes": n,
               "vertices": int(len(self.mesh[0])), "faces": int(len(self.mesh[1])), "total_s": round(time.time() - t0, 4),
               "mesh_stamp_ns": self.mesh_stamp_ns, "final": bool(final)}
        self.history.append(row)
        return row


def build_prefix_field(tracks, prefix_rows) -> ObjectField:
    """Fuse the cached prefix rows (the observations and poses that built the initial model)."""
    field = ObjectField()
    for j in np.asarray(prefix_rows, int):
        field.add_observation(tracks.stamp_ns(j), tracks.object_from_lidar(j), tracks.row_points(j))
    return field
