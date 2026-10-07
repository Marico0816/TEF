"""Step II -- local support and sampling.

For each scan:
  1. the persistent local-support statistics (model/local_support.py) are updated with the scan's returns and queried
     for a local frame per return (normal n, tangent axes u, v, extents, thickness tau, confidence); returns without a
     matching level -- or every return when the local support is switched off -- keep the ray-orthogonal default frame;
  2. data-support limits clip the lateral extent of each matched return to the neighbouring returns of the same surface;
  3. each return yields normalised signed-distance samples along its ray (normal-projected distance, tapered and
     tau-Gaussian weights), and the samples nearest the surface are copied across a lateral disc in the local tangent
     plane (golden-angle antipodal pairs), giving the scan's sample set S_t.

The sampling functions below are the research implementation (``tef/samples.py``, ``tef/ellipsoids.py``) without the
options the paper does not use; their arithmetic is unchanged.
"""
from __future__ import annotations

import math
from typing import Any, Callable

import numpy as np
import torch

from model.local_support import LocalSupport


# ----------------------------------------------------------------------------------------------------------------------
# sample counts and the footprint sampler (research tef/samples.py)
# ----------------------------------------------------------------------------------------------------------------------
def _odd_cap(maximum_samples_per_ray: int) -> int:
    maximum_odd = int(maximum_samples_per_ray)
    if maximum_odd % 2 == 0:
        maximum_odd -= 1
    return maximum_odd


def _oddify_and_cap(log_density, maximum_odd: int):
    saturated = log_density >= math.log(float(maximum_odd))
    finite_request = torch.isfinite(log_density) & ~saturated
    requested = torch.ones_like(log_density, dtype=torch.int64)
    requested = torch.where(saturated, torch.full_like(requested, maximum_odd), requested)
    req_f = torch.clamp(torch.ceil(torch.exp(torch.where(finite_request, log_density, torch.zeros_like(log_density)))), min=1.0).to(torch.int64)
    requested = torch.where(finite_request, req_f, requested)
    requested = 1 + 2 * torch.ceil((requested - 1).to(torch.float64) / 2.0).to(torch.int64)
    return torch.clamp(requested, max=maximum_odd)


def footprint_sample_counts_t(scales_m, voxel_size_m: float, extent_sigma: float, maximum_samples_per_ray: int):
    """Odd lateral sample count per return from the disc area (no data-support limits)."""
    maximum_odd = _odd_cap(maximum_samples_per_ray)
    log_area = math.log(math.pi) + 2.0 * math.log(float(extent_sigma)) + torch.log(scales_m[:, 0]) + torch.log(scales_m[:, 1]) - 2.0 * math.log(float(voxel_size_m))
    return _oddify_and_cap(log_area, maximum_odd)


def supported_footprint_sample_counts_t(support_half_axes_m, voxel_size_m: float, maximum_samples_per_ray: int):
    """Odd lateral sample count per return from the support-limited half axes."""
    maximum_odd = _odd_cap(maximum_samples_per_ray)
    axes = support_half_axes_m
    log_voxel = math.log(float(voxel_size_m))
    neg_inf = torch.full((len(axes),), -float("inf"), dtype=axes.dtype, device=axes.device)
    log_density = neg_inf.clone()
    positive_u = axes[:, 0] > 0.0
    positive_v = axes[:, 1] > 0.0
    lu = torch.where(positive_u, math.log(2.0) + torch.log(torch.where(positive_u, axes[:, 0], torch.ones_like(axes[:, 0]))) - log_voxel, neg_inf)
    lv = torch.where(positive_v, math.log(2.0) + torch.log(torch.where(positive_v, axes[:, 1], torch.ones_like(axes[:, 1]))) - log_voxel, neg_inf)
    log_density = torch.maximum(log_density, lu)
    log_density = torch.maximum(log_density, lv)
    both = positive_u & positive_v
    la = torch.where(both, math.log(math.pi) + torch.log(torch.where(both, axes[:, 0], torch.ones_like(axes[:, 0]))) + torch.log(torch.where(both, axes[:, 1], torch.ones_like(axes[:, 1]))) - 2.0 * log_voxel, neg_inf)
    log_density = torch.maximum(log_density, la)
    return _oddify_and_cap(log_density, maximum_odd)


def footprint_samples_t(
    world_points: np.ndarray,
    sensor_origin_world: np.ndarray,
    *,
    max_rays: int,
    samples_per_ray: int,
    ray_truncation_m: np.ndarray,
    ray_tangent_scales_m: np.ndarray,
    ray_tangent_basis_world: np.ndarray,
    ray_tangent_support_limits_m: np.ndarray | None,
    ray_tau_m: np.ndarray | None,
    ray_confidence: np.ndarray,
    normal_projected_sdf: bool = True,
    footprint_extent_sigma: float = 2.5,
    max_footprint_samples_per_ray: int = 25,
    maximum_sample_step_m: float | None = None,
    voxel_size_m: float,
    device: str = "cuda",
):
    """Return device tensors ``(samples (S,3), tsdf (S,), weight (S,))`` in float64: the along-ray samples of every
    selected return, then the lateral (tangent-disc) copies of its three central samples."""

    if samples_per_ray < 2:
        raise ValueError("samples_per_ray must be at least two")
    dev = torch.device(device)
    f64 = torch.float64
    source = np.asarray(world_points, dtype=np.float64)
    n = len(source)
    indices = deterministic_subsample_indices(n, max_rays)

    def sel(arr, default=None):
        if arr is None:
            return None
        return torch.from_numpy(np.ascontiguousarray(np.asarray(arr, dtype=np.float64)[indices])).to(dev)

    points = sel(source)
    truncation = sel(ray_truncation_m)
    scales = sel(ray_tangent_scales_m)
    basis = sel(ray_tangent_basis_world)
    limits = sel(ray_tangent_support_limits_m)
    tau = sel(ray_tau_m)
    confidence = sel(ray_confidence)
    origin = torch.from_numpy(np.asarray(sensor_origin_world, dtype=np.float64).reshape(3)).to(dev)
    ray = points - origin
    ranges = torch.linalg.norm(ray, dim=1)
    valid = torch.isfinite(ray).all(dim=1) & (ranges > truncation + 1e-6) & (confidence > 0.0)
    ray, ranges, truncation, scales, basis, tau, confidence = ray[valid], ranges[valid], truncation[valid], scales[valid], basis[valid], tau[valid], confidence[valid]
    if limits is not None:
        limits = limits[valid]
    empty = (torch.zeros((0, 3), dtype=f64, device=dev), torch.zeros(0, dtype=f64, device=dev), torch.zeros(0, dtype=f64, device=dev))
    if len(ranges) == 0:
        return empty
    directions = ray / ranges.clamp_min(1e-12)[:, None]
    base_half_steps = max(1, int(math.ceil((samples_per_ray - 1) / 2.0)))
    half_steps = torch.full((len(ranges),), base_half_steps, dtype=torch.int64, device=dev)
    if maximum_sample_step_m is not None:
        if not np.isfinite(maximum_sample_step_m) or maximum_sample_step_m <= 0.0:
            raise ValueError("maximum_sample_step_m must be finite and positive")
        half_steps = torch.maximum(half_steps, torch.ceil(truncation / float(maximum_sample_step_m)).to(torch.int64))
    hmax = int(half_steps.max())
    step_ids = torch.arange(-hmax, hmax + 1, device=dev)
    keep = step_ids.abs()[None, :] <= half_steps[:, None]
    normalized = step_ids[None, :].to(f64) / half_steps[:, None].to(f64)
    offsets = truncation[:, None] * normalized
    sample_ranges = ranges[:, None] + offsets
    base_samples = origin[None, None, :] + directions[:, None, :] * sample_ranges[:, :, None]
    if normal_projected_sdf:
        band_normals = torch.cross(basis[:, 0, :], basis[:, 1, :], dim=1)
        band_normals = band_normals / torch.linalg.norm(band_normals, dim=1, keepdim=True).clamp_min(1e-9)
        incidence = (directions * band_normals).sum(dim=1).abs()
        has_basis = torch.isfinite(incidence) & (torch.linalg.norm(basis[:, 0, :], dim=1) > 1e-9)
        incidence = torch.where(has_basis, incidence, torch.ones_like(incidence)).clamp(0.05, 1.0)
        projected = offsets * incidence[:, None]
        base_tsdf = torch.clamp(-projected / truncation[:, None], -1.0, 1.0)
    else:
        projected = offsets
        base_tsdf = -normalized
    base_weight = (1.0 - 0.35 * base_tsdf.abs()) * confidence[:, None]      # tapered towards the truncation, times confidence
    m = projected / tau[:, None]
    base_weight = base_weight * torch.exp(-0.5 * m * m)                      # Gaussian in the normal offset / thickness
    samples = base_samples[keep]
    tsdf = base_tsdf[keep]
    weight = base_weight[keep]
    # lateral footprint: copies of the central samples across the tangent disc (antipodal golden-angle pairs)
    support_half_axes = None
    if limits is None:
        count = footprint_sample_counts_t(scales, voxel_size_m, footprint_extent_sigma, max_footprint_samples_per_ray)
    else:
        probability_half_axes = footprint_extent_sigma * scales
        support_half_axes = torch.minimum(probability_half_axes, limits)
        count = supported_footprint_sample_counts_t(support_half_axes, voxel_size_m, max_footprint_samples_per_ray)
    pair_count = torch.div(count - 1, 2, rounding_mode="floor")
    footprint_rays = pair_count > 0
    if bool(footprint_rays.any()):
        central = (step_ids.abs()[None, :] <= 1) & keep & footprint_rays[:, None]
        central_samples = base_samples[central]
        central_tsdf = base_tsdf[central]
        central_weight = base_weight[central]
        ray_index = torch.arange(len(ranges), device=dev)[:, None].expand(-1, keep.shape[1])
        central_ray = ray_index[central]
        maximum_pairs = int(pair_count.max())
        pair_id = torch.arange(maximum_pairs, device=dev, dtype=f64)[None, :]
        central_pair_count = pair_count[central_ray]
        active = pair_id < central_pair_count[:, None].to(f64)
        golden_angle = math.pi * (3.0 - math.sqrt(5.0))
        angle = (golden_angle * pair_id).expand(len(central_samples), -1).clone()
        radius = torch.sqrt((pair_id + 1.0) / torch.clamp(central_pair_count[:, None].to(f64), min=1.0))
        if support_half_axes is not None:
            major_is_v = support_half_axes[central_ray, 1] > support_half_axes[central_ray, 0]
            angle[:, 0] = torch.where(major_is_v, torch.full_like(angle[:, 0], 0.5 * math.pi), torch.zeros_like(angle[:, 0]))
            radius[:, 0] = 1.0
            if maximum_pairs > 1:
                angle[:, 1] = torch.where(major_is_v, torch.zeros_like(angle[:, 1]), torch.full_like(angle[:, 1], 0.5 * math.pi))
                radius[:, 1] = 1.0
        unit_u = radius * torch.cos(angle)
        unit_v = radius * torch.sin(angle)
        active_ray = central_ray[:, None].expand(-1, maximum_pairs)[active]
        active_center = torch.arange(len(central_samples), device=dev)[:, None].expand(-1, maximum_pairs)[active]
        active_u = unit_u[active]
        active_v = unit_v[active]
        if support_half_axes is None:
            lateral = footprint_extent_sigma * ((scales[active_ray, 0] * active_u)[:, None] * basis[active_ray, 0] + (scales[active_ray, 1] * active_v)[:, None] * basis[active_ray, 1])
            tangent_weight = torch.exp(-0.5 * footprint_extent_sigma ** 2 * (active_u * active_u + active_v * active_v))
        else:
            offset_u = support_half_axes[active_ray, 0] * active_u
            offset_v = support_half_axes[active_ray, 1] * active_v
            lateral = offset_u[:, None] * basis[active_ray, 0] + offset_v[:, None] * basis[active_ray, 1]
            tangent_weight = torch.exp(-0.5 * ((offset_u / scales[active_ray, 0]) ** 2 + (offset_v / scales[active_ray, 1]) ** 2))
        repeated_tsdf = central_tsdf[active_center]
        repeated_weight = central_weight[active_center] * tangent_weight
        samples = torch.cat([samples, central_samples[active_center] + lateral, central_samples[active_center] - lateral], dim=0)
        tsdf = torch.cat([tsdf, repeated_tsdf, repeated_tsdf], dim=0)
        weight = torch.cat([weight, repeated_weight, repeated_weight], dim=0)
    return samples, tsdf, weight


def deterministic_subsample_indices(length: int, maximum: int) -> np.ndarray:
    """Return stable row indices so per-ray metadata stays aligned."""

    if length < 0:
        raise ValueError("length cannot be negative")
    if maximum <= 0:
        raise ValueError("maximum must be positive")
    if length <= maximum:
        return np.arange(length, dtype=np.int64)
    return np.linspace(0, length - 1, maximum, dtype=np.int64)


# ----------------------------------------------------------------------------------------------------------------------
# neighbours, default ray frame and data-support limits (research tef/ellipsoids.py)
# ----------------------------------------------------------------------------------------------------------------------
def _scipy_knn(reference: np.ndarray, queries: np.ndarray, neighbor_count: int) -> tuple[np.ndarray, np.ndarray]:
    from scipy.spatial import cKDTree

    count = min(max(int(neighbor_count), 1), len(reference))
    distance, index = cKDTree(np.asarray(reference, dtype=np.float64)).query(np.asarray(queries, dtype=np.float64), k=count, workers=-1)
    return (np.asarray(distance, dtype=np.float64).reshape(len(queries), count), np.asarray(index, dtype=np.int64).reshape(len(queries), count))


class FrameNeighbours:
    """One KD-tree build and one k-NN query per scan (on the CPU, in the preprocessing thread), read by the data-support
    limits.  The scan is queried in the LiDAR frame; the limits use world coordinates of the same rigid point set, so the
    neighbour indices coincide up to exact distance ties."""

    def __init__(self, points_lidar: np.ndarray, query_indices: np.ndarray, neighbor_count: int = 32):
        from scipy.spatial import cKDTree

        points = np.asarray(points_lidar, dtype=np.float64)
        self.count = min(max(int(neighbor_count), 1), len(points))
        self.query_indices = np.asarray(query_indices, dtype=np.int64).reshape(-1)
        self.tree = cKDTree(points)
        _, index = self.tree.query(points[self.query_indices], k=self.count, workers=-1)
        self.index = np.asarray(index, dtype=np.int64).reshape(len(self.query_indices), self.count)
        self._row_of = None

    def rows_for(self, global_indices: np.ndarray) -> np.ndarray:
        """Row positions of the given global point indices (must be among query_indices)."""

        if self._row_of is None:
            self._order = np.argsort(self.query_indices, kind="stable")
            self._sorted = self.query_indices[self._order]
            self._row_of = True
        pos = np.searchsorted(self._sorted, global_indices)
        pos = np.minimum(pos, len(self._sorted) - 1)
        if not np.array_equal(self._sorted[pos], global_indices):
            raise ValueError("global_indices must be a subset of query_indices")
        return self._order[pos]


def _normalize_rows_t(values):
    norms = torch.linalg.norm(values, dim=1, keepdim=True)
    return values / torch.clamp(norms, min=1e-12)


def ray_tangent_basis_world_t(points_lidar, world_from_lidar):
    """Torch twin of ``_ray_tangent_basis_world``: (N,2,3) fallback basis normal to each ray."""

    rotation = world_from_lidar[:3, :3]
    direction = _normalize_rows_t(points_lidar @ rotation.T)
    reference = torch.zeros_like(direction); reference[:, 2] = 1.0
    use_x = direction[:, 2].abs() > 0.90
    reference[use_x] = torch.tensor([1.0, 0.0, 0.0], dtype=direction.dtype, device=direction.device)
    tangent_u = _normalize_rows_t(torch.cross(direction, reference, dim=1))
    tangent_v = _normalize_rows_t(torch.cross(direction, tangent_u, dim=1))
    invalid = ~torch.isfinite(tangent_u).all(dim=1) | ~torch.isfinite(tangent_v).all(dim=1)
    if bool(invalid.any()):
        tangent_u[invalid] = torch.tensor([1.0, 0.0, 0.0], dtype=direction.dtype, device=direction.device)
        tangent_v[invalid] = torch.tensor([0.0, 1.0, 0.0], dtype=direction.dtype, device=direction.device)
    return torch.stack([tangent_u, tangent_v], dim=1)


def estimate_ray_tangent_support_limits_t(
    points_lidar: np.ndarray,
    world_from_lidar: np.ndarray,
    local_frames: dict[str, Any],
    *,
    query_indices: np.ndarray | None = None,
    fallback_limit_m: float,
    neighbor_count: int = 32,
    normal_offset_sigma: float = 2.5,
    maximum_normal_slope: float = 0.35,
    maximum_normal_angle_deg: float = 55.0,
    connectivity_spacing_factor: float = 2.5,
    boundary_margin_factor: float = 0.5,
    device: str = "cuda",
    knn: Callable = _scipy_knn,
    frame_neighbours: "FrameNeighbours | None" = None,
) -> np.ndarray:
    """Half-extents (N,2) along the two tangent axes that the lateral disc of each matched return may cover: the extent
    of the connected neighbouring returns that lie on the same local plane (normal offset <= ``normal_offset_sigma`` tau,
    compatible normal, continuous depth), plus a margin of ``boundary_margin_factor`` x the local spacing.  Unmatched
    returns get ``fallback_limit_m``."""

    dev = torch.device(device)
    points = np.asarray(points_lidar, dtype=np.float64)
    transform = np.asarray(world_from_lidar, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or transform.shape != (4, 4):
        raise ValueError("points_lidar must be (N, 3) and world_from_lidar (4, 4)")
    if fallback_limit_m < 0.0 or not np.isfinite(fallback_limit_m):
        raise ValueError("fallback_limit_m must be finite and non-negative")
    if neighbor_count < 2:
        raise ValueError("neighbor_count must be at least two")
    if (normal_offset_sigma <= 0.0 or maximum_normal_slope <= 0.0 or not 0.0 <= maximum_normal_angle_deg <= 90.0
            or connectivity_spacing_factor <= 0.0 or boundary_margin_factor < 0.0):
        raise ValueError("support-limit factors are invalid")
    required = ("tangent_basis_world", "tangent_scales_m", "tau_m", "matched", "endpoint_normal_world", "endpoint_normal_reliable")
    missing = [name for name in required if name not in local_frames]
    if missing:
        raise ValueError(f"local frames are missing fields: {missing}")
    basis_np = np.asarray(local_frames["tangent_basis_world"], dtype=np.float64)
    scales_np = np.asarray(local_frames["tangent_scales_m"], dtype=np.float64)
    tau_np = np.asarray(local_frames["tau_m"], dtype=np.float64).reshape(-1)
    matched_np = np.asarray(local_frames["matched"], dtype=bool).reshape(-1)
    count = len(points)
    if basis_np.shape != (count, 2, 3) or scales_np.shape != (count, 2) or len(tau_np) != count or len(matched_np) != count:
        raise ValueError("local-frame arrays do not match the LiDAR frame")
    selected = np.arange(count, dtype=np.int64) if query_indices is None else np.asarray(query_indices, dtype=np.int64).reshape(-1)
    if np.any(selected < 0) or np.any(selected >= count):
        raise ValueError("query_indices are outside the LiDAR frame")
    support_limits = np.full((count, 2), fallback_limit_m, dtype=np.float64)
    query_global_np = selected[matched_np[selected]]
    if len(query_global_np) == 0 or count < 2:
        return support_limits
    points_world = points @ transform[:3, :3].T + transform[:3, 3]
    if frame_neighbours is not None and neighbor_count <= frame_neighbours.count:
        candidate_np = frame_neighbours.index[frame_neighbours.rows_for(query_global_np), :neighbor_count]
    else:
        _, candidate_np = knn(points_world, points_world[query_global_np], neighbor_count)
    pw = torch.from_numpy(np.ascontiguousarray(points_world)).to(dev)
    q = torch.from_numpy(query_global_np).to(dev)
    cand = torch.from_numpy(candidate_np).to(dev)  # (Q,k)
    Q, k = cand.shape
    row = torch.arange(Q, device=dev)
    self_column = torch.argmax((cand == q[:, None]).to(torch.int64), dim=1)
    order = torch.arange(k, device=dev)[None, :].repeat(Q, 1)
    order[:, 0] = self_column
    order[row, self_column] = 0
    cand = torch.gather(cand, 1, order)
    basis = torch.from_numpy(basis_np).to(dev)
    endpoint_normal = torch.from_numpy(np.asarray(local_frames["endpoint_normal_world"], dtype=np.float64)).to(dev)
    endpoint_reliable = torch.from_numpy(np.asarray(local_frames["endpoint_normal_reliable"], dtype=bool)).to(dev)
    tau = torch.from_numpy(tau_np).to(dev)
    query_basis = basis[q]
    query_u = _normalize_rows_t(query_basis[:, 0])
    query_v = _normalize_rows_t(query_basis[:, 1] - (query_basis[:, 1] * query_u).sum(dim=1, keepdim=True) * query_u)
    query_normal = _normalize_rows_t(torch.cross(query_u, query_v, dim=1))
    delta = pw[cand] - pw[q][:, None, :]
    local_u = (delta * query_u[:, None, :]).sum(dim=2)
    local_v = (delta * query_v[:, None, :]).sum(dim=2)
    local_n = (delta * query_normal[:, None, :]).sum(dim=2)
    tangent_distance = torch.sqrt(local_u ** 2 + local_v ** 2)
    candidate_normal = endpoint_normal[cand]
    candidate_reliable = endpoint_reliable[cand]
    normal_cosine = (candidate_normal * query_normal[:, None, :]).sum(dim=2).abs()
    normal_compatible = (~candidate_reliable) | (normal_cosine >= float(np.cos(np.deg2rad(maximum_normal_angle_deg))))
    plane_compatible = normal_compatible & torch.isfinite(tangent_distance) & torch.isfinite(local_n) & (local_n.abs() <= normal_offset_sigma * tau[q][:, None])
    plane_compatible[:, 0] = True
    inf = torch.full_like(tangent_distance, float("inf"))
    positive_distance = torch.where(plane_compatible & (tangent_distance > 1e-12), tangent_distance, inf)
    nearest = torch.sort(positive_distance, dim=1).values[:, : min(3, k)]
    finite_count = torch.isfinite(nearest).sum(dim=1)
    median_column = torch.clamp((finite_count - 1) // 2, min=0)
    local_spacing = nearest[row, median_column]
    local_spacing = torch.where(finite_count > 0, local_spacing, torch.zeros_like(local_spacing))
    denominator = torch.maximum(tangent_distance, local_spacing[:, None])
    den_ok = denominator > 1e-12
    den_safe = torch.where(den_ok, denominator, torch.ones_like(denominator))
    normal_slope = torch.where(den_ok, local_n.abs() / den_safe, torch.zeros_like(local_n))
    sensor_origin = torch.from_numpy(transform[:3, 3].copy()).to(dev)
    candidate_ray = pw[cand] - sensor_origin
    candidate_range = torch.linalg.norm(candidate_ray, dim=2)
    candidate_direction = candidate_ray / torch.clamp(candidate_range[:, :, None], min=1e-12)
    plane_constant = (query_normal * (pw[q] - sensor_origin)).sum(dim=1)
    ray_plane_denominator = (candidate_direction * query_normal[:, None, :]).sum(dim=2)
    intersection_reliable = ray_plane_denominator.abs() > 1e-3
    rp_safe = torch.where(intersection_reliable, ray_plane_denominator, torch.ones_like(ray_plane_denominator))
    predicted_range = torch.where(intersection_reliable, plane_constant[:, None] / rp_safe, torch.zeros_like(candidate_range))
    depth_residual = (candidate_range - predicted_range).abs()
    depth_slope = torch.where(den_ok, depth_residual / den_safe, torch.zeros_like(depth_residual))
    depth_continuous = (normal_slope <= maximum_normal_slope) & ((~intersection_reliable) | ((predicted_range > 0.0) & (depth_slope <= maximum_normal_slope)))
    valid = plane_compatible & depth_continuous
    valid[:, 0] = True
    connection_radius = connectivity_spacing_factor * local_spacing
    connected = torch.zeros_like(valid)
    connected[:, 0] = True
    for column in range(1, k):
        delta_u = local_u[:, column, None] - local_u[:, :column]
        delta_v = local_v[:, column, None] - local_v[:, :column]
        previous_distance = torch.sqrt(delta_u ** 2 + delta_v ** 2)
        touches = (connected[:, :column] & (previous_distance <= connection_radius[:, None])).any(dim=1)
        connected[:, column] = valid[:, column] & touches
    axis_margin = boundary_margin_factor * local_spacing
    zero = torch.zeros_like(local_u)
    positive_u = torch.where(connected & (local_u > 0.0), local_u, zero).max(dim=1).values
    negative_u = torch.where(connected & (local_u < 0.0), -local_u, zero).max(dim=1).values
    positive_v = torch.where(connected & (local_v > 0.0), local_v, zero).max(dim=1).values
    negative_v = torch.where(connected & (local_v < 0.0), -local_v, zero).max(dim=1).values
    query_limits = torch.stack([torch.minimum(positive_u + axis_margin, negative_u + axis_margin), torch.minimum(positive_v + axis_margin, negative_v + axis_margin)], dim=1)
    query_limits = torch.where(torch.isfinite(query_limits), query_limits, torch.zeros_like(query_limits)).clamp_min(0.0)
    support_limits[query_global_np] = query_limits.cpu().numpy()
    return support_limits


# ----------------------------------------------------------------------------------------------------------------------
# step II as used by the mapper
# ----------------------------------------------------------------------------------------------------------------------
class Sampler:
    """Local-support update + query (II.1), data-support limits (II.2) and ray / lateral samples (II.3) of one scan."""

    def __init__(self, cfg, device: str):
        self.cfg, self.device = cfg, device
        self.voxel, self.truncation = cfg.voxel_m, cfg.truncation_m
        self.local_support = None if not cfg.local_support else LocalSupport(
            tuple(cfg.local_support_voxels_m), minimum_points=int(cfg.local_support_min_points), device=device,
            maximum_planarity=float(cfg.local_support_max_planarity))
        self.stats = {"queried_rays": 0, "matched_rays": 0}

    def release(self):
        """Free the local-support statistics before the final extraction (they are not needed any more)."""
        self.local_support = None

    def local_frame(self, points_lidar, world_from_lidar, selected) -> dict:
        """II.1: update the local-support statistics with every return of the scan, then query a frame for the selected ones."""
        cfg = self.cfg
        pts_l, T_wl = points_lidar, world_from_lidar
        world = np.asarray(pts_l, dtype=np.float64) @ T_wl[:3, :3].T + T_wl[:3, 3]
        world_t = torch.from_numpy(np.ascontiguousarray(world)).to(self.device)
        if self.local_support is not None:
            self.local_support.update(world_t)
        rows = torch.from_numpy(np.ascontiguousarray(selected)).to(self.device)
        basis = ray_tangent_basis_world_t(torch.from_numpy(np.ascontiguousarray(pts_l)).to(self.device)[rows],
                                          torch.from_numpy(np.ascontiguousarray(T_wl, dtype=np.float64)).to(self.device))
        origin = torch.from_numpy(np.asarray(T_wl, dtype=np.float64)[:3, 3]).to(self.device)
        to_sensor = origin - world_t[rows]
        to_sensor = to_sensor / torch.linalg.norm(to_sensor, dim=1, keepdim=True).clamp_min(1e-9)
        default_tau = self.truncation / cfg.normal_truncation_sigma
        if self.local_support is None:            # every return takes the default frame (research --ablate-band-field)
            nq = int(len(rows))
            queried = {"tangent_basis_world": basis.cpu().numpy(), "tangent_scales_m": np.full((nq, 2), self.voxel),
                       "tau_m": np.full(nq, default_tau), "confidence": np.ones(nq), "matched": np.zeros(nq, dtype=bool),
                       "endpoint_normal_world": (-to_sensor).cpu().numpy(), "queried_rays": nq, "matched_rays": 0}
        else:
            queried = self.local_support.query(world_t[rows], fallback_scale_m=self.voxel, fallback_tau_m=default_tau,
                                               ray_basis_world=basis, ray_normals_world=to_sensor)
        n = len(world)
        full = {"tangent_basis_world": np.zeros((n, 2, 3)), "tangent_scales_m": np.full((n, 2), self.voxel),
                "tau_m": np.full(n, default_tau), "confidence": np.ones(n), "matched": np.zeros(n, dtype=bool),
                "endpoint_normal_world": np.zeros((n, 3)), "endpoint_normal_reliable": np.zeros(n, dtype=bool)}
        full["tangent_basis_world"][:, 0, 0] = 1.0
        full["tangent_basis_world"][:, 1, 1] = 1.0
        full["endpoint_normal_world"][:, 2] = 1.0
        for key in ("tangent_basis_world", "tangent_scales_m", "tau_m", "confidence", "matched", "endpoint_normal_world"):
            full[key][selected] = queried[key]
        full["endpoint_normal_reliable"][selected] = queried["matched"]
        self.stats["queried_rays"] += queried["queried_rays"]
        self.stats["matched_rays"] += queried["matched_rays"]
        return full

    def samples(self, points_lidar, world_from_lidar, support: dict, selected, frame_neighbours):
        """II.2 + II.3: support limits, per-return truncation, then the ray and lateral samples (positions, values, weights)."""
        cfg = self.cfg
        pts_l, T_wl = points_lidar, world_from_lidar
        if cfg.data_support_limit:
            support["tangent_support_limits_m"] = estimate_ray_tangent_support_limits_t(
                pts_l, T_wl, support, query_indices=selected, fallback_limit_m=self.voxel,
                neighbor_count=cfg.support_neighbors, normal_offset_sigma=cfg.support_normal_offset_sigma,
                maximum_normal_slope=cfg.support_maximum_normal_slope,
                maximum_normal_angle_deg=cfg.match_maximum_normal_angle_deg,
                connectivity_spacing_factor=cfg.support_connectivity_factor,
                boundary_margin_factor=cfg.support_boundary_margin_factor, device=self.device, frame_neighbours=frame_neighbours,
            )
        matched = np.asarray(support["matched"], dtype=bool).reshape(-1)
        tau = np.asarray(support["tau_m"], dtype=np.float64).reshape(-1)
        trunc = np.full(len(pts_l), self.truncation, dtype=np.float64)
        if not cfg.per_return_truncation:         # research --ablate-per-return-truncation
            tau = np.full(len(pts_l), self.truncation / cfg.normal_truncation_sigma, dtype=np.float64)
        else:
            trunc[matched] = cfg.normal_truncation_sigma * tau[matched]
        world_pts = np.asarray(pts_l, dtype=np.float64) @ T_wl[:3, :3].T + T_wl[:3, 3]
        return footprint_samples_t(
            world_pts, np.asarray(T_wl, dtype=np.float64)[:3, 3], voxel_size_m=self.voxel, device=self.device,
            max_rays=cfg.max_rays_per_frame, samples_per_ray=cfg.samples_per_ray,
            ray_truncation_m=trunc, ray_tangent_scales_m=support["tangent_scales_m"],
            ray_tangent_basis_world=support["tangent_basis_world"], ray_tangent_support_limits_m=support.get("tangent_support_limits_m"),
            ray_tau_m=tau, ray_confidence=support["confidence"], normal_projected_sdf=cfg.normal_projected_sdf,
            footprint_extent_sigma=cfg.footprint_extent_sigma,
            max_footprint_samples_per_ray=cfg.max_footprint_samples_per_ray,
            maximum_sample_step_m=self.voxel * cfg.max_sample_step_factor,
        )
