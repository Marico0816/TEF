"""Step I -- load a scan and preprocess it.

Reads one frame of the canonical dataset, deskews its returns to the scan time with the given trajectory (2 ms bins),
keeps the deterministic 12 000-return subset used by every method of the shared evaluation protocol (``subsample
first``), optionally drops returns rejected by an external filter (keep mask), and builds the frame's k-nearest-neighbour
table used by the data-support limits of step II.  Runs one frame ahead on a CPU thread (see ``tef_mapping.py``).
"""
from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np

from dataset.trajectory import deskew_points_to_scan_frame
from utils.sampler import FrameNeighbours, deterministic_subsample_indices


def prepare_frame(dataset, frame_index: int, cfg, pose_fn, budget: dict):
    """Return ``(frame, points_lidar (N,3) float64, neighbours)`` with every per-return array kept aligned."""

    frame = dataset[int(frame_index)]
    if cfg.deskew and frame.point_time_offset_ns is not None:
        points, _ = deskew_points_to_scan_frame(frame.points_lidar, frame.point_time_offset_ns,
                                              int(frame.lidar_timestamp_ns), pose_fn, dataset.calibration.imu_from_lidar)
        frame = dataclasses.replace(frame, points_lidar=points)
    points = np.asarray(frame.points_lidar, dtype=np.float64)
    if cfg.subsample_first:
        frame, points = subsample_frame_first(frame, points, cfg.max_rays_per_frame)
        budget["frames"] = budget.get("frames", 0) + 1
        budget["points_after_subsample"] = budget.get("points_after_subsample", 0) + int(len(points))
    if cfg.input_keep_mask:
        frame, points = apply_input_keep_mask(frame, points, cfg.input_keep_mask, int(frame_index), cfg.subsample_first)
        budget["keep_mask_frames"] = budget.get("keep_mask_frames", 0) + 1
        budget["keep_mask_points_kept"] = budget.get("keep_mask_points_kept", 0) + int(len(points))
    selected = deterministic_subsample_indices(len(points), cfg.max_rays_per_frame)
    neighbours = FrameNeighbours(points, selected, int(cfg.support_neighbors))
    return frame, points, neighbours


def subsample_frame_first(frame, points_lidar, max_rays: int):
    """Keep only the deterministic ``max_rays`` subset (same rule and order as the common input package) in every
    per-point array used afterwards; a no-op when the frame already has <= max_rays points."""

    keep = deterministic_subsample_indices(len(points_lidar), int(max_rays))
    if len(keep) == len(points_lidar):
        return frame, points_lidar
    sl = lambda x: None if x is None else np.asarray(x)[keep]   # noqa: E731
    frame = dataclasses.replace(frame, points_lidar=np.asarray(frame.points_lidar)[keep], intensity=sl(getattr(frame, "intensity", None)),
                                point_time_offset_ns=sl(getattr(frame, "point_time_offset_ns", None)))
    return frame, np.asarray(points_lidar)[keep]


def apply_input_keep_mask(frame, points_lidar, mask_dir, frame_index: int, subsample_first: bool):
    """External input filter (off by default; used for the offline occupancy-filtering control): drop the returns an
    external method rejected.  ``DIR/<frame_index:06d>.npy`` is a boolean array over the subsample-first subset."""

    if not subsample_first:
        raise ValueError("input_keep_mask needs subsample_first (masks are indexed in the common-package point order)")
    path = Path(mask_dir) / f"{int(frame_index):06d}.npy"
    if not path.exists():
        raise FileNotFoundError(f"input_keep_mask: no mask for frame {frame_index} ({path})")
    keep = np.load(path)
    if keep.dtype != np.bool_ or keep.ndim != 1 or len(keep) != len(points_lidar):
        raise ValueError(f"input_keep_mask: {path} must be a 1-D bool array of length {len(points_lidar)}, got {keep.dtype} {keep.shape}")
    if keep.all():
        return frame, points_lidar
    sl = lambda x: None if x is None else np.asarray(x)[keep]   # noqa: E731
    frame = dataclasses.replace(frame, points_lidar=np.asarray(frame.points_lidar)[keep], intensity=sl(getattr(frame, "intensity", None)),
                                point_time_offset_ns=sl(getattr(frame, "point_time_offset_ns", None)))
    return frame, np.asarray(points_lidar)[keep]
