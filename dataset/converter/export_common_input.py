#!/usr/bin/env python3
"""Export a canonical GBMap sequence in KITTI layout for PIN-SLAM / PINGS.

Writes, for the selected frames:

    OUT/velodyne/000000.bin ...   float32 x y z intensity, LiDAR frame at the
                                  scan timestamp, *already motion-compensated*
                                  with the same 2 ms deskew bins GUBMap uses
    OUT/poses.txt                 KITTI 12-value rows, T_world_lidar (canonical
                                  FAST-LIVO2 T_WI composed with LiDAR extrinsics)
    OUT/poses_tum.txt             the same poses in TUM format
    OUT/calib.txt                 Tr = identity (poses are expressed directly in
                                  the LiDAR frame, so no camera->LiDAR change)
    OUT/timestamps.txt            LiDAR timestamps in seconds
    OUT/frames.json               canonical dataset indices of every exported scan

Deskewed points are mapped back into the nominal scan frame so that a mapper
that applies ``poses.txt`` to each scan reproduces exactly the world-frame
geometry GUBMap integrates.  Use ``--exclude-mod 5 10`` to keep the 5-mod-10
held-out split out of the export.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from dataset import DatasetLoader  # noqa: E402
from dataset.calibration import invert_transform  # noqa: E402

DESKEW_BIN_NS = 2_000_000


def _transform(matrix: np.ndarray, points: np.ndarray) -> np.ndarray:
    return points @ matrix[:3, :3].T + matrix[:3, 3]


def _quat_from_matrix(rotation: np.ndarray) -> np.ndarray:
    from scipy.spatial.transform import Rotation

    return Rotation.from_matrix(rotation).as_quat()  # x y z w


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--frame-step", type=int, default=1)
    parser.add_argument("--frames", type=int, default=None)
    parser.add_argument("--start", type=int, default=0, help="first dataset index to consider (default 0 = unchanged behaviour); combined with --frames N this exports the N selected frames from --start on")
    parser.add_argument("--exclude-mod", type=int, nargs=2, default=None, metavar=("REMAINDER", "MODULUS"))
    parser.add_argument("--no-deskew", action="store_true")
    parser.add_argument("--max-points-per-frame", type=int, default=None,
                        help="keep at most N returns per frame with gubmap's deterministic_subsample_indices (the mapper's --original-max-rays-per-frame "
                             "rule, applied after range filtering and deskew in the loader's point order) and write OUT/point_indices/NNNNNN.npy with the "
                             "kept indices into the loader's frame array; default None = all returns (unchanged behaviour)")
    parser.add_argument("--min-range-m", type=float, default=0.5)
    parser.add_argument("--max-range-m", type=float, default=50.0)
    args = parser.parse_args()

    dataset = DatasetLoader(args.dataset, load_images=False, min_range_m=args.min_range_m, max_range_m=args.max_range_m)
    indices = list(range(args.start, len(dataset), args.frame_step))
    if args.exclude_mod is not None:
        r, m = args.exclude_mod
        indices = [i for i in indices if i % m != r]
    if args.frames is not None:
        indices = indices[: args.frames]

    out = args.output
    (out / "velodyne").mkdir(parents=True, exist_ok=True)
    kitti_rows, tum_rows, stamps = [], [], []
    for n, index in enumerate(indices):
        frame = dataset[index]
        points = np.asarray(frame.points_lidar, dtype=np.float64)
        world_from_lidar = frame.world_from_lidar
        offsets = frame.point_time_offset_ns
        if not args.no_deskew and offsets is not None and len(offsets) == len(points):
            offsets = np.asarray(offsets, dtype=np.int64)
            bins = np.floor_divide(offsets, DESKEW_BIN_NS)
            unique_bins, inverse = np.unique(bins, return_inverse=True)
            world = np.empty_like(points)
            for g, _ in enumerate(unique_bins):
                sel = np.flatnonzero(inverse == g)
                pose = dataset.trajectory.pose_at(frame.lidar_timestamp_ns + int(np.median(offsets[sel])))
                world[sel] = _transform(dataset.calibration.world_from_lidar(pose.world_from_imu), points[sel])
            points = _transform(invert_transform(world_from_lidar), world)
        intensity = frame.intensity if frame.intensity is not None else np.zeros(len(points))
        if args.max_points_per_frame is not None:
            from utils.sampler import deterministic_subsample_indices
            keep = deterministic_subsample_indices(len(points), args.max_points_per_frame)
            (out / "point_indices").mkdir(parents=True, exist_ok=True); np.save(out / "point_indices" / f"{n:06d}.npy", keep.astype(np.int64))
            points = points[keep]; intensity = np.asarray(intensity)[keep]
        scan = np.concatenate([points.astype(np.float32), np.asarray(intensity, dtype=np.float32).reshape(-1, 1)], axis=1)
        scan.astype(np.float32).tofile(out / "velodyne" / f"{n:06d}.bin")
        kitti_rows.append(" ".join(f"{v:.9e}" for v in world_from_lidar[:3, :].reshape(-1)))
        q = _quat_from_matrix(world_from_lidar[:3, :3])
        t = world_from_lidar[:3, 3]
        stamp = frame.lidar_timestamp_ns * 1e-9
        stamps.append(stamp)
        tum_rows.append(f"{stamp:.9f} {t[0]:.6f} {t[1]:.6f} {t[2]:.6f} {q[0]:.6f} {q[1]:.6f} {q[2]:.6f} {q[3]:.6f}")
    (out / "poses.txt").write_text("\n".join(kitti_rows) + "\n")
    (out / "poses_tum.txt").write_text("\n".join(tum_rows) + "\n")
    (out / "timestamps.txt").write_text("\n".join(f"{s:.9f}" for s in stamps) + "\n")
    identity = " ".join(f"{v:.1f}" for v in np.eye(4)[:3, :].reshape(-1))
    (out / "calib.txt").write_text(
        "\n".join([f"P0: {identity}", f"P1: {identity}", f"P2: {identity}", f"P3: {identity}", f"Tr: {identity}"]) + "\n"
    )
    (out / "frames.json").write_text(json.dumps({
        "dataset": str(args.dataset.resolve()),
        "dataset_frame_indices": indices,
        "deskew": "off" if args.no_deskew else "2 ms bins, remapped into scan frame",
        "max_points_per_frame": args.max_points_per_frame,
        "pose_frame": "T_world_lidar from canonical FAST-LIVO2 T_WI",
        "range_m": [args.min_range_m, args.max_range_m],
    }, indent=2))
    print(f"exported {len(indices)} scans to {out}")


if __name__ == "__main__":
    main()
