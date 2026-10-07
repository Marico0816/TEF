"""Evaluate Mesh completeness and LiDAR free-space violations by ray casting.

This diagnostic deliberately uses the canonical dataset loader and never
changes input formats.  A Mesh hit substantially before a measured LiDAR
endpoint lies in observed free space and is therefore a useful proxy for
coarse-scale bridges or ghost surfaces.  A missing hit or a hit behind the
endpoint is a complementary hole/completeness proxy.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from dataset import DatasetLoader
from utils.sampler import deterministic_subsample_indices


RANGE_BINS = (
    ("0-5m", 0.0, 5.0),
    ("5-15m", 5.0, 15.0),
    ("15-30m", 15.0, 30.0),
    ("30-infm", 30.0, np.inf),
)


def _open3d() -> Any:
    try:
        import open3d as o3d
    except ImportError as error:
        raise RuntimeError("ray evaluation requires Open3D") from error
    return o3d


def _checkpoint_metadata(
    path: Path,
) -> tuple[np.ndarray, dict[int, np.ndarray]]:
    with np.load(path, allow_pickle=False) as checkpoint:
        trained = (
            np.asarray(checkpoint["support_frame_indices"], dtype=np.int64).reshape(-1)
            if "support_frame_indices" in checkpoint.files
            else np.empty(0, dtype=np.int64)
        )
        poses = (
            np.asarray(checkpoint["support_world_from_lidar"], dtype=np.float64)
            if "support_world_from_lidar" in checkpoint.files
            else None
        )
    corrected: dict[int, np.ndarray] = {}
    if poses is not None:
        if poses.shape != (len(trained), 4, 4):
            raise ValueError("checkpoint corrected LiDAR poses have invalid shape")
        corrected = {
            int(index): pose for index, pose in zip(trained, poses, strict=True)
        }
    return trained, corrected


def _metrics(
    observed_range: np.ndarray,
    hit_range: np.ndarray,
    endpoint_distance: np.ndarray,
    tolerances_m: tuple[float, ...],
) -> dict[str, object]:
    observed = np.asarray(observed_range, dtype=np.float64)
    hit = np.asarray(hit_range, dtype=np.float64)
    distance = np.asarray(endpoint_distance, dtype=np.float64)
    finite = np.isfinite(hit)
    intrusion = observed - hit
    rows: dict[str, object] = {
        "rays": int(len(observed)),
        "finite_hit_fraction": float(np.mean(finite)) if len(observed) else 0.0,
        "endpoint_distance_median_m": (
            float(np.median(distance)) if len(distance) else None
        ),
        "endpoint_distance_p90_m": (
            float(np.quantile(distance, 0.90)) if len(distance) else None
        ),
    }
    positive_intrusion = intrusion[finite & (intrusion > 0.0)]
    rows["positive_intrusion_p50_p90_m"] = (
        np.quantile(positive_intrusion, [0.5, 0.9]).tolist()
        if len(positive_intrusion)
        else [None, None]
    )
    tolerance_rows: dict[str, dict[str, float]] = {}
    for tolerance in tolerances_m:
        violation = finite & (hit < observed - tolerance)
        agreement = finite & (np.abs(hit - observed) <= tolerance)
        missing = (~finite) | (hit > observed + tolerance)
        tolerance_rows[f"{int(round(100 * tolerance))}cm"] = {
            "free_space_violation_fraction": float(np.mean(violation)),
            "surface_agreement_fraction": float(np.mean(agreement)),
            "missing_ray_fraction": float(np.mean(missing)),
        }
    rows["tolerances"] = tolerance_rows
    return rows


def evaluate(
    mesh_path: Path,
    checkpoint_path: Path,
    dataset_path: Path,
    *,
    selection: str,
    max_rays_per_frame: int,
    tolerances_m: tuple[float, ...],
    start: int,
    stop: int | None,
    frame_step: int,
    frame_indices: Sequence[int] | None = None,
    use_checkpoint_poses: bool = True,
    deskew_heldout: bool = False,
    mesh_components: bool = True,
) -> dict[str, object]:
    o3d = _open3d()
    mesh = o3d.io.read_triangle_mesh(str(mesh_path))
    if len(mesh.vertices) == 0 or len(mesh.triangles) == 0:
        raise ValueError("mesh is empty")
    if not mesh_components:
        # very large meshes: the ray metrics need only positions and triangles; colours/normals are dropped to save host memory
        mesh.vertex_colors = o3d.utility.Vector3dVector(); mesh.vertex_normals = o3d.utility.Vector3dVector(); mesh.triangle_normals = o3d.utility.Vector3dVector()
    tensor_mesh = o3d.t.geometry.TriangleMesh.from_legacy(mesh)
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(tensor_mesh)
    del tensor_mesh
    dataset = DatasetLoader(dataset_path, min_range_m=0.5, max_range_m=50.0)
    trained, corrected = _checkpoint_metadata(checkpoint_path)
    if not use_checkpoint_poses:
        corrected = {}
    trained_set = set(int(index) for index in trained)
    if frame_indices is not None:
        selected_indices = [int(index) for index in frame_indices]
        if (
            not selected_indices
            or len(set(selected_indices)) != len(selected_indices)
            or min(selected_indices) < 0
            or max(selected_indices) >= len(dataset)
        ):
            raise ValueError("frame_indices must be unique valid dataset indices")
    else:
        end = len(dataset) if stop is None else min(int(stop), len(dataset))
        candidates = list(range(max(0, int(start)), end, int(frame_step)))
        if selection == "heldout":
            selected_indices = [
                index for index in candidates if index not in trained_set
            ]
        elif selection == "trained":
            selected_indices = [
                index for index in candidates if index in trained_set
            ]
        else:
            selected_indices = candidates
    if not selected_indices:
        raise ValueError(f"selection '{selection}' produced no evaluation frames")

    observed_chunks: list[np.ndarray] = []
    hit_chunks: list[np.ndarray] = []
    distance_chunks: list[np.ndarray] = []
    frame_rows: list[dict[str, object]] = []
    deskew_fn = None
    if deskew_heldout:
        from dataset.trajectory import deskew_points_to_scan_frame, trajectory_pose_fn
        deskew_fn = (deskew_points_to_scan_frame, trajectory_pose_fn(dataset.trajectory), np.asarray(dataset.calibration.imu_from_lidar, dtype=np.float64))
    for index in selected_indices:
        frame = dataset[index]
        points = np.asarray(frame.points_lidar, dtype=np.float64)
        chosen = deterministic_subsample_indices(len(points), max_rays_per_frame)
        points = points[chosen]
        transform = corrected.get(index, frame.world_from_lidar)
        R = np.asarray(transform[:3, :3], dtype=np.float64); origin = np.asarray(transform[:3, 3], dtype=np.float64)
        if deskew_fn is not None and frame.point_time_offset_ns is not None:
            # each held-out return is placed with the sensor pose at its own time (relative motion within the scan from the
            # trajectory, applied on top of the frame's reference pose) and its ray starts where the sensor was at that time
            fn, pose_fn, ifl = deskew_fn
            pts_s, _, org_s = fn(points, np.asarray(frame.point_time_offset_ns)[chosen], int(frame.lidar_timestamp_ns), pose_fn, ifl, return_origins=True)
            rel = pts_s - org_s
            observed = np.linalg.norm(rel, axis=1)
            directions = (rel / np.maximum(observed[:, None], 1e-12)) @ R.T
            origins = org_s @ R.T + origin
        else:
            observed = np.linalg.norm(points, axis=1)
            directions_lidar = points / np.maximum(observed[:, None], 1e-12)
            directions = directions_lidar @ R.T
            origins = np.broadcast_to(origin, directions.shape)
        directions /= np.maximum(np.linalg.norm(directions, axis=1, keepdims=True), 1e-12)
        rays = np.concatenate((origins, directions), axis=1).astype(np.float32)
        cast = scene.cast_rays(o3d.core.Tensor(rays))
        hit = np.asarray(cast["t_hit"].numpy(), dtype=np.float64)
        endpoints = origins + observed[:, None] * directions
        endpoint_distance = scene.compute_distance(
            o3d.core.Tensor(endpoints.astype(np.float32))
        ).numpy().astype(np.float64)
        observed_chunks.append(observed)
        hit_chunks.append(hit)
        distance_chunks.append(endpoint_distance)
        frame_rows.append(
            {
                "dataset_index": int(index),
                **_metrics(observed, hit, endpoint_distance, tolerances_m),
            }
        )

    observed_all = np.concatenate(observed_chunks)
    hit_all = np.concatenate(hit_chunks)
    distance_all = np.concatenate(distance_chunks)
    by_range: dict[str, object] = {}
    for label, minimum, maximum in RANGE_BINS:
        keep = (observed_all >= minimum) & (observed_all < maximum)
        if np.any(keep):
            by_range[label] = _metrics(
                observed_all[keep],
                hit_all[keep],
                distance_all[keep],
                tolerances_m,
            )
    if mesh_components:
        labels, counts, areas = mesh.cluster_connected_triangles()
        component_counts = np.asarray(counts, dtype=np.int64)
        component_areas = np.asarray(areas, dtype=np.float64)
    else:
        # connected-component statistics are descriptive only (not a ray metric); on tens of millions of triangles the
        # adjacency they need does not fit in host memory, so they are reported as absent (-1 / NaN)
        component_counts = np.zeros(0, dtype=np.int64)
        component_areas = np.zeros(0, dtype=np.float64)
    return {
        "mesh": str(mesh_path.resolve()),
        "checkpoint": str(checkpoint_path.resolve()),
        "dataset": str(dataset_path.resolve()),
        "selection": selection,
        "frame_indices": selected_indices,
        "checkpoint_poses_used": bool(use_checkpoint_poses),
        "max_rays_per_frame": int(max_rays_per_frame),
        "overall": _metrics(
            observed_all, hit_all, distance_all, tolerances_m
        ),
        "by_range": by_range,
        "frames": frame_rows,
        "mesh_summary": {
            "vertices": int(len(mesh.vertices)),
            "triangles": int(len(mesh.triangles)),
            "surface_area_m2": float(mesh.get_surface_area()),
            "connected_components": int(len(component_counts)) if mesh_components else -1,
            "largest_component_triangle_fraction": (
                float(np.max(component_counts) / np.sum(component_counts))
                if len(component_counts)
                else 0.0
            ),
            "small_component_fraction_lt100_triangles": (
                float(np.mean(component_counts < 100))
                if len(component_counts)
                else 0.0
            ),
            "largest_component_area_fraction": (
                float(np.max(component_areas) / np.sum(component_areas))
                if len(component_areas) and np.sum(component_areas) > 0.0
                else 0.0
            ),
        },
    }


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Measure LiDAR free-space violations and Mesh completeness"
    )
    parser.add_argument("mesh", type=Path)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--selection", choices=("heldout", "trained", "all"), default="heldout")
    parser.add_argument("--max-rays-per-frame", type=int, default=10_000)
    parser.add_argument("--tolerances-m", type=float, nargs="+", default=(0.02, 0.05, 0.10))
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--stop", type=int, default=None)
    parser.add_argument("--frame-step", type=int, default=1)
    parser.add_argument("--frame-indices", type=int, nargs="+", default=None)
    parser.add_argument(
        "--no-checkpoint-poses",
        action="store_true",
        help="Use canonical dataset/FAST-LIVO2 poses for an explicit split",
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    arguments = _arguments()
    if arguments.max_rays_per_frame <= 0:
        raise ValueError("--max-rays-per-frame must be positive")
    tolerances = tuple(float(value) for value in arguments.tolerances_m)
    if not tolerances or any(not np.isfinite(value) or value <= 0.0 for value in tolerances):
        raise ValueError("--tolerances-m must contain finite positive values")
    result = evaluate(
        arguments.mesh,
        arguments.checkpoint,
        arguments.dataset,
        selection=arguments.selection,
        max_rays_per_frame=arguments.max_rays_per_frame,
        tolerances_m=tolerances,
        start=arguments.start,
        stop=arguments.stop,
        frame_step=arguments.frame_step,
        frame_indices=arguments.frame_indices,
        use_checkpoint_poses=not arguments.no_checkpoint_poses,
    )
    output = arguments.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
