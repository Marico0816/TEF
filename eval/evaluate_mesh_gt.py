#!/usr/bin/env python3
"""Surface-quality evaluation of a mesh against a survey-grade reference cloud (PREREG_newer_college_2026-09-13.md §6-7).

observed-space mask   built ONLY from the mapping frames' returns (<= --mask-range m, canonical poses) on a --mask-voxel grid,
                      dilated by --mask-dilate voxels; the same mask crops the reference cloud and the sampled mesh points.  The
                      mask never depends on a method's output; cache it (--mask-cache) so every method is scored with identical keys.
metrics               accuracy = mean distance sampled-mesh -> GT; completeness = mean distance GT -> sampled-mesh; Chamfer-L1 =
                      their mean; F-score at each --thresholds (harmonic mean of precision = frac(acc <= t) and recall =
                      frac(comp <= t)); outlier fractions beyond --outlier m in both directions.  No distance truncation.
sampling              uniform surface samples on the mesh, count = number of reference points inside the mask (both capped at
                      --max-points with a seeded subsample).
uncertainty           paired spatial-block bootstrap: the masked region is tiled into --block m x --block m columns; blocks are
                      resampled with replacement (--bootstrap draws, --seed); the same draws are used for every method scored with
                      the same seed and block size, so differences between methods are paired.
local ROI             --roi-frames I J K ... : additionally report every metric inside the union of --roi-radius m discs around the
                      canonical sensor positions of those frames (wrong-chunk experiments), intersected with the mask.

usage: evaluate_mesh_gt.py MESH.ply --gt REF.ply --dataset ROOT --frames "0-311:skip5mod10" --output OUT.json
       [--mask-cache MASK.npz] [--roi-frames 20 21 ... ] [--thresholds 0.05 0.10 0.20] [--block 10] [--bootstrap 2000]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))


def parse_frames(spec: str, n: int) -> list[int]:
    """"0-311:skip5mod10" -> 0..311 without 5 mod 10; "0-40" -> 0..39; "3,7,9" -> list."""
    if "," in spec and "-" not in spec.split(":")[0]:
        return [int(x) for x in spec.split(",") if x]
    rng, _, rule = spec.partition(":")
    a, b = rng.split("-")
    idx = list(range(int(a), min(int(b), n)))
    if rule == "skip5mod10":
        idx = [i for i in idx if i % 10 != 5]
    elif rule:
        raise ValueError(f"unknown frame rule {rule}")
    return idx


def load_cloud(path: Path, max_points: int | None, seed: int) -> np.ndarray:
    import open3d as o3d
    pc = o3d.io.read_point_cloud(str(path))
    pts = np.asarray(pc.points, dtype=np.float64)
    if len(pts) == 0:
        raise ValueError(f"empty point cloud {path}")
    if max_points is not None and len(pts) > max_points:
        rng = np.random.default_rng(seed)
        pts = pts[rng.choice(len(pts), max_points, replace=False)]
    return pts


def voxel_keys(points: np.ndarray, voxel: float) -> np.ndarray:
    c = np.floor(points / voxel).astype(np.int64)
    return (c[:, 0] + (1 << 20)) * (1 << 42) + (c[:, 1] + (1 << 20)) * (1 << 21) + (c[:, 2] + (1 << 20))


def build_mask(dataset_root: Path, frames: list[int], voxel: float, dilate: int, max_range: float, deskew: bool = True) -> np.ndarray:
    """Sorted int64 voxel keys occupied by the mapping frames' returns (canonical poses; with ``deskew`` every return is placed
    with the trajectory pose at its own time, exactly as the mapper's --deskew-points does; scans without per-point times are
    used as they are), dilated by `dilate` voxels in 26-neighbourhood rings."""
    from dataset import DatasetLoader
    from dataset.trajectory import deskew_points_to_scan_frame, trajectory_pose_fn
    ds = DatasetLoader(dataset_root, min_range_m=0.5, max_range_m=max_range, load_images=False)
    pose_fn = trajectory_pose_fn(ds.trajectory)
    keys = []
    for i in frames:
        f = ds[i]
        T = np.asarray(f.world_from_lidar, dtype=np.float64)
        pts = np.asarray(f.points_lidar, dtype=np.float64)
        if deskew and f.point_time_offset_ns is not None:
            pts, _ = deskew_points_to_scan_frame(pts, f.point_time_offset_ns, int(f.lidar_timestamp_ns), pose_fn, ds.calibration.imu_from_lidar)
        P = pts @ T[:3, :3].T + T[:3, 3]
        keys.append(np.unique(voxel_keys(P, voxel)))
    occ = np.unique(np.concatenate(keys))
    if dilate > 0:
        offs = np.array([(dx, dy, dz) for dx in (-1, 0, 1) for dy in (-1, 0, 1) for dz in (-1, 0, 1)], dtype=np.int64)
        offs_key = offs[:, 0] * (1 << 42) + offs[:, 1] * (1 << 21) + offs[:, 2]
        for _ in range(dilate):
            occ = np.unique((occ[:, None] + offs_key[None, :]).reshape(-1))
    return occ


def key_centres(keys: np.ndarray, voxel: float) -> np.ndarray:
    """Inverse of voxel_keys: voxel centres (N,3)."""
    x = keys // (1 << 42) - (1 << 20); r = keys % (1 << 42)
    y = r // (1 << 21) - (1 << 20); z = r % (1 << 21) - (1 << 20)
    return (np.stack([x, y, z], axis=1).astype(np.float64) + 0.5) * voxel


def region_halfspaces(xy: np.ndarray) -> np.ndarray:
    """Outward halfspaces [a b c] of the 2D convex hull of the reference cloud (a x + b y + c <= 0 inside)."""
    from scipy.spatial import ConvexHull
    return ConvexHull(np.asarray(xy, dtype=np.float64)).equations


def inside_region(xy: np.ndarray, eq: np.ndarray, erode: float) -> np.ndarray:
    return (np.asarray(xy, dtype=np.float64) @ eq[:, :2].T + eq[:, 2] <= -float(erode)).all(axis=1)


def in_mask(points: np.ndarray, mask_keys: np.ndarray, voxel: float) -> np.ndarray:
    k = voxel_keys(points, voxel)
    pos = np.searchsorted(mask_keys, k)
    pos = np.clip(pos, 0, len(mask_keys) - 1)
    return mask_keys[pos] == k


def nn_dist(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    from scipy.spatial import cKDTree
    return cKDTree(dst).query(src, k=1, workers=-1)[0]


def metrics(d_acc: np.ndarray, d_comp: np.ndarray, thresholds, outlier: float, w_acc=None, w_comp=None) -> dict:
    """Means / fractions, optionally with per-point weights (block-bootstrap multiplicities)."""
    wa = np.ones_like(d_acc) if w_acc is None else w_acc
    wc = np.ones_like(d_comp) if w_comp is None else w_comp
    sa, sc = wa.sum(), wc.sum()
    acc = float((d_acc * wa).sum() / sa); comp = float((d_comp * wc).sum() / sc)
    out = {"accuracy_m": acc, "completeness_m": comp, "chamfer_l1_m": 0.5 * (acc + comp),
           "outlier_frac_pred": float(((d_acc > outlier) * wa).sum() / sa), "outlier_frac_gt": float(((d_comp > outlier) * wc).sum() / sc)}
    for t in thresholds:
        p = float(((d_acc <= t) * wa).sum() / sa); r = float(((d_comp <= t) * wc).sum() / sc)
        out[f"precision_{int(round(t * 100))}cm"] = p; out[f"recall_{int(round(t * 100))}cm"] = r
        out[f"fscore_{int(round(t * 100))}cm"] = 2 * p * r / (p + r) if p + r > 0 else 0.0
    return out


def block_ids(points: np.ndarray, block: float) -> np.ndarray:
    c = np.floor(points[:, :2] / block).astype(np.int64)
    return (c[:, 0] + (1 << 20)) * (1 << 21) + (c[:, 1] + (1 << 20))


def bootstrap(d_acc, p_pred, d_comp, p_gt, thresholds, outlier, block, draws, seed) -> dict:
    """Paired spatial-block bootstrap; blocks are the union of blocks touched by predictions and GT."""
    ba, bc = block_ids(p_pred, block), block_ids(p_gt, block)
    blocks = np.unique(np.concatenate([ba, bc]))
    ia, ic = np.searchsorted(blocks, ba), np.searchsorted(blocks, bc)
    rng = np.random.default_rng(seed)
    samples = {}
    for _ in range(draws):
        mult = np.bincount(rng.integers(0, len(blocks), len(blocks)), minlength=len(blocks)).astype(np.float64)
        wa, wc = mult[ia], mult[ic]
        if wa.sum() == 0 or wc.sum() == 0:
            continue
        m = metrics(d_acc, d_comp, thresholds, outlier, wa, wc)
        for k, v in m.items():
            samples.setdefault(k, []).append(v)
    return {"blocks": int(len(blocks)), "draws": draws, "block_m": block, "seed": seed,
            "ci95": {k: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] for k, v in samples.items()},
            "std": {k: float(np.std(v, ddof=1)) for k, v in samples.items()}}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mesh", type=Path)
    ap.add_argument("--gt", type=Path, required=True, help="reference cloud (ply/pcd)")
    ap.add_argument("--dataset", type=Path, required=True, help="canonical dataset root (mask from its mapping frames)")
    ap.add_argument("--frames", type=str, required=True, help='mapping frames for the mask, e.g. "0-1300:skip5mod10"')
    ap.add_argument("--mask-voxel", type=float, default=0.3)
    ap.add_argument("--mask-dilate", type=int, default=1)
    ap.add_argument("--mask-range", type=float, default=50.0)
    ap.add_argument("--mask-cache", type=Path, default=None, help="npz holding the mask keys; created if absent, reused if present")
    ap.add_argument("--no-mask-deskew", action="store_true", help="build the observed-space mask from raw (undeskewed) returns; the pre-registered protocol uses deskewed returns")
    ap.add_argument("--region-erode", type=float, default=None,
                    help="official reference region: keep only mask voxels whose centre lies inside the 2D convex hull of the reference cloud eroded by this many metres (default: off)")
    ap.add_argument("--max-points", type=int, default=3_000_000)
    ap.add_argument("--thresholds", type=float, nargs="+", default=(0.05, 0.10, 0.20))
    ap.add_argument("--outlier", type=float, default=0.5)
    ap.add_argument("--block", type=float, default=10.0)
    ap.add_argument("--bootstrap", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--roi-frames", type=int, nargs="*", default=None, help="frames whose sensor positions define the local ROI")
    ap.add_argument("--roi-radius", type=float, default=20.0)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--label", type=str, default=None)
    a = ap.parse_args()
    import open3d as o3d
    from dataset import DatasetLoader
    t0 = time.time()
    n_frames = len(DatasetLoader(a.dataset, load_images=False))
    frames = parse_frames(a.frames, n_frames)
    # ---- reference cloud (loaded first: the optional region hull comes from it)
    gt_all = load_cloud(a.gt, None, a.seed)
    region_tag = -1.0 if a.region_erode is None else float(a.region_erode)
    # ---- mask (method-independent): observed-space voxels, optionally intersected with the eroded reference-region hull
    if a.mask_cache is not None and a.mask_cache.exists():
        z = np.load(a.mask_cache); mask = z["keys"]
        cached_region = float(z["region_erode"]) if "region_erode" in z.files else -1.0
        cached_deskew = bool(z["deskew"]) if "deskew" in z.files else False
        if float(z["voxel"]) != a.mask_voxel or int(z["dilate"]) != a.mask_dilate or list(z["frames"]) != frames or cached_region != region_tag or cached_deskew != (not a.no_mask_deskew):
            raise SystemExit("--mask-cache was built with different voxel / dilation / frames / region / deskew")
    else:
        mask = build_mask(a.dataset, frames, a.mask_voxel, a.mask_dilate, a.mask_range, deskew=not a.no_mask_deskew)
        if a.region_erode is not None:
            eq = region_halfspaces(gt_all[:, :2])
            mask = mask[inside_region(key_centres(mask, a.mask_voxel)[:, :2], eq, a.region_erode)]
            if len(mask) == 0:
                raise SystemExit("no observed-space voxel lies inside the eroded reference region")
        if a.mask_cache is not None:
            a.mask_cache.parent.mkdir(parents=True, exist_ok=True)
            np.savez(a.mask_cache, keys=mask, voxel=a.mask_voxel, dilate=a.mask_dilate, frames=np.asarray(frames), range=a.mask_range, region_erode=region_tag, deskew=not a.no_mask_deskew)
    mask_hash = hashlib.sha256(mask.tobytes()).hexdigest()[:16]
    gt = gt_all[in_mask(gt_all, mask, a.mask_voxel)]
    del gt_all
    if len(gt) == 0:
        raise SystemExit("no reference points inside the observed-space mask: check frames / poses / coordinate frame")
    rng = np.random.default_rng(a.seed)
    if len(gt) > a.max_points:
        gt = gt[rng.choice(len(gt), a.max_points, replace=False)]
    # ---- mesh samples inside the mask (count = reference points in the mask)
    mesh = o3d.io.read_triangle_mesh(str(a.mesh))
    if len(mesh.vertices) == 0 or len(mesh.triangles) == 0:
        raise SystemExit("mesh is empty")
    o3d.utility.random.seed(a.seed)
    n_target = len(gt)
    pred = np.asarray(mesh.sample_points_uniformly(number_of_points=int(n_target * 1.5) + 1000).points, dtype=np.float64)
    pred = pred[in_mask(pred, mask, a.mask_voxel)]
    if len(pred) > n_target:
        pred = pred[rng.choice(len(pred), n_target, replace=False)]
    if len(pred) == 0:
        raise SystemExit("no mesh samples inside the observed-space mask")
    d_acc = nn_dist(pred, gt); d_comp = nn_dist(gt, pred)
    result = {"mesh": str(a.mesh.resolve()), "gt": str(a.gt.resolve()), "dataset": str(a.dataset.resolve()), "label": a.label or a.mesh.stem,
              "protocol": {"frames": a.frames, "n_mapping_frames": len(frames), "mask_voxel_m": a.mask_voxel, "mask_dilate": a.mask_dilate, "mask_range_m": a.mask_range,
                           "region_erode_m": a.region_erode, "mask_deskew": not a.no_mask_deskew, "mask_voxels": int(len(mask)), "mask_sha256_16": mask_hash, "gt_points_in_mask": int(len(gt)), "mesh_samples_in_mask": int(len(pred)),
                           "max_points": a.max_points, "thresholds_m": list(a.thresholds), "outlier_m": a.outlier, "seed": a.seed, "no_distance_truncation": True},
              "overall": metrics(d_acc, d_comp, a.thresholds, a.outlier),
              "mesh_summary": {"vertices": int(len(mesh.vertices)), "triangles": int(len(mesh.triangles))}}
    if a.bootstrap > 0:
        result["bootstrap"] = bootstrap(d_acc, pred, d_comp, gt, a.thresholds, a.outlier, a.block, a.bootstrap, a.seed)
    if a.roi_frames:
        ds = DatasetLoader(a.dataset, load_images=False)
        centres = np.array([np.asarray(ds[i].world_from_lidar)[:3, 3] for i in a.roi_frames], dtype=np.float64)
        def near(P):
            d2 = ((P[:, None, :2] - centres[None, :, :2]) ** 2).sum(-1)
            return d2.min(axis=1) <= a.roi_radius ** 2
        ma, mc = near(pred), near(gt)
        result["roi"] = {"frames": list(a.roi_frames), "radius_m": a.roi_radius, "mesh_samples": int(ma.sum()), "gt_points": int(mc.sum()),
                         "metrics": metrics(d_acc[ma], d_comp[mc], a.thresholds, a.outlier) if ma.sum() and mc.sum() else None}
    result["seconds"] = round(time.time() - t0, 1)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    o = result["overall"]
    print(f"{result['label']}: acc={o['accuracy_m']*100:.2f}cm comp={o['completeness_m']*100:.2f}cm chamfer={o['chamfer_l1_m']*100:.2f}cm "
          + " ".join(f"F{int(round(t*100))}={o[f'fscore_{int(round(t*100))}cm']*100:.2f}" for t in a.thresholds)
          + f" outl(pred/gt)={o['outlier_frac_pred']*100:.2f}/{o['outlier_frac_gt']*100:.2f}% | gt {len(gt)} pred {len(pred)} mask {len(mask)} vox [{mask_hash}] {result['seconds']}s")


if __name__ == "__main__":
    main()
