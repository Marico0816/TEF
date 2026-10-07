#!/usr/bin/env python3
"""Paired uncertainty for the reference-cloud comparison of two or more EXISTING meshes (no mapping), plus a height-band attribution.

Reproduces evaluate_mesh_gt.py sample for sample (same mask cache, seed, sampling order; the script checks that its overall
metrics equal the stored .gt.json to 1e-9), then
  * paired spatial-block bootstrap of the DIFFERENCES between methods: one block set (10 m XY columns touched by the reference
    points or by any method's samples), one multiplicity draw per replicate applied to every method, 2000 draws, seed 0;
    computed on per-block sufficient statistics (count, sum of distances, counts under each threshold / over the outlier
    distance), which is exactly the per-point weighted bootstrap of evaluate_mesh_gt.py
  * attribution: accuracy / precision / outlier fraction of each method's samples and completeness / recall of the reference
    points, split by height above the local ground (reference cloud, 2 m cells, 2nd percentile) and by horizontal distance to
    the mapping trajectory
usage: paired_gt_bootstrap.py --meshes A.ply B.ply --labels a b --stored a.gt.json b.gt.json --gt REF.ply --dataset ROOT
       --frames "0-1300:skip5mod10" --mask-cache MASK.npz --output OUT.json [--draws 2000] [--block 10]
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("evaluate_mesh_gt", HERE / "evaluate_mesh_gt.py")
ev = importlib.util.module_from_spec(spec); spec.loader.exec_module(ev)
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

HEIGHT_BANDS = ((-1e9, 0.3, "ground <0.3 m"), (0.3, 2.2, "0.3-2.2 m (people / low objects)"), (2.2, 1e9, ">2.2 m (walls, roofs, trees)"))
RANGE_BANDS = ((0.0, 5.0, "<5 m"), (5.0, 15.0, "5-15 m"), (15.0, 1e9, ">15 m"))


def sample_like_stage1(mesh_path: Path, gt: np.ndarray, mask: np.ndarray, voxel: float, seed: int, rng: np.random.Generator) -> np.ndarray:
    import open3d as o3d
    mesh = o3d.io.read_triangle_mesh(str(mesh_path))
    o3d.utility.random.seed(seed)
    n_target = len(gt)
    pred = np.asarray(mesh.sample_points_uniformly(number_of_points=int(n_target * 1.5) + 1000).points, dtype=np.float64)
    pred = pred[ev.in_mask(pred, mask, voxel)]
    if len(pred) > n_target:
        pred = pred[rng.choice(len(pred), n_target, replace=False)]
    return pred


def block_stats(ids: np.ndarray, d: np.ndarray, nb: int, thresholds, outlier: float) -> np.ndarray:
    """(nb, 3 + len(thresholds)) per-block [count, sum d, count d > outlier, count d <= t ...]."""
    cols = [np.bincount(ids, minlength=nb), np.bincount(ids, weights=d, minlength=nb), np.bincount(ids, weights=(d > outlier).astype(float), minlength=nb)]
    cols += [np.bincount(ids, weights=(d <= t).astype(float), minlength=nb) for t in thresholds]
    return np.stack(cols, axis=1).astype(np.float64)


def metrics_from_blocks(A: np.ndarray, C: np.ndarray, w: np.ndarray, thresholds) -> dict:
    a = w @ A; c = w @ C
    out = {"accuracy_m": a[1] / a[0], "completeness_m": c[1] / c[0], "outlier_frac_pred": a[2] / a[0], "outlier_frac_gt": c[2] / c[0]}
    out["chamfer_l1_m"] = 0.5 * (out["accuracy_m"] + out["completeness_m"])
    for i, t in enumerate(thresholds):
        p, r = a[3 + i] / a[0], c[3 + i] / c[0]
        k = int(round(t * 100))
        out[f"precision_{k}cm"], out[f"recall_{k}cm"] = p, r
        out[f"fscore_{k}cm"] = 2 * p * r / (p + r) if p + r > 0 else 0.0
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--meshes", type=Path, nargs="+", required=True)
    ap.add_argument("--labels", nargs="+", required=True)
    ap.add_argument("--stored", type=Path, nargs="+", required=True, help="the stage-1 .gt.json of each mesh (reproduction check)")
    ap.add_argument("--gt", type=Path, required=True)
    ap.add_argument("--dataset", type=Path, required=True)
    ap.add_argument("--frames", required=True)
    ap.add_argument("--mask-cache", type=Path, required=True)
    ap.add_argument("--mask-voxel", type=float, default=0.3)
    ap.add_argument("--max-points", type=int, default=3_000_000)
    ap.add_argument("--thresholds", type=float, nargs="+", default=(0.05, 0.10, 0.20))
    ap.add_argument("--outlier", type=float, default=0.5)
    ap.add_argument("--block", type=float, default=10.0)
    ap.add_argument("--draws", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--output", type=Path, required=True)
    a = ap.parse_args()
    assert len(a.meshes) == len(a.labels) == len(a.stored)
    t0 = time.time()
    mask = np.load(a.mask_cache)["keys"]
    gt_all = ev.load_cloud(a.gt, None, a.seed)
    gt = gt_all[ev.in_mask(gt_all, mask, a.mask_voxel)]
    # ground height from ALL reference points in the mask (before the 3 M subsample): 2 m cells, 2nd percentile of z
    cell = 2.0
    gk = np.floor(gt[:, :2] / cell).astype(np.int64); gkey = gk[:, 0] * 1_000_003 + gk[:, 1]
    order = np.argsort(gkey); gs, zs = gkey[order], gt[order, 2]
    uk, start = np.unique(gs, return_index=True); ground = {}
    bounds = list(start) + [len(gs)]
    for i, k in enumerate(uk):
        ground[int(k)] = float(np.percentile(zs[bounds[i]:bounds[i + 1]], 2))
    zg_default = float(np.percentile(gt[:, 2], 2))
    def height(P):
        k = np.floor(P[:, :2] / cell).astype(np.int64); kk = k[:, 0] * 1_000_003 + k[:, 1]
        return P[:, 2] - np.array([ground.get(int(x), zg_default) for x in kk]) if len(kk) < 2_000_000 else P[:, 2] - np.vectorize(lambda x: ground.get(int(x), zg_default))(kk)
    del gt_all
    methods = {}
    for mesh_path, lab, stored in zip(a.meshes, a.labels, a.stored):
        rng = np.random.default_rng(a.seed)                     # same rng stream as evaluate_mesh_gt: gt subsample first, then mesh subsample
        g = gt[rng.choice(len(gt), a.max_points, replace=False)] if len(gt) > a.max_points else gt
        pred = sample_like_stage1(mesh_path, g, mask, a.mask_voxel, a.seed, rng)
        d_acc, d_comp = ev.nn_dist(pred, g), ev.nn_dist(g, pred)
        m = ev.metrics(d_acc, d_comp, a.thresholds, a.outlier)
        ref = json.load(open(stored))["overall"]
        dev_ = max(abs(m[k] - ref[k]) for k in ref)
        print(f"{lab}: reproduction of {stored.name}: max |diff| {dev_:.2e}  (acc {m['accuracy_m']*100:.2f} comp {m['completeness_m']*100:.2f} F10 {m['fscore_10cm']*100:.2f})")
        if dev_ > 1e-9:
            raise SystemExit(f"{lab}: samples do not reproduce the stage-1 evaluation (max diff {dev_:.2e})")
        methods[lab] = {"pred": pred, "gt": g, "d_acc": d_acc, "d_comp": d_comp, "overall": m}
    labels = list(methods)
    if any(not np.array_equal(methods[labels[0]]["gt"], methods[l]["gt"]) for l in labels[1:]):
        raise SystemExit("reference subsample differs between methods")
    G = methods[labels[0]]["gt"]
    # ---- one block set for all methods
    all_ids = [ev.block_ids(G, a.block)] + [ev.block_ids(methods[l]["pred"], a.block) for l in labels]
    blocks = np.unique(np.concatenate(all_ids)); nb = len(blocks)
    gid = np.searchsorted(blocks, all_ids[0])
    stats = {}
    for i, l in enumerate(labels):
        pid = np.searchsorted(blocks, all_ids[1 + i])
        stats[l] = (block_stats(pid, methods[l]["d_acc"], nb, a.thresholds, a.outlier), block_stats(gid, methods[l]["d_comp"], nb, a.thresholds, a.outlier))
    one = np.ones(nb)
    for l in labels:  # block aggregation reproduces the point metrics
        mm = metrics_from_blocks(*stats[l], one, a.thresholds)
        assert max(abs(mm[k] - methods[l]["overall"][k]) for k in mm) < 1e-9
    rng = np.random.default_rng(a.seed)
    draws = {l: [] for l in labels}
    for _ in range(a.draws):
        w = np.bincount(rng.integers(0, nb, nb), minlength=nb).astype(np.float64)
        for l in labels:
            draws[l].append(metrics_from_blocks(*stats[l], w, a.thresholds))
    keys = list(draws[labels[0]][0])
    per_method = {l: {k: [float(np.percentile([d[k] for d in draws[l]], 2.5)), float(np.percentile([d[k] for d in draws[l]], 97.5))] for k in keys} for l in labels}
    paired = {}
    for i, l1 in enumerate(labels):
        for l2 in labels[i + 1:]:
            diff = {k: np.array([d1[k] - d2[k] for d1, d2 in zip(draws[l1], draws[l2])]) for k in keys}
            point = {k: methods[l1]["overall"][k] - methods[l2]["overall"][k] for k in keys}
            paired[f"{l1} - {l2}"] = {k: {"point": float(point[k]), "ci95": [float(np.percentile(diff[k], 2.5)), float(np.percentile(diff[k], 97.5))],
                                          "frac_draws_positive": float((diff[k] > 0).mean())} for k in keys}
    # ---- attribution: height above ground and horizontal distance to the mapping trajectory
    from dataset import DatasetLoader
    ds = DatasetLoader(a.dataset, load_images=False)
    frames = ev.parse_frames(a.frames, len(ds))
    traj = np.array([np.asarray(ds[i].world_from_lidar)[:3, 3] for i in frames[::5]])
    from scipy.spatial import cKDTree
    ttree = cKDTree(traj[:, :2])
    attrib = {}
    hG = height(G); rG = ttree.query(G[:, :2], k=1, workers=-1)[0]
    for l in labels:
        P = methods[l]["pred"]; hP = height(P); rP = ttree.query(P[:, :2], k=1, workers=-1)[0]
        dA, dC = methods[l]["d_acc"], methods[l]["d_comp"]
        rows = {}
        for kind, bands, vP, vG in (("height", HEIGHT_BANDS, hP, hG), ("range", RANGE_BANDS, rP, rG)):
            for lo, hi, name in bands:
                sp, sg = (vP >= lo) & (vP < hi), (vG >= lo) & (vG < hi)
                rows[f"{kind}: {name}"] = {"pred_samples": int(sp.sum()), "gt_points": int(sg.sum()),
                                           "accuracy_cm": float(dA[sp].mean() * 100) if sp.any() else None, "precision_10cm": float((dA[sp] <= 0.10).mean()) if sp.any() else None,
                                           "pred_frac_gt20cm": float((dA[sp] > 0.20).mean()) if sp.any() else None,
                                           "completeness_cm": float(dC[sg].mean() * 100) if sg.any() else None, "recall_10cm": float((dC[sg] <= 0.10).mean()) if sg.any() else None}
        attrib[l] = rows
    out = {"labels": labels, "meshes": [str(p) for p in a.meshes], "gt": str(a.gt), "mask_cache": str(a.mask_cache), "mask_sha256_16": __import__("hashlib").sha256(mask.tobytes()).hexdigest()[:16],
           "blocks": nb, "block_m": a.block, "draws": a.draws, "seed": a.seed, "overall": {l: methods[l]["overall"] for l in labels},
           "ci95_per_method": per_method, "paired_differences": paired, "attribution": attrib, "ground_cell_m": cell, "seconds": round(time.time() - t0, 1)}
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(out, indent=1))
    for pair, d in paired.items():
        print(f"{pair}: " + ", ".join(f"{k} {d[k]['point']*100:+.2f} [{d[k]['ci95'][0]*100:+.2f}, {d[k]['ci95'][1]*100:+.2f}]" for k in ("accuracy_m", "completeness_m", "chamfer_l1_m", "fscore_5cm", "fscore_10cm", "fscore_20cm", "outlier_frac_pred", "outlier_frac_gt")))
    for l in labels:
        print(f"{l} attribution:")
        for k, r in attrib[l].items():
            print(f"   {k:40s} pred {r['pred_samples']:8d} acc {r['accuracy_cm'] or float('nan'):6.2f} cm P10 {(r['precision_10cm'] or 0)*100:5.1f}% >20cm {(r['pred_frac_gt20cm'] or 0)*100:5.2f}% | gt {r['gt_points']:8d} comp {r['completeness_cm'] or float('nan'):6.2f} cm R10 {(r['recall_10cm'] or 0)*100:5.1f}%")
    print(f"{nb} blocks, {a.draws} paired draws, {out['seconds']} s -> {a.output}")


if __name__ == "__main__":
    main()
