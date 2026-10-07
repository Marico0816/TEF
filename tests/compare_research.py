#!/usr/bin/env python3
"""Equivalence check against the research implementation (needs the research checkout, read only).

Runs the research mapper (``tef.pipeline.main`` with the frozen paper flags + the formal-run flags) and this repository's
``tef_mapping.py`` on the same frames, then compares
  * the output meshes: vertex / face / colour arrays (exact), else face counts and the vertex-to-vertex distances;
  * the per-block counts: new nodes, candidates, votes, new votes, solve nodes / iterations, moved nodes.

    python tests/compare_research.py --case t2 --dataset ~/Desktop/NewerCollege/ready/ncd_quad --frames train:0-40 --device cpu

``--deterministic`` turns on ``torch.use_deterministic_algorithms`` in both runs (GPU: reproducible scatter-adds).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

HERE = Path(__file__).resolve().parent
CLEAN = HERE.parent

# research flags  <->  clean config overrides (each case = the frozen method + these changes)
CASES = {
    "t2": ("t2", [], []),
    "p2": ("p2", [], []),
    "t2_default_frame": ("t2", ["--ablate-band-field"], ["local_support=false"]),
    "t2_no_lateral": ("t2", ["--learned-band-max-footprint-samples-per-ray", "1"], ["max_footprint_samples_per_ray=1"]),
    "t2_const_trunc": ("t2", ["--ablate-per-return-truncation"], ["per_return_truncation=false"]),
    "t2_sum": ("t2", ["--chunk-fusion", "sum"], ["block_weight=sum"]),
    "t2_beta0": ("t2", ["--regularization-pass-weight", "0"], ["pass_weight=0"]),
    "t2_constant": ("t2", ["--regularization-free-target", "constant", "--regularization-constant-target", "1.0"],
                    ["free_target=constant", "constant_target=1.0"]),
    "t2_frame_unit": ("t2", ["--evidence-count-unit", "frame"], ["count_unit=frame"]),
    "t2_ray_unit": ("t2", ["--evidence-count-unit", "ray"], ["count_unit=ray"]),
    "t2_wmin": ("t2", ["--original-minimum-node-weight", "0.2"], ["min_node_weight=0.2"]),
    # storage path: eviction at 20 m, small tiles, periodic extraction every 2 blocks
    "t2_evict_tiles": ("t2", ["--evict-distance-m", "20", "--extract-tile-nodes", "300000", "--remesh-every-chunks", "2"],
                       ["evict_distance_m=20", "extract_tile_nodes=300000", "extract_every_blocks=2"]),
    "p2_evict_tiles": ("p2", ["--evict-distance-m", "20", "--extract-tile-nodes", "300000", "--remesh-every-chunks", "2"],
                       ["evict_distance_m=20", "extract_tile_nodes=300000", "extract_every_blocks=2"]),
    # optional extension: online output ({out} = this case's output directory; deltas compared file by file)
    "t2_online": ("t2", ["--remesh-every-chunks", "1", "--incremental-remesh", "--region-candidates", "--background-remesh",
                         "--mesh-delta-dir", "{out}/research_deltas"], ["@online", "mesh_delta_dir={out}/clean_deltas"]),
    "p2_online": ("p2", ["--remesh-every-chunks", "1", "--incremental-remesh", "--region-candidates", "--background-remesh",
                         "--mesh-delta-dir", "{out}/research_deltas"], ["@online", "mesh_delta_dir={out}/clean_deltas"]),
    "t2_online_inline": ("t2", ["--remesh-every-chunks", "1", "--incremental-remesh", "--region-candidates",
                                "--mesh-delta-dir", "{out}/research_deltas"],
                         ["@online", "background_remesh=false", "mesh_delta_dir={out}/clean_deltas"]),
    "t2_online_evict": ("t2", ["--remesh-every-chunks", "1", "--incremental-remesh", "--region-candidates", "--background-remesh",
                               "--mesh-delta-dir", "{out}/research_deltas", "--evict-distance-m", "20", "--extract-tile-nodes", "300000"],
                        ["@online", "mesh_delta_dir={out}/clean_deltas", "evict_distance_m=20", "extract_tile_nodes=300000"]),
    "t2_periodic_deltas": ("t2", ["--remesh-every-chunks", "1", "--mesh-delta-dir", "{out}/research_deltas"],
                           ["extract_every_blocks=1", "mesh_delta_dir={out}/clean_deltas"]),
    "t2_region": ("t2", ["--region-candidates"], ["region_candidates=true"]),
}
OVERLAYS = {"@online": CLEAN / "config" / "extensions" / "online.yaml"}

RESEARCH_RUNNER = r'''
import sys, json
root, det = sys.argv[1], sys.argv[2] == "1"
argv = json.loads(sys.argv[3])
sys.path.insert(0, root + "/src")
import torch
if det:
    torch.use_deterministic_algorithms(True, warn_only=True)
from tef.configuration import method_flags
from tef.pipeline import main
method, rest = argv[0], argv[1:]
main(["unused-checkpoint.npz", *method_flags(method, formal=True), *rest])
'''

CLEAN_RUNNER = r'''
import sys, json
root, det = sys.argv[1], sys.argv[2] == "1"
argv = json.loads(sys.argv[3])
sys.path.insert(0, root)
import torch
if det:
    torch.use_deterministic_algorithms(True, warn_only=True)
import tef_mapping
tef_mapping.main(argv)
'''


def resolve_frames(spec):
    sys.path.insert(0, str(CLEAN))
    from utils.tools import resolve_frames as rf
    return rf(spec)


def run(cmd, log, env):
    t = time.time()
    with open(log, "w") as f:
        code = subprocess.call(cmd, stdout=f, stderr=subprocess.STDOUT, env=env)
    if code:
        raise SystemExit(f"run failed ({code}), see {log}")
    return time.time() - t


def load_ply(path):
    import open3d as o3d
    m = o3d.io.read_triangle_mesh(str(path))
    return np.asarray(m.vertices), np.asarray(m.triangles), np.asarray(m.vertex_colors)


def compare_meshes(a, b):
    va, fa, ca = load_ply(a)
    vb, fb, cb = load_ply(b)
    out = {"vertices": [len(va), len(vb)], "faces": [len(fa), len(fb)]}
    out["identical"] = bool(va.shape == vb.shape and fa.shape == fb.shape and np.array_equal(va, vb) and np.array_equal(fa, fb) and np.array_equal(ca, cb))
    out["files_identical"] = Path(a).read_bytes() == Path(b).read_bytes()
    if not out["identical"] and len(va) and len(vb):
        from scipy.spatial import cKDTree
        d_ab = cKDTree(vb).query(va)[0]
        d_ba = cKDTree(va).query(vb)[0]
        out["vertex_dist_m"] = {"a_to_b_max": float(d_ab.max()), "b_to_a_max": float(d_ba.max()),
                                "a_to_b_p99": float(np.percentile(d_ab, 99)), "frac_a_over_1mm": float((d_ab > 1e-3).mean()),
                                "frac_b_over_1mm": float((d_ba > 1e-3).mean())}
    return out


def compare_deltas(a_dir, b_dir):
    """Every delta file of the two runs: same names and identical arrays."""
    a = sorted(p.name for p in Path(a_dir).glob("delta_*.npz"))
    b = sorted(p.name for p in Path(b_dir).glob("delta_*.npz"))
    out = {"files": [len(a), len(b)], "same_names": a == b, "differences": []}
    for name in sorted(set(a) & set(b)):
        with np.load(Path(a_dir) / name) as za, np.load(Path(b_dir) / name) as zb:
            keys = sorted(set(za.files) | set(zb.files))
            bad = [k for k in keys if k not in za.files or k not in zb.files or not np.array_equal(za[k], zb[k])]
        if bad:
            out["differences"].append(f"{name}: {bad}")
    out["identical"] = bool(out["same_names"] and not out["differences"] and len(a) > 0)
    return out


def compare_blocks(research_json, clean_json):
    r = json.load(open(research_json))["chunks_detail"]
    c = json.load(open(clean_json))["blocks_detail"]
    pairs = (("new_nodes", "new_nodes"), ("field_nodes", "field_nodes"), ("candidates", "candidates"), ("voted", "voted"),
             ("newly_voted", "newly_voted"), ("solve_nodes", "solve_nodes"), ("solve_iters", "solve_iters"),
             ("changed_nodes", "changed_nodes"), ("evicted_nodes", "evicted_nodes"))
    diffs = []
    if len(r) != len(c):
        diffs.append(f"block count {len(r)} vs {len(c)}")
    for i, (ra, ca) in enumerate(zip(r, c)):
        for kr, kc in pairs:
            if ra.get(kr) != ca.get(kc):
                diffs.append(f"block {i} {kr}: research {ra.get(kr)} clean {ca.get(kc)}")
        if ra.get("solve_max_update") != ca.get("solve_max_update"):
            diffs.append(f"block {i} solve_max_update: research {ra.get('solve_max_update')!r} clean {ca.get('solve_max_update')!r}")
    return {"blocks": [len(r), len(c)], "differences": diffs}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", choices=sorted(CASES), required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--frames", required=True)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--research-root", default=str(Path.home() / "Desktop/TEF"))
    ap.add_argument("--out", default=str(CLEAN / "outputs" / "compare"))
    ap.add_argument("--deterministic", action="store_true")
    ap.add_argument("--threads", default="4")
    ap.add_argument("--skip-research", action="store_true", help="reuse an existing research run of this case")
    a = ap.parse_args()
    import os
    method, research_extra, clean_set = CASES[a.case]
    frames = resolve_frames(a.frames)
    out = Path(a.out) / f"{a.case}_{a.device}{'_det' if a.deterministic else ''}_{a.frames.replace(':', '_')}"
    out.mkdir(parents=True, exist_ok=True)
    research_extra = [x.replace("{out}", str(out)) for x in research_extra]
    overlays = [str(OVERLAYS[x]) for x in clean_set if x.startswith("@")]
    clean_set = [x.replace("{out}", str(out)) for x in clean_set if not x.startswith("@")]
    for d in ("research_deltas", "clean_deltas"):
        if (out / d).exists():
            for f in (out / d).glob("delta_*.npz"):
                f.unlink()                       # this case's own previous deltas (both sides are rewritten)
    env = dict(os.environ, OMP_NUM_THREADS=a.threads, PYTHONUNBUFFERED="1", PYTHONDONTWRITEBYTECODE="1",
               CUBLAS_WORKSPACE_CONFIG=":4096:8", PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True")
    env.pop("PYTHONPATH", None)
    det = "1" if a.deterministic else "0"
    r_argv = [method, "--device", a.device, "--dataset", str(Path(a.dataset).expanduser()), "--original-frame-indices",
              *[str(i) for i in frames], "--output", str(out / "research.ply"), "--timing-json", str(out / "research.json"), *research_extra]
    c_argv = [str(CLEAN / "config" / f"tef_{method}.yaml"), *overlays, "--dataset", str(Path(a.dataset).expanduser()), "--frames", a.frames,
              "--output", str(out / "clean.ply"), "--device", a.device, "--timing-json", str(out / "clean.json"),
              *(["--set", *clean_set] if clean_set else [])]
    times = {}
    if not (a.skip_research and (out / "research.ply").exists()):
        times["research_s"] = run([sys.executable, "-c", RESEARCH_RUNNER, a.research_root, det, json.dumps(r_argv)], out / "research.log", env)
    times["clean_s"] = run([sys.executable, "-c", CLEAN_RUNNER, str(CLEAN), det, json.dumps(c_argv)], out / "clean.log", env)
    report = {"case": a.case, "device": a.device, "deterministic": a.deterministic, "frames": a.frames, **times,
              "mesh": compare_meshes(out / "research.ply", out / "clean.ply"),
              "per_block": compare_blocks(out / "research.json", out / "clean.json")}
    if (out / "research_deltas").exists() or (out / "clean_deltas").exists():
        report["deltas"] = compare_deltas(out / "research_deltas", out / "clean_deltas")
    (out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    m, b = report["mesh"], report["per_block"]
    deltas_ok = report.get("deltas", {"identical": True})["identical"]
    verdict = "IDENTICAL" if m["identical"] and not b["differences"] and deltas_ok else "DIFFERENT"
    print(f"{a.case} [{a.device}{' det' if a.deterministic else ''}, {a.frames}]: {verdict}; faces {m['faces']}; "
          f"files identical {m['files_identical']}; block diffs {len(b['differences'])}"
          + (f"; {m.get('vertex_dist_m')}" if not m["identical"] else "")
          + (f"; deltas {report['deltas']['files']} identical {report['deltas']['identical']}" if "deltas" in report else "") + f"; {times}")
    for d in b["differences"][:8]:
        print("   ", d)
    return 0 if verdict == "IDENTICAL" else 1


if __name__ == "__main__":
    raise SystemExit(main())
