#!/usr/bin/env python3
"""Sanity test without a dataset: a synthetic floor (z = -1.5 m) inside a cylindrical wall (r = 8 m), scanned by a sensor
moving 0.3 m per frame.  Writes a canonical sequence, maps it with T2 or P2 and checks the mesh geometry.

    python scripts/demo_tiny_scene.py --out outputs/demo --method t2 --device cpu
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tef_mapping import run_tef  # noqa: E402
from utils.config import Config  # noqa: E402


def write_pcd(path: Path, xyz, intensity, t_ns) -> int:
    n = len(xyz)
    rec = np.empty(n, dtype=np.dtype([("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("intensity", "<f4"), ("offset_time", "<u4")]))
    rec["x"], rec["y"], rec["z"] = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    rec["intensity"] = intensity
    rec["offset_time"] = t_ns
    header = (f"# .PCD v0.7\nVERSION 0.7\nFIELDS x y z intensity offset_time\nSIZE 4 4 4 4 4\nTYPE F F F F U\nCOUNT 1 1 1 1 1\n"
              f"WIDTH {n}\nHEIGHT 1\nVIEWPOINT 0 0 0 1 0 0 0\nPOINTS {n}\nDATA binary\n")
    path.write_bytes(header.encode() + rec.tobytes())
    return n


def scene(origin):
    """Returns of a 900 x 48 beam scan (360 deg x +-16 deg) from ``origin`` against the floor and the wall (0.5-40 m)."""
    az = np.radians(np.arange(900) / 900 * 360 - 180)
    el = np.radians(np.linspace(-16, 16, 48))
    A, E = np.meshgrid(az, el)
    d = np.stack([np.cos(E) * np.cos(A), np.cos(E) * np.sin(A), np.sin(E)], -1).reshape(-1, 3)
    t_floor = np.where(d[:, 2] < -1e-6, (-1.5 - origin[2]) / d[:, 2], np.inf)
    a = d[:, 0] ** 2 + d[:, 1] ** 2
    b = 2 * (d[:, 0] * origin[0] + d[:, 1] * origin[1])
    c = origin[0] ** 2 + origin[1] ** 2 - 8.0 ** 2
    disc = b * b - 4 * a * c
    t_wall = np.where(disc > 0, (-b + np.sqrt(np.maximum(disc, 0))) / (2 * a), np.inf)
    t = np.minimum(np.where((t_floor > 0.5) & (t_floor < 40), t_floor, np.inf), np.where((t_wall > 0.5) & (t_wall < 40), t_wall, np.inf))
    hit = np.isfinite(t)
    return (d[hit] * t[hit, None]).astype(np.float64)


def write_sequence(ds: Path, frames: int):
    (ds / "lidar").mkdir(parents=True, exist_ok=True)
    manifest, trajectory = [], []
    for k in range(frames):
        origin = np.array([0.3 * k, 0.0, 0.0])
        points = scene(origin)      # sensor frame = world frame translated by origin (identity rotation)
        stamp = int(k * 1e8)
        write_pcd(ds / "lidar" / f"frame_{k:06d}.pcd", points, np.ones(len(points), np.float32), np.zeros(len(points), np.uint32))
        manifest.append({"frame_id": f"frame_{k:06d}", "lidar_path": f"lidar/frame_{k:06d}.pcd", "lidar_timestamp_ns": stamp, "image_path": None,
                         "image_timestamp_ns": None, "point_time_field": "offset_time", "point_time_unit": "nanoseconds",
                         "metadata": {"points": int(len(points))}})
        trajectory.append(f"{stamp * 1e-9:.9f} {origin[0]:.6f} {origin[1]:.6f} {origin[2]:.6f} 0 0 0 1")
    (ds / "manifest.jsonl").write_text("".join(json.dumps(m) + "\n" for m in manifest))
    (ds / "trajectory.txt").write_text("\n".join(trajectory) + "\n")
    (ds / "calibration.json").write_text(json.dumps({"imu_from_lidar": {"rotation": [1, 0, 0, 0, 1, 0, 0, 0, 1], "translation": [0, 0, 0]}, "camera": None}))
    (ds / "dataset.json").write_text(json.dumps({"format": "gubmap-canonical-v1", "sequence": "tiny_scene", "source": "synthetic"}))


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, default=ROOT / "outputs" / "demo")
    ap.add_argument("--frames", type=int, default=12)
    ap.add_argument("--method", choices=("t2", "p2"), default="t2")
    ap.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    ap.add_argument("--max-floor-err-cm", type=float, default=5.0, help="pass if the median |z + 1.5| of floor vertices is below this")
    ap.add_argument("--max-wall-err-cm", type=float, default=5.0, help="pass if the median |r - 8| of wall vertices is below this")
    ap.add_argument("--min-vertices", type=int, default=1000)
    a = ap.parse_args()

    ds = a.out / "dataset"
    write_sequence(ds, a.frames)
    print(f"[1/2] synthetic sequence written to {ds} ({a.frames} frames)")
    cfg = Config.load(ROOT / "config" / f"tef_{a.method}.yaml", [f"device={a.device}"])
    out = a.out / "mesh.ply"
    print(f"[2/2] mapping with {a.method.upper()} on {a.device}")
    run_tef(cfg, ds, list(range(a.frames)), out)

    import open3d as o3d
    m = o3d.io.read_triangle_mesh(str(out))
    v = np.asarray(m.vertices)
    floor = v[np.abs(v[:, 2] + 1.5) < 0.3]
    wall = v[np.abs(np.linalg.norm(v[:, :2], axis=1) - 8.0) < 0.5]
    checks = [("vertices", len(v), f">= {a.min_vertices}", len(v) >= a.min_vertices)]
    if len(floor) >= 100:
        e = float(np.median(np.abs(floor[:, 2] + 1.5)) * 100)
        checks.append(("floor median |z+1.5| cm", round(e, 2), f"<= {a.max_floor_err_cm}", e <= a.max_floor_err_cm))
    else:
        checks.append(("floor vertices", len(floor), ">= 100", False))
    if len(wall) >= 100:
        e = float(np.median(np.abs(np.linalg.norm(wall[:, :2], axis=1) - 8.0)) * 100)
        checks.append(("wall median |r-8| cm", round(e, 2), f"<= {a.max_wall_err_cm}", e <= a.max_wall_err_cm))
    else:
        checks.append(("wall vertices", len(wall), ">= 100", False))
    for name, got, want, ok in checks:
        print(f"  {'pass' if ok else 'FAIL'}  {name} = {got}  (required {want})")
    ok_all = all(c[3] for c in checks)
    print("[demo passed]" if ok_all else "[demo FAILED]")
    return 0 if ok_all else 1


if __name__ == "__main__":
    raise SystemExit(main())
