#!/usr/bin/env python3
"""Export an Oxford Spires sequence (Hesai QT64, 10 Hz, per-point timestamps) into the gubmap-canonical-v1 layout, mirroring
dataset/converter/export_newer_college.py.

Inputs (downloaded minimal set): <raw>/lidar-clouds/<stamp>.pcd (binary PCD, fields x y z _ intensity _ timestamp ring _,
timestamp = absolute seconds per point), <raw>/gt-tum.txt (TUM, 20 Hz, base frame in the TLS world frame; official
statement), calibration README: T_base_lidar t = (0, 0, 0.124) m, q_xyzw = (0, 0, 1, 0).
Canonical mapping: trajectory.txt = gt-tum rows verbatim (canonical 'imu' frame = Spires base frame), calibration.json
imu_from_lidar = T_base_lidar, lidar/frame_NNNNNN.pcd = x y z intensity offset_time (ns since the frame stamp, U4),
manifest point_time_field = offset_time.  Frames outside the GT trajectory span are dropped (recorded).

usage: python dataset/converter/export_oxford_spires.py --raw RAW_DIR --output OUT --name NAME [--start I --count N]
"""
from __future__ import annotations
import argparse, hashlib, json, shutil
from pathlib import Path
import numpy as np

T_BASE_LIDAR_T = (0.0, 0.0, 0.124); T_BASE_LIDAR_Q_XYZW = (0.0, 0.0, 1.0, 0.0)
RAW_DTYPE = np.dtype([("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("_a", "V4"), ("intensity", "<f4"), ("_b", "V4"), ("timestamp", "<f8"), ("ring", "<u2"), ("_c", "V14")])


def read_raw_pcd(p: Path):
    with open(p, "rb") as f:
        hdr = []
        while True:
            l = f.readline(); hdr.append(l.decode("ascii", "ignore").strip())
            if l.startswith(b"DATA"):
                break
        fields = [h for h in hdr if h.startswith("FIELDS")][0].split()[1:]
        if fields != ["x", "y", "z", "_", "intensity", "_", "timestamp", "ring", "_"]:
            raise ValueError(f"{p}: unexpected FIELDS {fields}")
        n = int([h for h in hdr if h.startswith("POINTS")][0].split()[1])
        a = np.frombuffer(f.read(n * RAW_DTYPE.itemsize), dtype=RAW_DTYPE)
    return a


def drop_duplicate_returns(a_, max_dt_s: float):
    """Drop the later copy of every (x, y, z, ring, intensity)-identical pair whose stamps are within max_dt_s (dual-return
    duplicate of one physical return).  Returns (kept records in file order, stats)."""
    key = np.stack([a_["x"], a_["y"], a_["z"], a_["intensity"], a_["ring"].astype("<f4")], 1); ts = a_["timestamp"]
    order = np.lexsort((ts, key[:, 4], key[:, 3], key[:, 2], key[:, 1], key[:, 0])); ks = key[order]; tso = ts[order]
    same_key = np.all(ks[1:] == ks[:-1], 1); dt = tso[1:] - tso[:-1]; match = same_key & (np.abs(dt) <= max_dt_s)
    drop = np.zeros(len(a_), bool); drop[order[1:][match]] = True; partner = np.zeros(len(a_), bool); partner[order[:-1][match]] = True
    _, inv, cnt = np.unique(key[:, :3], axis=0, return_inverse=True, return_counts=True); dup_xyz = cnt[inv.reshape(-1)] > 1
    st = {"dropped": int(drop.sum()), "dup_xyz_points": int(dup_xyz.sum()), "dup_xyz_kept_not_matching": int((dup_xyz & ~drop & ~partner).sum()),
          "dt_ms_min": float(np.abs(dt[match]).min() * 1e3) if match.any() else None, "dt_ms_max": float(np.abs(dt[match]).max() * 1e3) if match.any() else None}
    return a_[~drop], st


def write_pcd(path: Path, xyz, intensity, t_ns) -> int:
    n = len(xyz); rec = np.empty(n, dtype=np.dtype([("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("intensity", "<f4"), ("offset_time", "<u4")]))
    rec["x"], rec["y"], rec["z"] = xyz[:, 0], xyz[:, 1], xyz[:, 2]; rec["intensity"] = intensity; rec["offset_time"] = t_ns.astype(np.uint32)
    header = ("# .PCD v0.7 - Oxford Spires Hesai QT64 scan, offset_time = ns since the frame stamp\nVERSION 0.7\n"
              "FIELDS x y z intensity offset_time\nSIZE 4 4 4 4 4\nTYPE F F F F U\nCOUNT 1 1 1 1 1\n"
              f"WIDTH {n}\nHEIGHT 1\nVIEWPOINT 0 0 0 1 0 0 0\nPOINTS {n}\nDATA binary\n")
    with open(path, "wb") as f:
        f.write(header.encode("ascii")); f.write(rec.tobytes())
    return n


def quat_to_R(q):
    x, y, z, w = q; return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)], [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)], [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def sha256_files(paths):
    h = hashlib.sha256()
    for p in paths:
        with open(p, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--raw", type=Path, required=True); ap.add_argument("--output", type=Path, required=True); ap.add_argument("--name", required=True)
    ap.add_argument("--start", type=int, default=0, help="first raw frame index (sorted by stamp) to export"); ap.add_argument("--count", type=int, default=None, help="number of frames (default: all inside the GT span)")
    ap.add_argument("--clouds", type=Path, default=None, help="directory of raw <stamp>.pcd files (default: <raw>/lidar-clouds or <raw>/lidar-clouds_unzip/lidar-clouds)"); ap.add_argument("--drop-duplicate-returns", action="store_true", help="drop the second copy of a dual-return pair: identical (x, y, z, ring, intensity) and |dt| <= --duplicate-max-dt-ms (the Hesai QT64 dual-return packet carries the two returns of one firing as two blocks; when they coincide the same return is written twice, the driver stamps the second block one azimuth column = 0.1667 ms later; identical-xyz points that do NOT satisfy the full rule are kept). Keeps the earlier stamp / first in file order. Counts recorded in export_checks.json"); ap.add_argument("--duplicate-max-dt-ms", type=float, default=1.0)
    ap.add_argument("--drop-duplicate-xyz", action="store_true", help="drop exact-xyz duplicate returns (Hesai dual-return packets carry the same return twice with identical ring/intensity and timestamps 0.1667 ms apart; ~98 %% of Keble-02 returns are such pairs); keeps the first occurrence in file order"); ap.add_argument("--min-offset-ms", type=float, default=-1.0); ap.add_argument("--max-offset-ms", type=float, default=150.0)
    a = ap.parse_args(); out = a.output; (out / "lidar").mkdir(parents=True, exist_ok=True)
    clouds = a.clouds or next((d for d in (a.raw / "lidar-clouds", a.raw / "lidar-clouds_unzip" / "lidar-clouds") if d.is_dir()), None)
    if clouds is None:
        raise SystemExit(f"no lidar-clouds directory under {a.raw}")
    files = sorted(clouds.glob("*.pcd"), key=lambda p: float(p.stem))
    if not files:
        raise SystemExit(f"no PCD files in {clouds}")
    gt = np.loadtxt(a.raw / "gt-tum.txt")
    stamps = np.array([float(p.stem) for p in files]); inside = (stamps >= gt[0, 0]) & (stamps <= gt[-1, 0])
    sel = [i for i in range(len(files)) if inside[i]][a.start:(a.start + a.count) if a.count else None]
    manifest = []; checks = {"drop_duplicate_returns": bool(a.drop_duplicate_returns), "duplicate_max_dt_ms": a.duplicate_max_dt_ms, "drop_duplicate_xyz": bool(a.drop_duplicate_xyz), "duplicate_points_dropped": 0, "duplicate_xyz_points_total": 0, "duplicate_xyz_points_kept_not_matching_rule": 0, "dropped_pair_dt_ms_min": None, "dropped_pair_dt_ms_max": None, "raw_frames": len(files), "frames_outside_gt_span": int((~inside).sum()), "exported": len(sel), "offset_ms_min": None, "offset_ms_max": None, "points_min": None, "points_max": None, "dropped_points_offset": 0}
    omin, omax, pmin, pmax = 1e9, -1e9, 1e12, 0
    for k, i in enumerate(sel):
        p = files[i]; a_ = read_raw_pcd(p)
        if a.drop_duplicate_returns:
            a_, st = drop_duplicate_returns(a_, a.duplicate_max_dt_ms * 1e-3)
            checks["duplicate_points_dropped"] += st["dropped"]; checks["duplicate_xyz_points_total"] += st["dup_xyz_points"]; checks["duplicate_xyz_points_kept_not_matching_rule"] += st["dup_xyz_kept_not_matching"]
            if st["dropped"]:
                checks["dropped_pair_dt_ms_min"] = st["dt_ms_min"] if checks["dropped_pair_dt_ms_min"] is None else min(checks["dropped_pair_dt_ms_min"], st["dt_ms_min"]); checks["dropped_pair_dt_ms_max"] = st["dt_ms_max"] if checks["dropped_pair_dt_ms_max"] is None else max(checks["dropped_pair_dt_ms_max"], st["dt_ms_max"])
        elif a.drop_duplicate_xyz:
            _, first = np.unique(np.stack([a_["x"], a_["y"], a_["z"]], 1), axis=0, return_index=True); checks["duplicate_points_dropped"] += int(len(a_) - len(first)); a_ = a_[np.sort(first)]
        stamp_ns = int(round(float(p.stem) * 1e9)); off_ns = np.round((a_["timestamp"] - float(p.stem)) * 1e9).astype(np.int64)
        ok = (off_ns >= a.min_offset_ms * 1e6) & (off_ns <= a.max_offset_ms * 1e6) & np.isfinite(a_["x"]) & np.isfinite(a_["y"]) & np.isfinite(a_["z"]); checks["dropped_points_offset"] += int((~ok).sum())
        xyz = np.stack([a_["x"], a_["y"], a_["z"]], 1)[ok].astype(np.float32); off = np.clip(off_ns[ok], 0, None)
        fid = f"frame_{k:06d}"; n = write_pcd(out / "lidar" / f"{fid}.pcd", xyz, a_["intensity"][ok].astype(np.float32), off)
        omin = min(omin, off_ns.min() / 1e6); omax = max(omax, off_ns.max() / 1e6); pmin = min(pmin, n); pmax = max(pmax, n)
        manifest.append({"frame_id": fid, "lidar_path": f"lidar/{fid}.pcd", "lidar_timestamp_ns": stamp_ns, "image_path": None, "image_timestamp_ns": None, "point_time_field": "offset_time", "point_time_unit": "nanoseconds",
                         "metadata": {"source_file": p.name, "raw_index": int(i), "points": int(n), "ring_min": int(a_["ring"].min()), "ring_max": int(a_["ring"].max())}})
    checks.update({"offset_ms_min": omin, "offset_ms_max": omax, "points_min": pmin, "points_max": pmax})
    (out / "manifest.jsonl").write_text("".join(json.dumps(m) + "\n" for m in manifest))
    shutil.copyfile(a.raw / "gt-tum.txt", out / "trajectory.txt")
    R = quat_to_R(T_BASE_LIDAR_Q_XYZW)
    (out / "calibration.json").write_text(json.dumps({"imu_from_lidar": {"rotation": R.reshape(-1).tolist(), "translation": list(T_BASE_LIDAR_T)}, "camera": None,
                                                       "note": "imu frame = Oxford Spires base frame (calibration/README.md: T_base_lidar t=(0,0,0.124) q_xyzw=(0,0,1,0)); trajectory.txt = official gt-tum.txt (base frame in the TLS world frame)"}, indent=2))
    (out / "export_checks.json").write_text(json.dumps(checks, indent=2))
    meta_hash = sha256_files([out / p for p in ("manifest.jsonl", "trajectory.txt", "calibration.json")]); lidar_hash = sha256_files(sorted((out / "lidar").glob("*.pcd")))
    (out / "dataset.json").write_text(json.dumps({"format": "gubmap-canonical-v1", "sequence": a.name, "source": f"Oxford Spires {a.raw.name} (HuggingFace ori-drs/oxford_spires_dataset, raw/lidar-clouds.zip + processed/trajectory/gt-tum.txt)",
                                                  "sensor": "Hesai QT64 (64 x 600, 10 Hz), handheld", "pose_frame": "T_W_base: official gt-tum.txt (base frame in the TLS world frame)", "point_time": "offset_time = ns since the frame stamp (per-point absolute timestamps in the raw PCD)",
                                                  "exported_frames": len(sel), "duplicate_rule": ("drop_duplicate_returns: identical (x,y,z,ring,intensity) and |dt| <= %g ms, keep earlier" % a.duplicate_max_dt_ms) if a.drop_duplicate_returns else ("drop_duplicate_xyz (xyz only, superseded)" if a.drop_duplicate_xyz else "none"), "raw_index_range": [int(sel[0]), int(sel[-1])] if sel else None, "sha256_manifest_trajectory_calibration": meta_hash, "sha256_lidar_pcds": lidar_hash, "exporter": "export_oxford_spires.py"}, indent=2))
    (out / "README.md").write_text(f"# {a.name}\n\nOxford Spires {a.raw.name} in gubmap-canonical-v1 layout (dataset/converter/export_oxford_spires.py).\n- LiDAR: {len(sel)} scans (raw index {sel[0] if sel else None}..{sel[-1] if sel else None}), binary PCD `x y z intensity offset_time` (ns since the frame stamp; per-point timestamps from the raw PCD).\n- Poses: official `gt-tum.txt` (20 Hz, base frame in the TLS world frame) as T_W_base; imu_from_lidar = T_base_lidar from calibration/README.md.\n- No images exported.\n")
    print(f"[export] {len(sel)} frames -> {out} ({checks})")


if __name__ == "__main__":
    main()
