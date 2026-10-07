#!/usr/bin/env python3
"""Export a Newer College 2020 sequence (handheld Ouster OS1-64) into the gubmap-canonical-v1 layout.

Inputs (official download in RAW_DIR):
  <sequence>/rooster_*.bag                      topic /os1_cloud_node/points, 64x1024 organised scans, per-point ``t`` = ns since the
                                                scan header stamp (header stamp = first column), 10 Hz
  <sequence>/ground_truth/registered_poses.csv  10 Hz, one row per scan (stamps equal the scan header stamps); pose of the NCD 'base'
                                                frame (left RealSense position, robotic axes x-forward / y-left / z-up) in the BLK360
                                                prior-map frame  ->  T_WB
  04_calibration/kalibr_output/cam-ouster-imu/camchain-*.yaml   T_cam_imu (imu = Ouster IMU) from the official kalibr run
  04_calibration/kalibr_output/ouster_imu_lidar_transforms.yaml  os1_imu <-> os1_lidar (180 deg yaw, 28.5 mm)

base_from_lidar (nominal) = R_body_from_optical @ T_cam0_imu @ T_imu_lidar.  With --reference and --refine-extrinsic N the constant
residual of that chain is estimated by point-to-plane ICP of N deskewed scans against the prior map (body-frame corrections averaged,
one rigid transform for the whole sequence, hence identical for every mapping method); --validate-frames M records the PREREG §1.1 hard
checks on M disjoint frames (deskewed residual <= raw residual, median distance to the prior map) into <output>/export_checks.json.

Canonical mapping: trajectory.txt = T_WB rows of registered_poses.csv (the canonical "imu" frame is the NCD base frame),
calibration.json imu_from_lidar = base_from_lidar, lidar/*.pcd = x y z intensity offset_time (ns, U4), manifest point_time_field.

usage: python dataset/converter/export_newer_college.py --raw RAW_DIR --sequence 05_quad_with_dynamics --output OUT_DIR
       [--reference <prior map ply> --refine-extrinsic 40 --validate-frames 20 --exclude-frames-mod 10:5] [--count N] [--no-lidar]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import yaml
from scipy.spatial.transform import Rotation, Slerp

NS = 1_000_000_000
POINTS_TOPIC = "/os1_cloud_node/points"
SCAN_DTYPE = np.dtype({"names": ["x", "y", "z", "intensity", "t", "reflectivity", "ring", "noise", "range"],
                       "formats": ["<f4", "<f4", "<f4", "<f4", "<u4", "<u2", "u1", "<u2", "<u4"],
                       "offsets": [0, 4, 8, 16, 20, 24, 26, 28, 32], "itemsize": 48})
# optical (x right, y down, z forward) -> robotic body (x forward, y left, z up)
R_BODY_FROM_OPTICAL = np.array([[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]])


def T_of(R, t=(0.0, 0.0, 0.0)) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = np.asarray(R, dtype=np.float64)
    T[:3, 3] = np.asarray(t, dtype=np.float64)
    return T


def quat_xyzw_to_T(q, t) -> np.ndarray:
    return T_of(Rotation.from_quat(np.asarray(q, dtype=np.float64)).as_matrix(), t)


def rt(T: np.ndarray) -> dict:
    return {"rotation": [float(x) for x in T[:3, :3].reshape(-1)], "translation": [float(x) for x in T[:3, 3]]}


def ypr_deg(T: np.ndarray) -> list[float]:
    return [float(x) for x in Rotation.from_matrix(T[:3, :3]).as_euler("zyx", degrees=True)]


def nominal_base_from_lidar(calib_dir: Path) -> tuple[np.ndarray, dict]:
    """base_from_lidar = body_from_optical @ cam0_from_imu @ imu_from_lidar (all from the official calibration files)."""
    chain = sorted((calib_dir / "kalibr_output" / "cam-ouster-imu").glob("camchain-ouster_imu-cam*.yaml"))
    if not chain:
        raise FileNotFoundError(f"no camchain-ouster_imu-cam*.yaml under {calib_dir}")
    cc = yaml.safe_load(chain[0].read_text())
    T_cam0_imu = np.asarray(cc["cam0"]["T_cam_imu"], dtype=np.float64)
    tf = yaml.safe_load((calib_dir / "kalibr_output" / "ouster_imu_lidar_transforms.yaml").read_text())
    blk = tf["os1_imu_to_os1_lidar"]  # translation of the lidar origin in the imu frame, rotation imu<-lidar (180 deg yaw)
    T_imu_lidar = quat_xyzw_to_T(blk["rotation"], blk["translation"])
    T = T_of(R_BODY_FROM_OPTICAL) @ T_cam0_imu @ T_imu_lidar
    info = {"camchain": chain[0].name, "T_cam0_imu": T_cam0_imu.tolist(), "T_imu_lidar": T_imu_lidar.tolist(),
            "body_from_optical": R_BODY_FROM_OPTICAL.tolist(), "nominal_base_from_lidar": T.tolist(), "nominal_ypr_deg": ypr_deg(T)}
    return T, info


class GtInterp:
    """Linear translation + SLERP rotation between the 10 Hz registered poses (T_WB)."""

    def __init__(self, t_ns: np.ndarray, p: np.ndarray, q_xyzw: np.ndarray) -> None:
        self.t_ns = np.asarray(t_ns, dtype=np.int64)
        self.t = (self.t_ns - self.t_ns[0]).astype(np.float64) / NS
        self.p = np.asarray(p, dtype=np.float64)
        self.slerp = Slerp(self.t, Rotation.from_quat(q_xyzw))

    def poses(self, query_ns: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        tq = np.clip((np.asarray(query_ns, dtype=np.int64) - self.t_ns[0]).astype(np.float64) / NS, self.t[0], self.t[-1])
        R = self.slerp(tq).as_matrix()
        P = np.stack([np.interp(tq, self.t, self.p[:, j]) for j in range(3)], axis=1)
        return R, P

    def pose(self, query_ns: int) -> np.ndarray:
        R, P = self.poses(np.asarray([query_ns]))
        return T_of(R[0], P[0])


def load_gt(csv_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rows = np.loadtxt(csv_path, delimiter=",", comments="#")
    t_ns = rows[:, 0].astype(np.int64) * NS + rows[:, 1].astype(np.int64)
    if np.any(np.diff(t_ns) <= 0):
        raise ValueError("registered_poses.csv stamps must be strictly increasing")
    q = rows[:, 5:9]
    if np.abs(np.linalg.norm(q, axis=1) - 1.0).max() > 1e-6:
        raise ValueError("quaternions are not unit norm")
    return t_ns, rows[:, 2:5], q


def index_scans(bags: list[Path]):
    """(bag path, bag time ns, header stamp ns) for every PointCloud2 on the points topic, sorted by header stamp."""
    from rosbags.rosbag1 import Reader
    from rosbags.typesys import Stores, get_typestore
    ts = get_typestore(Stores.ROS1_NOETIC)
    rows = []
    for bag in bags:
        with Reader(bag) as r:
            conns = [c for c in r.connections if c.topic == POINTS_TOPIC]
            for conn, t, raw in r.messages(connections=conns):
                m = ts.deserialize_ros1(raw, conn.msgtype)
                rows.append((bag, int(t), int(m.header.stamp.sec) * NS + int(m.header.stamp.nanosec)))
    rows.sort(key=lambda r: r[2])
    return rows


def iter_scans(bags: list[Path], wanted: dict[int, int]):
    """Yield (header stamp ns, structured scan array) for the bag times in ``wanted`` (bag_time_ns -> header ns), in bag order."""
    from rosbags.rosbag1 import Reader
    from rosbags.typesys import Stores, get_typestore
    ts = get_typestore(Stores.ROS1_NOETIC)
    for bag in bags:
        with Reader(bag) as r:
            conns = [c for c in r.connections if c.topic == POINTS_TOPIC]
            for conn, t, raw in r.messages(connections=conns):
                if int(t) not in wanted:
                    continue
                m = ts.deserialize_ros1(raw, conn.msgtype)
                hdr = int(m.header.stamp.sec) * NS + int(m.header.stamp.nanosec)
                if hdr != wanted[int(t)]:
                    raise RuntimeError(f"header stamp mismatch in {bag.name} at bag time {t}")
                if m.point_step != SCAN_DTYPE.itemsize or [f.name for f in m.fields] != list(SCAN_DTYPE.names):
                    raise RuntimeError(f"unexpected PointCloud2 layout in {bag.name}: {[f.name for f in m.fields]} step {m.point_step}")
                yield hdr, np.frombuffer(m.data, dtype=SCAN_DTYPE)


def scan_arrays(a: np.ndarray):
    valid = a["range"] > 0
    xyz = np.stack([a["x"], a["y"], a["z"]], axis=1)[valid].astype(np.float64)
    return xyz, a["intensity"][valid].astype(np.float32), a["t"][valid].astype(np.int64), a["range"][valid].astype(np.float64) * 1e-3


def write_pcd(path: Path, xyz: np.ndarray, intensity: np.ndarray, t_ns: np.ndarray) -> int:
    n = len(xyz)
    rec = np.empty(n, dtype=np.dtype([("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("intensity", "<f4"), ("offset_time", "<u4")]))
    rec["x"], rec["y"], rec["z"] = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    rec["intensity"] = intensity
    rec["offset_time"] = t_ns.astype(np.uint32)
    header = ("# .PCD v0.7 - Newer College OS1-64 scan, offset_time = ns since the scan header stamp\nVERSION 0.7\n"
              "FIELDS x y z intensity offset_time\nSIZE 4 4 4 4 4\nTYPE F F F F U\nCOUNT 1 1 1 1 1\n"
              f"WIDTH {n}\nHEIGHT 1\nVIEWPOINT 0 0 0 1 0 0 0\nPOINTS {n}\nDATA binary\n").encode("ascii")
    with path.open("wb") as f:
        f.write(header)
        f.write(rec.tobytes())
    return n


def world_points(xyz: np.ndarray, t_rel_ns: np.ndarray, stamp_ns: int, gt: GtInterp, T_BL: np.ndarray, deskew: bool) -> np.ndarray:
    B = xyz @ T_BL[:3, :3].T + T_BL[:3, 3]
    if not deskew:
        T = gt.pose(stamp_ns)
        return B @ T[:3, :3].T + T[:3, 3]
    R, P = gt.poses(stamp_ns + t_rel_ns)
    return np.einsum("nij,nj->ni", R, B) + P


class Reference:
    """Prior map cropped around the trajectory: KD-tree (3 cm voxels) for residuals, 5 cm cloud with normals for ICP."""

    def __init__(self, ply: Path, traj_p: np.ndarray, margin: float = 60.0, residual_voxel: float = 0.0) -> None:
        import open3d as o3d
        from scipy.spatial import cKDTree
        pc = o3d.io.read_point_cloud(str(ply))
        lo, hi = traj_p.min(0) - margin, traj_p.max(0) + margin
        pc = pc.crop(o3d.geometry.AxisAlignedBoundingBox(lo, hi))
        self.n_points = len(pc.points)
        self.residual_voxel = float(residual_voxel)
        self.tree = cKDTree(np.asarray((pc.voxel_down_sample(self.residual_voxel) if self.residual_voxel > 0 else pc).points))
        self.icp = pc.voxel_down_sample(0.05)
        self.icp.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.3, max_nn=30))

    def residual(self, W: np.ndarray) -> dict:
        d, _ = self.tree.query(W, k=1, workers=8)
        return {"median_cm": float(np.median(d) * 100), "p90_cm": float(np.percentile(d, 90) * 100),
                "frac_lt_5cm": float((d < 0.05).mean()), "frac_lt_10cm": float((d < 0.10).mean())}

    def icp_correction(self, W: np.ndarray, T_WB: np.ndarray) -> tuple[np.ndarray, float]:
        """Body-frame rigid correction dT (world = T_WB dT base_from_lidar lidar) that best aligns W to the map, and ICP fitness."""
        import open3d as o3d
        src = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(W)).voxel_down_sample(0.1)
        T = np.eye(4)
        for thr in (0.5, 0.2):
            reg = o3d.pipelines.registration.registration_icp(src, self.icp, thr, T, o3d.pipelines.registration.TransformationEstimationPointToPlane(),
                                                              o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=60))
            T = reg.transformation
        return np.linalg.inv(T_WB) @ T @ T_WB, float(reg.fitness)


def average_corrections(dTs: list[np.ndarray]) -> tuple[np.ndarray, int]:
    """Mean rotation-vector / translation after dropping frames whose translation is a MAD outlier."""
    tr = np.stack([d[:3, 3] for d in dTs])
    rv = np.stack([Rotation.from_matrix(d[:3, :3]).as_rotvec() for d in dTs])
    dev = np.linalg.norm(tr - np.median(tr, 0), axis=1)
    keep = dev <= max(3.0 * np.median(dev), 0.02)
    return T_of(Rotation.from_rotvec(rv[keep].mean(0)).as_matrix(), tr[keep].mean(0)), int(keep.sum())


def sha256_files(paths: list[Path]) -> str:
    h = hashlib.sha256()
    for p in paths:
        with p.open("rb") as f:
            for block in iter(lambda: f.read(1 << 22), b""):
                h.update(block)
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw", type=Path, required=True)
    ap.add_argument("--sequence", default="05_quad_with_dynamics")
    ap.add_argument("--calibration-dir", default="04_calibration")
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--count", type=int, default=None, help="export only the first N scans that have a registered pose (default all)")
    ap.add_argument("--max-stamp-delta-ms", type=float, default=1.0, help="a scan is exported only if a registered pose lies within this")
    ap.add_argument("--reference", type=Path, default=None, help="prior-map ply (BLK360) used for --refine-extrinsic / --validate-frames")
    ap.add_argument("--refine-extrinsic", type=int, default=0, help="N evenly spaced exported frames used to estimate the constant base_from_lidar residual (0 = nominal chain)")
    ap.add_argument("--refine-passes", type=int, default=2)
    ap.add_argument("--validate-frames", type=int, default=0, help="M frames (disjoint from the refinement set) for the PREREG §1.1 checks")
    ap.add_argument("--check-range", type=float, nargs=2, default=(1.0, 40.0), help="range window of the returns used for ICP / residuals")
    ap.add_argument("--exclude-frames-mod", default=None, metavar="M:R",
                    help="never use exported frames with index %% M == R for the refinement / validation sets (e.g. 10:5 keeps the held-out split untouched)")
    ap.add_argument("--residual-voxel", type=float, default=0.0, help="voxel size of the prior-map copy used for the residual statistics (0 = full resolution)")
    ap.add_argument("--no-lidar", action="store_true", help="skip writing lidar/*.pcd (calibration / checks only)")
    a = ap.parse_args()

    raw = a.raw.expanduser(); seq_dir = raw / a.sequence; out = a.output.expanduser()
    bags = sorted(seq_dir.glob("rooster_*.bag"))
    if not bags:
        raise SystemExit(f"no rooster_*.bag under {seq_dir}")
    gt_csv = seq_dir / "ground_truth" / "registered_poses.csv"
    t0 = time.time()
    gt_t, gt_p, gt_q = load_gt(gt_csv)
    gt = GtInterp(gt_t, gt_p, gt_q)
    scans = index_scans(bags)
    hdr = np.asarray([s[2] for s in scans], dtype=np.int64)
    # match every registered pose to the scan with the same stamp
    j = np.searchsorted(hdr, gt_t)
    j = np.clip(j, 1, len(hdr) - 1)
    near = np.where(np.abs(hdr[j] - gt_t) < np.abs(hdr[j - 1] - gt_t), j, j - 1)
    delta_ms = (gt_t - hdr[near]) / 1e6
    ok = np.abs(delta_ms) <= a.max_stamp_delta_ms
    if len(np.unique(near[ok])) != int(ok.sum()):
        raise SystemExit("two registered poses map to one scan")
    frames = [(int(gi), int(si)) for gi, si in zip(np.flatnonzero(ok), near[ok])]  # (gt row, scan index in header order)
    if a.count is not None:
        frames = frames[: a.count]
    print(f"{len(scans)} scans in {len(bags)} bags, {len(gt_t)} registered poses, {int(ok.sum())} matched within {a.max_stamp_delta_ms} ms "
          f"(|delta| max {np.abs(delta_ms[ok]).max():.3f} ms), exporting {len(frames)} frames ({time.time()-t0:.0f}s)")

    T_BL, calib_info = nominal_base_from_lidar(raw / a.calibration_dir)
    print(f"nominal base_from_lidar ypr {np.round(calib_info['nominal_ypr_deg'], 3)} deg, t {np.round(T_BL[:3, 3] * 100, 2)} cm")

    checks: dict = {"reference": None, "refinement": None, "validation": None}
    ref = None
    if a.reference is not None and (a.refine_extrinsic > 0 or a.validate_frames > 0):
        ref = Reference(a.reference.expanduser(), gt_p[[f[0] for f in frames]], residual_voxel=a.residual_voxel)
        checks["reference"] = {"file": str(a.reference.expanduser()), "points_in_crop": ref.n_points, "residual_voxel_m": a.residual_voxel}
    n_ref, n_val = a.refine_extrinsic, a.validate_frames
    excluded = None
    if a.exclude_frames_mod:
        mod, rem = (int(v) for v in a.exclude_frames_mod.split(":"))
        excluded = (mod, rem)

    def allowed(k: int) -> int:
        """nearest exported frame index that is not excluded (searching upwards, then downwards)."""
        k = min(max(int(k), 0), len(frames) - 1)
        if excluded is None:
            return k
        for step in range(0, len(frames)):
            for cand in (k + step, k - step):
                if 0 <= cand < len(frames) and cand % excluded[0] != excluded[1]:
                    return cand
        raise SystemExit("every frame is excluded")
    ref_frames = [allowed(round(x)) for x in np.linspace(0, len(frames) - 1, n_ref + 2)[1:-1]] if n_ref else []
    val_frames = [allowed(round(x)) for x in np.linspace(0, len(frames) - 1, n_val + 2)[1:-1] + (len(frames) / (2 * (n_val + 1)) if n_ref else 0)] if n_val else []
    val_frames = [v for v in val_frames if v not in ref_frames]
    checks["frame_selection"] = {"exclude_frames_mod": a.exclude_frames_mod, "refinement_frames": ref_frames, "validation_frames": val_frames}
    need = sorted(set(ref_frames) | set(val_frames))
    cache: dict[int, tuple] = {}
    if need:
        wanted = {scans[frames[k][1]][1]: scans[frames[k][1]][2] for k in need}
        by_hdr = {scans[frames[k][1]][2]: k for k in need}
        lo, hi = a.check_range
        for h, arr in iter_scans(bags, wanted):
            xyz, inten, t_ns, rng = scan_arrays(arr)
            m = (rng >= lo) & (rng <= hi)
            cache[by_hdr[h]] = (xyz[m], t_ns[m])
    if ref is not None and n_ref:
        passes = []
        for p in range(a.refine_passes):
            dTs, fits = [], []
            for k in ref_frames:
                xyz, t_ns = cache[k]; stamp = scans[frames[k][1]][2]
                W = world_points(xyz, t_ns, stamp, gt, T_BL, deskew=True)
                dT, fit = ref.icp_correction(W, gt.pose(stamp)); dTs.append(dT); fits.append(fit)
            dT_mean, kept = average_corrections(dTs)
            tr = np.stack([d[:3, 3] for d in dTs]) * 100; ang = [np.degrees(np.linalg.norm(Rotation.from_matrix(d[:3, :3]).as_rotvec())) for d in dTs]
            passes.append({"pass": p, "frames": ref_frames, "kept": kept, "per_frame_translation_cm_mean": tr.mean(0).tolist(), "per_frame_translation_cm_std": tr.std(0).tolist(),
                           "per_frame_rotation_deg_mean": float(np.mean(ang)), "fitness_mean": float(np.mean(fits)),
                           "correction_ypr_deg": ypr_deg(dT_mean), "correction_translation_cm": (dT_mean[:3, 3] * 100).tolist()})
            print(f"refine pass {p}: {kept}/{len(dTs)} frames, mean per-frame correction {np.round(tr.mean(0), 2)} cm (std {np.round(tr.std(0), 2)}), "
                  f"{np.mean(ang):.2f} deg -> applying ypr {np.round(ypr_deg(dT_mean), 3)} deg, t {np.round(dT_mean[:3, 3] * 100, 2)} cm")
            T_BL = dT_mean @ T_BL
        checks["refinement"] = {"passes": passes, "refined_base_from_lidar": T_BL.tolist(), "refined_ypr_deg": ypr_deg(T_BL)}
    if ref is not None and val_frames:
        rows = []
        for k in val_frames:
            xyz, t_ns = cache[k]; stamp = scans[frames[k][1]][2]
            r_raw = ref.residual(world_points(xyz, t_ns, stamp, gt, T_BL, deskew=False))
            W = world_points(xyz, t_ns, stamp, gt, T_BL, deskew=True); r_dsk = ref.residual(W)
            dT, fit = ref.icp_correction(W, gt.pose(stamp))
            T_WB = gt.pose(stamp); T_fit = T_WB @ dT @ np.linalg.inv(T_WB)      # world-frame ICP transform of this scan
            r_fit = ref.residual(W @ T_fit[:3, :3].T + T_fit[:3, 3])            # floor: the same scan after its own best rigid alignment
            rows.append({"frame": k, "raw": r_raw, "deskewed": r_dsk, "deskewed_after_own_icp": r_fit, "residual_icp_translation_cm": (dT[:3, 3] * 100).tolist(),
                         "residual_icp_rotation_deg": float(np.degrees(np.linalg.norm(Rotation.from_matrix(dT[:3, :3]).as_rotvec()))), "fitness": fit})
        med_raw = np.array([r["raw"]["median_cm"] for r in rows]); med_dsk = np.array([r["deskewed"]["median_cm"] for r in rows])
        med_fit = np.array([r["deskewed_after_own_icp"]["median_cm"] for r in rows])
        rt_ = np.array([r["residual_icp_translation_cm"] for r in rows]); rr = np.array([r["residual_icp_rotation_deg"] for r in rows])
        summary = {"frames": val_frames, "median_cm_raw_mean": float(med_raw.mean()), "median_cm_deskewed_mean": float(med_dsk.mean()),
                   "median_cm_deskewed_median_over_frames": float(np.median(med_dsk)), "median_cm_after_own_icp_median_over_frames": float(np.median(med_fit)),
                   "deskew_le_raw_frames": int((med_dsk <= med_raw).sum()), "n_frames": len(rows),
                   "residual_icp_translation_cm_mean": rt_.mean(0).tolist(), "residual_icp_translation_cm_rms": float(np.sqrt((rt_ ** 2).sum(1).mean())),
                   "residual_icp_rotation_deg_mean": float(rr.mean()),
                   "check_deskew_reduces_residual": bool(med_dsk.mean() <= med_raw.mean()), "check_median_le_5cm": bool(np.median(med_dsk) <= 5.0)}
        checks["validation"] = {"summary": summary, "frames": rows}
        print(f"validation on {len(rows)} frames: median residual raw {med_raw.mean():.2f} cm -> deskewed {med_dsk.mean():.2f} cm (mean over frames; "
              f"median over frames {np.median(med_dsk):.2f} cm, floor after each scan's own ICP {np.median(med_fit):.2f} cm); deskew <= raw in {summary['deskew_le_raw_frames']}/{len(rows)}; "
              f"residual ICP correction rms {summary['residual_icp_translation_cm_rms']:.2f} cm, {rr.mean():.2f} deg; median<=5cm: {summary['check_median_le_5cm']}")

    # ---- write the canonical sequence
    out.mkdir(parents=True, exist_ok=True); (out / "lidar").mkdir(exist_ok=True)
    manifest, counts = [], []
    if not a.no_lidar:
        wanted = {scans[si][1]: scans[si][2] for _, si in frames}
        k_of = {scans[si][2]: k for k, (_, si) in enumerate(frames)}
        done = 0
        for h, arr in iter_scans(bags, wanted):
            k = k_of[h]; xyz, inten, t_ns, rng = scan_arrays(arr)
            fid = f"frame_{k:06d}"
            n = write_pcd(out / "lidar" / f"{fid}.pcd", xyz, inten, t_ns)
            counts.append(n)
            gi, si = frames[k]
            manifest.append({"frame_id": fid, "lidar_path": f"lidar/{fid}.pcd", "lidar_timestamp_ns": int(h), "image_path": None, "image_timestamp_ns": None,
                             "point_time_field": "offset_time", "point_time_unit": "nanoseconds",
                             "metadata": {"source_scan_index": si, "gt_row": gi, "bag": scans[si][0].name, "point_count": n, "pose_source": "NCD registered_poses.csv (T_WB, exact stamp)"}})
            done += 1
            if done % 500 == 0:
                print(f"  {done}/{len(frames)} scans written ({time.time()-t0:.0f}s)")
        manifest.sort(key=lambda m: m["lidar_timestamp_ns"])
        (out / "manifest.jsonl").write_text("".join(json.dumps(m) + "\n" for m in manifest))
    traj = [f"{t / NS:.9f} {p[0]:.6f} {p[1]:.6f} {p[2]:.6f} {q[0]:.9f} {q[1]:.9f} {q[2]:.9f} {q[3]:.9f}" for t, p, q in zip(gt_t, gt_p, gt_q)]
    first, last = traj[0].split(" ", 1), traj[-1].split(" ", 1)
    traj = [f"{float(first[0]) - 0.1:.9f} {first[1]}"] + traj + [f"{float(last[0]) + 0.1:.9f} {last[1]}"]
    (out / "trajectory.txt").write_text("\n".join(traj) + "\n")
    (out / "calibration.json").write_text(json.dumps({
        "imu_from_lidar": rt(T_BL),
        "notes": "canonical 'imu' frame = NCD base frame (left RealSense position, robotic axes); imu_from_lidar = base_from_lidar"
                 + (" refined by ICP of deskewed scans against the prior map (see export_checks.json)" if checks["refinement"] else " (nominal kalibr chain)"),
        "ypr_deg": ypr_deg(T_BL), "provenance": calib_info}, indent=2))
    checks["frames"] = {"exported": len(frames), "first_gt_row": frames[0][0], "first_scan_index": frames[0][1], "last_scan_index": frames[-1][1],
                        "stamp_delta_ms_max": float(np.abs(delta_ms[ok]).max())}
    (out / "export_checks.json").write_text(json.dumps(checks, indent=2))
    meta_hash = sha256_files([out / p for p in ("manifest.jsonl", "trajectory.txt", "calibration.json") if (out / p).exists()])
    lidar_hash = sha256_files(sorted((out / "lidar").glob("*.pcd"))) if not a.no_lidar else None
    (out / "dataset.json").write_text(json.dumps({
        "format": "gubmap-canonical-v1", "sequence": out.name, "source": f"Newer College 2020 {a.sequence} ({', '.join(b.name for b in bags)})",
        "sensor": "Ouster OS1-64 (64x1024, 10 Hz), handheld", "pose_frame": "T_WB: NCD base frame (RS_C1 position, robotic axes) in the BLK360 prior-map frame = registered_poses.csv",
        "point_time": "offset_time = ns since the scan header stamp (first column)", "exported_frames": len(frames),
        "scan_index_range": [frames[0][1], frames[-1][1]], "gt_rows_used": [frames[0][0], frames[-1][0]],
        "extrinsic": "refined" if checks["refinement"] else "nominal", "reference_map": checks["reference"]["file"] if checks["reference"] else None,
        "sha256_manifest_trajectory_calibration": meta_hash, "sha256_lidar_pcds": lidar_hash, "gt_csv_sha256": sha256_files([gt_csv]),
        "exporter": Path(__file__).name}, indent=2))
    (out / "README.md").write_text(
        f"# {out.name}\n\nNewer College 2020 `{a.sequence}` (Ouster OS1-64 handheld) in gubmap-canonical-v1 layout, exported by dataset/converter/{Path(__file__).name}.\n\n"
        f"- LiDAR: {len(frames)} scans with a registered pose (scan index {frames[0][1]}..{frames[-1][1]}), binary PCD `x y z intensity offset_time` (offset_time = ns since the header stamp).\n"
        "- Poses: `registered_poses.csv` (10 Hz, ICP of each scan to the BLK360 prior map; stamps equal the scan header stamps) written as T_WB; the canonical\n"
        "  'imu' frame is the NCD base frame (left RealSense position, x-forward / y-left / z-up).\n"
        f"- imu_from_lidar = base_from_lidar ({'refined against the prior map, see export_checks.json' if checks['refinement'] else 'nominal kalibr chain'}); ypr {np.round(ypr_deg(T_BL), 3)} deg.\n"
        "- No images.\n")
    print(f"exported {len(frames)} frames to {out} in {time.time()-t0:.0f}s; points/scan mean {np.mean(counts) if counts else 0:.0f}; "
          f"meta sha256 {meta_hash[:16]}… lidar sha256 {(lidar_hash or '-')[:16]}…")


if __name__ == "__main__":
    main()
