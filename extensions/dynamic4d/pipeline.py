"""Integrated causal object tracking and shape fusion on frozen ownership inputs.

Per frame of the case interval: predict the target's 4-DoF state from its velocity, register the target's returns
(ownership partition) to the object mesh, optionally refine the pose with image features (30 Adam steps on
[x, y, z, yaw], accepted only after the gates in ``visual.py``), update the velocity from accepted LiDAR states only,
fuse the returns into the object-frame TSDF after an accepted registration, and extract the object mesh every
``mesh_every`` frames.  The T2/P2 mapper is not involved; the object field is ``object_field.py``.
"""
from __future__ import annotations
import hashlib
import json
import time
from pathlib import Path
import numpy as np
from .visual import FeatureTrack, inverse, matrix, refine


def clean_json(value):
    if isinstance(value, dict):
        return {str(k): clean_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean_json(v) for v in value]
    if isinstance(value, np.ndarray):
        return clean_json(value.tolist())
    if isinstance(value, np.generic):
        return clean_json(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def save_json(path, value):
    with Path(path).open("x", encoding="utf-8") as handle:
        json.dump(clean_json(value), handle, indent=2, allow_nan=False)


def digest(path):
    sha = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            sha.update(chunk)
    return sha.hexdigest()


def owned_points(directory, frame, packet, identity):
    path = Path(directory) / f"{frame:04d}.npz"
    with np.load(path, allow_pickle=False) as data:
        required = {"cluster_id", "frame", "stamp_ns", "point_sha256", "source_kind"}
        if not required.issubset(data.files):
            raise ValueError(f"Incomplete partition: {path}")
        points = np.asarray(packet.points_lidar, dtype="<f8")
        sha = hashlib.sha256(str(points.shape).encode() + points.tobytes(order="C")).hexdigest()
        if (int(data["frame"]) != frame or int(data["stamp_ns"]) != int(packet.lidar_timestamp_ns)
                or str(data["point_sha256"]) != sha or str(data["source_kind"]) != "full_return_frontend"):
            raise ValueError(f"Partition identity, timestamp, or point ordering mismatch: {path}")
        ids = data["cluster_id"]
        if ids.shape != (len(points),) or ids.dtype.kind not in "iu" or np.any(ids < -2):
            raise ValueError(f"Invalid ownership labels: {path}")
        return points[ids == identity]


def image_frame(directory, frame, packet, calibration):
    path = Path(directory) / f"{frame:04d}.npz"
    if not path.exists():
        return None
    names = ("gray", "vehicle", "camera", "K", "image_dt", "lidar_timestamp_ns", "image_timestamp_ns")
    with np.load(path, allow_pickle=False) as data:
        if not set(names).issubset(data.files):
            raise ValueError(f"Incomplete image cache: {path}")
        result = {name: data[name].copy() for name in names}
    if packet.world_from_camera is None or packet.image_timestamp_ns is None:
        return None
    if (int(result["lidar_timestamp_ns"]) != int(packet.lidar_timestamp_ns)
            or int(result["image_timestamp_ns"]) != int(packet.image_timestamp_ns)):
        raise ValueError(f"Image cache timestamp mismatch: {path}")
    delta = (packet.image_timestamp_ns - packet.lidar_timestamp_ns) * 1e-9
    if abs(float(result["image_dt"]) - delta) > 1e-9:
        raise ValueError(f"Image cache time offset mismatch: {path}")
    if not np.allclose(result["camera"], inverse(packet.world_from_camera), atol=1e-7, rtol=0):
        raise ValueError(f"Image cache pose mismatch: {path}")
    if not np.allclose(result["K"], calibration.camera_intrinsics, atol=1e-7, rtol=0):
        raise ValueError(f"Image cache intrinsics mismatch: {path}")
    gray, mask = result["gray"], result["vehicle"]
    if gray.dtype != np.uint8 or gray.ndim != 2 or mask.shape != gray.shape:
        raise ValueError(f"Image cache shape mismatch: {path}")
    result["vehicle"] = mask.astype(bool)
    result["image_dt"] = float(result["image_dt"])
    return result


def update_velocity(velocity, measured, last_measured, dt, ema=.5, limit=.5):
    """Use accepted pre-image LiDAR states, not image-correction jumps."""
    difference = measured - last_measured
    difference[3] = np.arctan2(np.sin(difference[3]), np.cos(difference[3]))
    observed = difference / max(dt, 1e-6)
    proposed = ema * velocity + (1 - ema) * observed
    return velocity + np.clip(proposed - velocity, -limit, limit)


def track(config, output, image_mode="features", end=None, fusion_policy="accepted", vehicle_heading_guard_deg=None):
    from dataset import DatasetLoader
    from .object_field import build_prefix_field
    from .objects import CachedTracks, InitialModel, PointToPlaneRegistration, RegistrationConfig, deterministic_points
    import torch
    torch.set_num_threads(4)
    if image_mode == "features":
        import cv2
        cv2.setNumThreads(4)
    started = time.perf_counter()
    scene = json.loads(Path(config["scenes_json"]).read_text())
    case = scene["case"]
    start = int(case["start"])
    end = int(case["end"] if end is None else end)
    if not start < end <= int(case["end"]):
        raise ValueError("end must be after the case start and within the frozen case")
    model = InitialModel.load(config["initial_model"])
    if model.frame >= start or model.frame != int(case["initial"]):
        raise ValueError("Initial model must precede the requested interval")
    ds = DatasetLoader(case["dataset"], min_range_m=.5, max_range_m=50., load_images=False)
    tracks = CachedTracks(case["tracks"])
    if any(int(tracks.z["frame"][j]) > model.frame or int(tracks.z["track_id"][j]) != case["track_id"]
           for j in case["prefix_rows"]):
        raise ValueError("Prefix contains future or foreign-object observations")
    field = build_prefix_field(tracks, case["prefix_rows"])
    geometry = PointToPlaneRegistration(model.vertices, model.faces, model.reference, RegistrationConfig())
    output = Path(output)
    shapes = output / "meshes"
    shapes.mkdir()
    prefix_path = shapes / "initial.npz"
    np.savez_compressed(prefix_path, vertices=model.vertices, faces=model.faces)
    revision = dict(path=str(prefix_path), available_frame=model.frame, source_frame=model.frame)
    t0 = int(ds[0].lidar_timestamp_ns)
    if abs((int(ds[model.frame].lidar_timestamp_ns) - t0) * 1e-9 - model.time) > 1e-6:
        raise ValueError("Initial model timestamp does not match this dataset")
    features = FeatureTrack()
    last, velocity, last_t = model.state.copy(), model.velocity.copy(), model.time
    last_lidar, last_lidar_t = last.copy(), last_t
    rows, revisions = [], [revision.copy()]
    hashes = {str(Path(p).resolve()): digest(p) for p in (
        config["scenes_json"], config["initial_model"], config["scene_manifest"],
        Path(case["dataset"]) / "manifest.jsonl", Path(case["dataset"]) / "trajectory.txt")}
    guard = None
    if vehicle_heading_guard_deg is not None:
        from .guard import COMMITTED, FUSED, VehicleHeadingGuard, track_class
        if fusion_policy != "accepted":
            raise ValueError("The vehicle heading guard requires --fusion-policy accepted")
        guard = VehicleHeadingGuard(vehicle_heading_guard_deg, track_class(tracks, case["track_id"]), model,
                                    geometry.cfg.step_clip_deg, geometry.cfg.step_clip_m)
    guarded = guard is not None and guard.enabled
    for f in range(start, end):
        tick = time.perf_counter()
        packet = ds[f]
        now = (int(packet.lidar_timestamp_ns) - t0) * 1e-9
        predicted = last + velocity * (now - last_t)
        state, base = predicted.copy(), predicted.copy()
        heldout = f % 10 == 5
        points_l, points_w = np.empty((0, 3)), np.empty((0, 3))
        geo = dict(accepted=False, reason="heldout", pairs=0, rmse_before=None, rmse_after=None)
        imeta = dict(attempted=False, accepted=False, reason="heldout" if heldout else "disabled", steps=0)
        flow, seed, image = {}, {}, None
        registered_s = optimized_s = fused_s = extracted_s = 0.
        ginfo = None
        if not guarded:
            if not heldout:
                points_l = owned_points(config["partitions"], f, packet, int(case["track_id"]))
                hashes[str(Path(config["partitions"]) / f"{f:04d}.npz")] = digest(Path(config["partitions"]) / f"{f:04d}.npz")
                points_w = deterministic_points(points_l @ packet.world_from_lidar[:3, :3].T + packet.world_from_lidar[:3, 3])
                t1 = time.perf_counter()
                base, result = geometry.refine(predicted, points_w)
                registered_s = time.perf_counter() - t1
                geo = result.as_dict()
                state = base.copy()
                if image_mode == "features":
                    t1 = time.perf_counter()
                    image = image_frame(config["image_cache"], f, packet, ds.calibration)
                    if image is None:
                        features.reset()
                        imeta["reason"] = "missing_image"
                    else:
                        hashes[str(Path(config["image_cache"]) / f"{f:04d}.npz")] = digest(Path(config["image_cache"]) / f"{f:04d}.npz")
                        flow = features.advance(image)
                        state, imeta = refine(base, predicted, velocity, points_w, features.anchors,
                                             features.pixels, image, geometry, joint=bool(geo["accepted"]))
                    optimized_s = time.perf_counter() - t1
                if geo["accepted"]:
                    velocity = update_velocity(velocity, base, last_lidar, now - last_lidar_t)
                    last_lidar, last_lidar_t = base.copy(), now
                if geo["accepted"] or imeta["accepted"]:
                    last, last_t = state.copy(), now
                if image is not None:
                    if geo["accepted"]:
                        seed = features.seed(geometry, state, velocity, points_w, image)
                    features.finish(image)
            # Image-only updates never authorize new shape. Baseline policy is an explicit
            # replay control; accepted is the safer extension default.
            fused = bool(not heldout and len(points_l) and
                         (geo["accepted"] or (fusion_policy == "baseline" and not imeta["accepted"])))
            status = ("lidar_image" if geo["accepted"] and imeta["accepted"] else "lidar" if geo["accepted"]
                      else "image_only" if imeta["accepted"] else "heldout_prediction" if heldout else "prediction")
            accepted_row = bool(geo["accepted"] or imeta["accepted"])
        else:
            # Optional vehicle heading guard (guard.py): one outcome per frame decides display, commit, motion and fusion.
            if not heldout:
                points_l = owned_points(config["partitions"], f, packet, int(case["track_id"]))
                hashes[str(Path(config["partitions"]) / f"{f:04d}.npz")] = digest(Path(config["partitions"]) / f"{f:04d}.npz")
                points_w = deterministic_points(points_l @ packet.world_from_lidar[:3, :3].T + packet.world_from_lidar[:3, 3])
                t1 = time.perf_counter()
                base, result = geometry.refine(predicted, points_w)
                geo = result.as_dict()
                registered_s = time.perf_counter() - t1
            t1 = time.perf_counter()
            course, travel_age = guard.course(), now - guard.travel_t
            outcome, lidar_state, ginfo = guard.classify(heldout, base, geo, predicted, points_w, geometry)
            registered_s += time.perf_counter() - t1
            candidate = discarded = None
            if not heldout and image_mode == "features":
                t1 = time.perf_counter()
                image = image_frame(config["image_cache"], f, packet, ds.calibration)
                if image is None:
                    features.reset()
                    imeta["reason"] = "missing_image"
                else:
                    hashes[str(Path(config["image_cache"]) / f"{f:04d}.npz")] = digest(Path(config["image_cache"]) / f"{f:04d}.npz")
                    flow = features.advance(image)
                    if outcome == "valid":
                        candidate, imeta = refine(base, predicted, velocity, points_w, features.anchors,
                                                  features.pixels, image, geometry, joint=True)
                    elif outcome == "prediction":
                        candidate, imeta = refine(predicted, predicted, velocity, points_w, features.anchors,
                                                  features.pixels, image, geometry, joint=False)
                    else:
                        imeta = dict(attempted=False, accepted=False, reason="heading_guard_" + outcome, steps=0)
                    if imeta["accepted"] and not guard.consistent(candidate, course):
                        ginfo["events"].append("image_discard")
                        discarded = candidate.tolist()
                        imeta = dict(imeta, accepted=False, reason="heading_guard_discard", discarded_candidate=discarded)
                optimized_s = time.perf_counter() - t1
            if outcome == "prediction" and imeta["accepted"]:
                outcome = "image_only"
            h_appended = outcome in FUSED
            if outcome == "valid":
                state = candidate.copy() if imeta["accepted"] else lidar_state.copy()
                velocity = update_velocity(velocity, lidar_state, last_lidar, now - last_lidar_t)
                last_lidar, last_lidar_t = lidar_state.copy(), now
                if ginfo["check_applicable"] and now - guard.start_t <= 1.0:
                    guard.early.append(guard.deviation(lidar_state, course))
                guard.append(now, lidar_state)
            elif outcome in ("restart", "reacquired"):
                state = lidar_state.copy()
                guard.append(now, lidar_state)
                velocity = guard.velocity()
                last_lidar, last_lidar_t = lidar_state.copy(), now
                guard.mode = "tracking"
            elif outcome == "enter":
                state = guard.anchor(predicted, course)
                velocity = guard.velocity()
                guard.mode = "constrained"
            elif outcome == "image_only":
                state = candidate.copy()
            else:
                state = predicted.copy()
            committed = outcome in COMMITTED
            if committed:
                last, last_t = state.copy(), now
            if image is not None:
                if outcome in FUSED:
                    seed = features.seed(geometry, state, velocity, points_w, image)
                features.finish(image)
            fused = bool(not heldout and len(points_l) and outcome in FUSED)
            for event in ginfo["events"]:
                guard.event_counts[event] += 1
            ginfo.update(mode=guard.mode, outcome=outcome, committed=committed, h_appended=h_appended,
                         lidar_state=None if lidar_state is None else lidar_state.tolist(),
                         course_after_deg=float(np.degrees(guard.course())), travel_age_s=travel_age,
                         displayed_deviation_deg=float(np.degrees(guard.deviation(state, course))) if ginfo["check_applicable"] else None,
                         discarded_state=discarded)
            status = ("lidar_image" if outcome == "valid" and imeta["accepted"] else "lidar" if outcome in FUSED
                      else "image_only" if outcome == "image_only"
                      else ("heldout_constrained" if heldout else "constrained_prediction") if outcome in ("enter", "constrained")
                      else "heldout_prediction" if heldout else "prediction")
            accepted_row = outcome in ("valid", "restart", "reacquired", "image_only")
        if fused:
            t1 = time.perf_counter()
            field.add_observation(int(packet.lidar_timestamp_ns), inverse(matrix(state, model.reference)) @ packet.world_from_lidar, points_l)
            fused_s = time.perf_counter() - t1
        if (f - start) % int(config.get("mesh_every", 5)) == 0 or heldout or f == end - 1:
            t1 = time.perf_counter()
            if field.dirty or field.mesh is None:
                field.extract(final=(f == end - 1))
                V, F = field.mesh
                path = shapes / f"target_f{f:04d}.npz"
                np.savez_compressed(path, vertices=np.asarray(V, np.float32), faces=np.asarray(F, np.int32))
                revision = dict(path=str(path), available_frame=f,
                                source_frame=f if fused else max([r["frame"] for r in rows if r["fused"]], default=model.frame))
                revisions.append(revision.copy())
            extracted_s = time.perf_counter() - t1
        row = dict(frame=f, stamp_ns=int(packet.lidar_timestamp_ns), t=now, heldout=heldout,
                   prediction=predicted.tolist(), state=state.tolist(), pose=matrix(state, model.reference).tolist(),
                   velocity=velocity.tolist(), accepted=accepted_row,
                   status=status, fused=fused, n_obs=len(points_l), geometry=geo, image=imeta, flow=flow, seed=seed,
                   shape_revision=revision.copy(), registration_s=registered_s, image_s=optimized_s,
                   fusion_s=fused_s, extraction_s=extracted_s, total_s=time.perf_counter() - tick,
                   observation_available_stamp_ns=int(packet.image_timestamp_ns) if image is not None else int(packet.lidar_timestamp_ns))
        if ginfo is not None:
            row["vehicle_guard"] = ginfo
        rows.append(row)
        if f == start or f % 10 == 9 or f == end - 1:
            print(f"frame {f}: {status}, points={len(points_l)}, image_steps={imeta['steps']}, mesh={Path(revision['path']).name}", flush=True)
    record = dict(case=case, image_mode=image_mode, fusion_policy=fusion_policy, start=start, end=end,
                  rows=rows, meshes=revisions, final_mesh=revision, gt_used=False,
                  input_sha256=hashes, elapsed_s=time.perf_counter() - started,
                  limits=["Frozen ownership and semantic image caches; not an online frontend.",
                          "New run estimates one target; background/other-object snapshots are reused.",
                          "Image correction is four-DoF pose optimization, not neural-network training.",
                          "No future-pose interpolation; current estimates include marked predictions.",
                          "Timing excludes producing the frozen frontend caches."])
    if guard is not None:
        record["vehicle_guard"] = guard.record()
    save_json(output / "run.json", record)
    return clean_json(record)
