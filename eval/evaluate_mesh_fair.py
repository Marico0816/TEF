#!/usr/bin/env python3
"""Score any mesh with GUBMap's own held-out LiDAR ray protocol.

Thin wrapper around the bundled ``evaluate_mesh_rays.py`` so
that baseline meshes (VDBFusion, PIN-SLAM, PINGS, ...) are scored with exactly
the same code path, rays, tolerances and canonical FAST-LIVO2 poses as the
GUBMap ``fair_compare`` numbers:

    selection = held-out indices 5, 15, ..., (stop-5)
    10,000 rays per frame, tolerances 2 / 5 / 10 cm, canonical poses

The evaluator needs a checkpoint only to know which frames were used for
training; for external baselines we synthesise a minimal NPZ that lists the
training indices (default every-10: 0, 10, ..., stop-10).

Requires Open3D from the project dependencies.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))


def _load_evaluator():
    spec = importlib.util.spec_from_file_location("evaluate_mesh_rays", REPO / "eval" / "evaluate_mesh_rays.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("mesh", type=Path)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=None, help="GUBMap checkpoint; if omitted a protocol stub is synthesised")
    parser.add_argument("--train-start", type=int, default=0)
    parser.add_argument("--train-step", type=int, default=10)
    parser.add_argument("--heldout-start", type=int, default=5)
    parser.add_argument("--heldout-step", type=int, default=10)
    parser.add_argument("--stop", type=int, default=None, help="dataset length by default (300 for Campus01, 175 for Campus00)")
    parser.add_argument("--max-rays-per-frame", type=int, default=10_000)
    parser.add_argument("--tolerances-m", type=float, nargs="+", default=(0.02, 0.05, 0.10))
    parser.add_argument("--label", type=str, default=None)
    parser.add_argument("--skip-mesh-components", action="store_true",
                        help="do not compute connected-component statistics of the mesh (descriptive only; needed for meshes of tens of millions of triangles on a 30 GB host)")
    parser.add_argument("--deskew-heldout", action="store_true",
                        help="held-out returns are placed with the sensor pose at their own time (per-point time offsets + trajectory) and rays start "
                             "from where the sensor was at that time; needed to score maps built from deskewed Livox scans fairly (KITTI raw: no-op)")
    parser.add_argument("--poses-npz", type=Path, default=None,
                        help="online_mapper.py --pose-refine-out file: held-out frames listed in it are evaluated at their scan-to-map "
                             "localised pose instead of the canonical FAST-LIVO2 pose (diagnostic protocol; always report both)")
    args = parser.parse_args()

    evaluator = _load_evaluator()
    from dataset import DatasetLoader

    # The ray metrics never touch RGB, so skip image decoding (also lets the
    # evaluation run on a copy of the dataset that has no images/ folder).
    import functools

    evaluator.DatasetLoader = functools.partial(DatasetLoader, load_images=False)

    length = len(DatasetLoader(args.dataset, load_images=False))
    stop = length if args.stop is None else min(args.stop, length)
    train = list(range(args.train_start, stop, args.train_step))
    heldout = [i for i in range(args.heldout_start, stop, args.heldout_step) if i not in set(train)]

    pose_source = "canonical FAST-LIVO2 (DatasetLoader)" + (" + per-point deskew of held-out returns" if args.deskew_heldout else "")
    if args.poses_npz is not None:
        import dataclasses
        z = np.load(args.poses_npz)
        override = {int(i): np.asarray(T, dtype=np.float64) for i, T in zip(z["frame_index"], z["world_from_lidar_refined"])}
        missing = [i for i in heldout if i not in override]
        if missing:
            raise SystemExit(f"--poses-npz has no pose for held-out frames {missing[:10]}{'...' if len(missing) > 10 else ''}")

        class _LocalisedLoader(DatasetLoader):
            def __getitem__(self, index):
                f = super().__getitem__(index)
                T = override.get(int(index))
                if T is None:
                    return f
                corr = T @ np.linalg.inv(np.asarray(f.world_from_lidar, dtype=np.float64))
                return dataclasses.replace(f, world_from_lidar=T, world_from_imu=corr @ np.asarray(f.world_from_imu, dtype=np.float64),
                                           world_from_camera=None if f.world_from_camera is None else corr @ np.asarray(f.world_from_camera, dtype=np.float64))

        evaluator.DatasetLoader = functools.partial(_LocalisedLoader, load_images=False)
        pose_source = f"scan-to-map localised poses from {args.poses_npz} ({sum(1 for i in heldout if i in override)} held-out frames overridden)"

    checkpoint = args.checkpoint
    temp = None
    if checkpoint is None:
        temp = tempfile.NamedTemporaryFile(suffix=".npz", delete=False)
        np.savez(temp.name, support_frame_indices=np.asarray(train, dtype=np.int64))
        checkpoint = Path(temp.name)

    result = evaluator.evaluate(
        args.mesh,
        checkpoint,
        args.dataset,
        selection="heldout",
        max_rays_per_frame=args.max_rays_per_frame,
        tolerances_m=tuple(float(v) for v in args.tolerances_m),
        start=0,
        stop=stop,
        frame_step=1,
        frame_indices=heldout,
        use_checkpoint_poses=False,
        deskew_heldout=bool(args.deskew_heldout),
        mesh_components=not bool(args.skip_mesh_components),
    )
    result["label"] = args.label or args.mesh.stem
    result["protocol"] = {
        "training_frame_indices": train,
        "heldout_frame_indices": heldout,
        "pose_source": pose_source,
        "rays_per_frame": args.max_rays_per_frame,
        "note": "same code path as eval/evaluate_mesh_rays.py --no-checkpoint-poses",
    }
    if temp is not None:
        os.unlink(temp.name)
        result["checkpoint"] = "synthesised protocol stub (training indices only)"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    o = result["overall"]
    t5 = o["tolerances"]["5cm"]
    print(
        f"{result['label']}: rays={o['rays']} hit={o['finite_hit_fraction']:.4f} "
        f"median={o['endpoint_distance_median_m']*100:.2f}cm p90={o['endpoint_distance_p90_m']*100:.2f}cm "
        f"5cm free={t5['free_space_violation_fraction']*100:.2f}% agree={t5['surface_agreement_fraction']*100:.2f}% "
        f"missing={t5['missing_ray_fraction']*100:.2f}% | V={result['mesh_summary']['vertices']} F={result['mesh_summary']['triangles']} "
        f"CC={result['mesh_summary']['connected_components']}"
    )


if __name__ == "__main__":
    main()
