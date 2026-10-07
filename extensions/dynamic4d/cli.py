"""Explicit command boundary for the optional dynamic reconstruction extension."""
from __future__ import annotations
import argparse
import importlib.util
import json
import os
from pathlib import Path


def load_config(path):
    """The case inputs (see INPUTS.md): scenes_json, initial_model, partitions, image_cache, scene_manifest."""
    config = json.loads(Path(path).read_text())
    required = {"scenes_json", "initial_model", "partitions", "image_cache", "scene_manifest"}
    allowed = required | {"mesh_every", "background_radius_m"}
    if not required.issubset(config) or set(config) - allowed:
        raise ValueError(f"Invalid config keys; required={sorted(required)}, unknown={sorted(set(config)-allowed)}")
    for key in required:
        if not Path(config[key]).exists():
            raise FileNotFoundError(f"{key}: {config[key]}")
    if int(config.get("mesh_every", 5)) < 1 or float(config.get("background_radius_m", 12)) <= 0:
        raise ValueError("mesh_every and background_radius_m must be positive")
    return config


def main(argv=None):
    parser = argparse.ArgumentParser(description="Track a dynamic target with optional image gradients, fuse its local mesh, and export an RViz ROS 2 bag.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="New output directory; never overwrite an existing experiment")
    parser.add_argument("--image-mode", choices=("features", "off"), default="features")
    parser.add_argument("--fusion-policy", choices=("accepted", "baseline"), default="accepted",
                        help="accepted protects shape after rejected registration; baseline retains the old prediction-fusion control")
    parser.add_argument("--end", type=int, help="Exclusive final frame within the frozen case")
    parser.add_argument("--vehicle-heading-guard-deg", type=float, default=None,
                        help="Optional vehicle heading guard (guard.py): cone in degrees between heading and reliable travel direction; off by default")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--track-only", action="store_true", help="Diagnostic only; omit bag export")
    parser.add_argument("--export-run", type=Path, help="Export a saved run to a NEW output directory without rerunning tracking")
    args = parser.parse_args(argv)
    try:
        output = args.output.expanduser().resolve()
        if output.exists() or args.output.is_symlink():
            raise FileExistsError(f"Refusing to reuse output: {output}")
        config = load_config(args.config)
        if args.export_run and (args.track_only or args.end is not None):
            raise ValueError("--export-run cannot be combined with --track-only or --end")
        if args.vehicle_heading_guard_deg is not None:
            if not 0 < args.vehicle_heading_guard_deg < 180:
                raise ValueError("--vehicle-heading-guard-deg must be in (0, 180)")
            if args.fusion_policy != "accepted" or args.export_run:
                raise ValueError("--vehicle-heading-guard-deg requires --fusion-policy accepted and cannot be combined with --export-run")
        if args.dry_run:
            print(json.dumps(dict(output=str(output), image_mode=args.image_mode, fusion_policy=args.fusion_policy,
                                  config=config, bag=str(output / "scene_bag"), end=args.end,
                                  vehicle_heading_guard_deg=args.vehicle_heading_guard_deg), indent=2))
            return 0
        os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")
        for module in ("numpy", "torch", "open3d", "scipy"):
            if importlib.util.find_spec(module) is None:
                raise RuntimeError(f"Missing dependency: {module}; see extensions/dynamic4d/INPUTS.md")
        if args.image_mode == "features" and importlib.util.find_spec("cv2") is None:
            raise RuntimeError("Missing OpenCV; install opencv-python or supply its existing runtime path")
        if not args.track_only:
            for module in ("rosbag2_py", "rclpy", "visualization_msgs"):
                if importlib.util.find_spec(module) is None:
                    raise RuntimeError("ROS 2 environment is missing. Source /opt/ros/jazzy/setup.bash before running.")
        from .pipeline import save_json, track
        output.mkdir(parents=True, exist_ok=False)
        serialized_args = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
        save_json(output / "config.json", dict(config=config, arguments=serialized_args))
        try:
            if args.export_run:
                run = json.loads(args.export_run.read_text())
                if run.get("gt_used") is not False or not run.get("rows"):
                    raise ValueError("Expected a completed, non-GT tracking run")
                if any(str(Path(config[key]).resolve()) not in run["input_sha256"]
                       for key in ("scenes_json", "initial_model", "scene_manifest")):
                    raise ValueError("Export configuration does not match the saved run inputs")
                from .pipeline import digest
                if any(digest(p) != h for p, h in run["input_sha256"].items()):
                    raise ValueError("Recorded run inputs no longer match their hashes")
                save_json(output / "run.json", run)
            else:
                run = track(config, output, args.image_mode, args.end, args.fusion_policy, args.vehicle_heading_guard_deg)
            if not args.track_only:
                from .bag import export_bag
                export_bag(config, run, output)
            save_json(output / "COMPLETE.json", dict(tracking_complete=True, bag_complete=not args.track_only,
                       image_attempted=sum(r["image"]["attempted"] for r in run["rows"]),
                       image_accepted=sum(r["image"]["accepted"] for r in run["rows"]),
                       output=str(output)))
            print(f"Completed: {output}", flush=True)
        except Exception as error:
            save_json(output / "FAILED.json", dict(error=type(error).__name__, detail=str(error)))
            raise
        return 0
    except (ValueError, OSError, RuntimeError, ImportError) as error:
        parser.exit(2, f"error: {error}\n")


if __name__ == "__main__":
    raise SystemExit(main())
