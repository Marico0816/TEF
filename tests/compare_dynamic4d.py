#!/usr/bin/env python3
"""Equivalence check of the dynamic extension against the research implementation (research checkout read only).

Runs the research ``reconstruct4d.py`` (its JSON config, with its runtime paths) and this repository's ``reconstruct4d.py``
(the same case without runtime paths) into new output directories, then compares
  * run.json: every per-frame row (states, poses, velocities, statuses, registration / image metadata, mesh revisions)
    except wall-clock timings and output paths;
  * every mesh revision (meshes/*.npz) array by array;
  * with the bag export: every message of the ROS 2 bag (topic, timestamp, every decoded field).

    source /opt/ros/jazzy/setup.bash      # for the bag export / comparison
    python tests/compare_dynamic4d.py --out outputs/compare_4d [--track-only] [--extra-path DIR]
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np

CLEAN = Path(__file__).resolve().parents[1]
TIMING = {"registration_s", "image_s", "fusion_s", "extraction_s", "total_s", "elapsed_s"}

CLEAN_RUNNER = r'''
import sys, json
root, extra = sys.argv[1], json.loads(sys.argv[2])
for p in extra:
    sys.path.append(p)                      # optional runtime packages (e.g. OpenCV), appended after the environment's own
sys.path.insert(0, root)
from extensions.dynamic4d.cli import main
raise SystemExit(main(json.loads(sys.argv[3])))
'''
RESEARCH_RUNNER = r'''
import sys, json
sys.path.insert(0, sys.argv[1])
from extensions.dynamic4d.cli import main
raise SystemExit(main(json.loads(sys.argv[2])))
'''


def scrub(value, out_dir):
    """Drop timings, map output paths to their names."""
    if isinstance(value, dict):
        return {k: scrub(v, out_dir) for k, v in value.items() if k not in TIMING}
    if isinstance(value, list):
        return [scrub(v, out_dir) for v in value]
    if isinstance(value, str) and value.startswith(str(out_dir)):
        return value[len(str(out_dir)):]
    return value


def compare_runs(a_dir: Path, b_dir: Path) -> dict:
    a = scrub(json.loads((a_dir / "run.json").read_text()), a_dir)
    b = scrub(json.loads((b_dir / "run.json").read_text()), b_dir)
    diffs = []
    ra, rb = a.pop("rows"), b.pop("rows")
    if len(ra) != len(rb):
        diffs.append(f"rows {len(ra)} vs {len(rb)}")
    for x, y in zip(ra, rb):
        for k in sorted(set(x) | set(y)):
            if x.get(k) != y.get(k):
                diffs.append(f"frame {x.get('frame')} {k}: {str(x.get(k))[:120]} vs {str(y.get(k))[:120]}")
    for k in sorted(set(a) | set(b)):
        if a.get(k) != b.get(k) and k != "limits":
            diffs.append(f"run {k}: {str(a.get(k))[:160]} vs {str(b.get(k))[:160]}")
    meshes_a = sorted(p.name for p in (a_dir / "meshes").glob("*.npz"))
    meshes_b = sorted(p.name for p in (b_dir / "meshes").glob("*.npz"))
    if meshes_a != meshes_b:
        diffs.append(f"mesh files differ: {len(meshes_a)} vs {len(meshes_b)}")
    for name in sorted(set(meshes_a) & set(meshes_b)):
        with np.load(a_dir / "meshes" / name) as za, np.load(b_dir / "meshes" / name) as zb:
            bad = [k for k in sorted(set(za.files) | set(zb.files)) if k not in za.files or k not in zb.files or not np.array_equal(za[k], zb[k])]
        if bad:
            diffs.append(f"mesh {name}: {bad}")
    return {"rows": [len(ra), len(rb)], "meshes": [len(meshes_a), len(meshes_b)], "differences": diffs}


def compare_bags(a_bag: Path, b_bag: Path, a_dir: Path, b_dir: Path) -> dict:
    """Every message pair decoded and compared with the message classes' field-by-field equality, one pair at a time (CDR
    alignment padding is not initialised by the serializer, so raw bytes may differ for equal messages); status JSON
    without timings and output-directory prefixes."""
    import rosbag2_py
    from rclpy.serialization import deserialize_message
    from rosidl_runtime_py.utilities import get_message

    def open_reader(path):
        r = rosbag2_py.SequentialReader()
        r.open(rosbag2_py.StorageOptions(uri=str(path), storage_id="mcap"), rosbag2_py.ConverterOptions("cdr", "cdr"))
        return r, {t.name: get_message(t.type) for t in r.get_all_topics_and_types()}

    ra, ta = open_reader(a_bag)
    rb, tb = open_reader(b_bag)
    n, diffs, topics = 0, [], set()
    while ra.has_next() and rb.has_next():
        (topic_a, data_a, ns_a), (topic_b, data_b, ns_b) = ra.read_next(), rb.read_next()
        topics.add(topic_a)
        if topic_a != topic_b or ns_a != ns_b:
            diffs.append(f"message {n}: topic {topic_a} / {topic_b}, stamp {ns_a} / {ns_b}")
        else:
            ma, mb = deserialize_message(data_a, ta[topic_a]), deserialize_message(data_b, tb[topic_b])
            if topic_a == "/tef4d/status":
                equal = scrub(json.loads(ma.data), a_dir) == scrub(json.loads(mb.data), b_dir)
            else:
                equal = ma == mb
            if not equal:
                diffs.append(f"message {n}: {topic_a} at {ns_a} differs")
        n += 1
    extra = int(ra.has_next()) + int(rb.has_next())
    if extra:
        diffs.append("message count differs")
    return {"messages": n, "topics": sorted(topics), "differences": diffs[:20], "n_differences": len(diffs)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="new directory for both runs")
    ap.add_argument("--research-root", default=str(Path.home() / "Desktop/TEF"))
    ap.add_argument("--research-config", default=str(Path.home() / "Desktop/TEF/configs/dynamic4d_kitti0059.json"))
    ap.add_argument("--config", default=str(CLEAN / "config/extensions/dynamic4d_kitti0059.json"))
    ap.add_argument("--extra-path", action="append", default=[], help="append to sys.path of the clean run (e.g. an OpenCV site-packages)")
    ap.add_argument("--image-mode", choices=("features", "off"), default="features")
    ap.add_argument("--track-only", action="store_true")
    ap.add_argument("--end", type=int)
    ap.add_argument("--guard-deg", type=float, help="also enable the vehicle heading guard (cone in degrees)")
    a = ap.parse_args()
    out = Path(a.out).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=False)
    common = ["--image-mode", a.image_mode] + (["--track-only"] if a.track_only else []) + (["--end", str(a.end)] if a.end else []) \
        + (["--vehicle-heading-guard-deg", f"{a.guard_deg:g}"] if a.guard_deg else [])
    env = dict(os.environ, OMP_NUM_THREADS="4", PYTHONDONTWRITEBYTECODE="1")
    if not os.environ.get("AMENT_PREFIX_PATH"):           # keep PYTHONPATH only when ROS 2 is sourced (bag export)
        env.pop("PYTHONPATH", None)
    runs = {"research": [sys.executable, "-c", RESEARCH_RUNNER, a.research_root,
                         json.dumps(["--config", a.research_config, "--output", str(out / "research"), *common])],
            "clean": [sys.executable, "-c", CLEAN_RUNNER, str(CLEAN), json.dumps(a.extra_path),
                      json.dumps(["--config", a.config, "--output", str(out / "clean"), *common])]}
    for name, cmd in runs.items():
        with open(out / f"{name}.log", "w") as log:
            if subprocess.call(cmd, stdout=log, stderr=subprocess.STDOUT, env=env):
                raise SystemExit(f"{name} run failed, see {out / (name + '.log')}")
    report = {"track_only": a.track_only, "image_mode": a.image_mode, "run": compare_runs(out / "research", out / "clean")}
    if not a.track_only:
        report["bag"] = compare_bags(out / "research" / "scene_bag", out / "clean" / "scene_bag", out / "research", out / "clean")
    (out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    ok = not report["run"]["differences"] and (a.track_only or report["bag"]["n_differences"] == 0)
    print(f"dynamic4d [{a.image_mode}{', track only' if a.track_only else ', with bag'}]: {'IDENTICAL' if ok else 'DIFFERENT'}; "
          f"rows {report['run']['rows']}, meshes {report['run']['meshes']}"
          + (f", bag messages {report['bag']['messages']}" if "bag" in report else ""))
    for d in report["run"]["differences"][:10] + (report.get("bag", {}).get("differences", [])[:5]):
        print("   ", d)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
