"""ROS 2 MCAP export of new estimates and causal mesh revisions.

Target and background use actual triangle faces. Background is a labelled local
display crop of saved Gc geometry; it is not recomputed by this extension.
No ground truth, future-pose interpolation, or duplicated target cloud is used.
"""
from __future__ import annotations
import json
import shlex
from collections import Counter
from pathlib import Path
import numpy as np
from .pipeline import digest, save_json
from .visual import matrix, project


def load_mesh(path):
    with np.load(path, allow_pickle=False) as data:
        V, F = np.asarray(data["vertices"], float), np.asarray(data["faces"], np.int64)
    if V.ndim != 2 or V.shape[1] != 3 or not np.isfinite(V).all():
        raise ValueError(f"Invalid mesh vertices: {path}")
    if F.ndim != 2 or F.shape[1] != 3 or not len(F) or F.min() < 0 or F.max() >= len(V):
        raise ValueError(f"Invalid mesh triangles: {path}")
    return V, F


def crop_mesh(V, F, center, radius):
    """Display-only crop; face connectivity of retained triangles is unchanged."""
    inside = np.linalg.norm(V[:, :2] - np.asarray(center)[:2], axis=1) <= radius
    kept = F[inside[F].all(1)]
    if not len(kept):
        return np.empty((0, 3)), np.empty((0, 3), np.int64)
    unique, inverse = np.unique(kept, return_inverse=True)
    return V[unique], inverse.reshape(-1, 3)


def colors(points_world, image, C, K, mask):
    from scipy.ndimage import minimum_filter
    H, W = image.shape[:2]
    uv, depth = project(points_world, C, K)
    ij = np.rint(np.nan_to_num(uv)).astype(np.int64)
    valid = ((depth > .5) & np.isfinite(uv).all(1) & (ij[:, 0] >= 0)
             & (ij[:, 0] < W) & (ij[:, 1] >= 0) & (ij[:, 1] < H))
    ids = np.flatnonzero(valid)
    raster = np.full((H, W), np.inf, np.float32)
    np.minimum.at(raster, (ij[ids, 1], ij[ids, 0]), depth[ids])
    raster = minimum_filter(raster, size=3)
    good = ids[(depth[ids] <= raster[ij[ids, 1], ij[ids, 0]] + .20)
               & mask[ij[ids, 1], ij[ids, 0]]]
    rgb = np.tile(np.array([115, 120, 127], np.uint8), (len(points_world), 1))
    rgb[good] = image[ij[good, 1], ij[good, 0]]
    return rgb


def verify_bag(path, expected):
    import rosbag2_py
    from rclpy.serialization import deserialize_message
    from std_msgs.msg import String
    from visualization_msgs.msg import Marker
    reader = rosbag2_py.SequentialReader()
    reader.open(rosbag2_py.StorageOptions(uri=str(path), storage_id="mcap"),
                rosbag2_py.ConverterOptions("cdr", "cdr"))
    counts, stamps = Counter(), []
    last = -1
    for_check = {r["stamp_ns"]: r for r in expected}
    while reader.has_next():
        topic, data, ns = reader.read_next()
        if ns < last:
            raise ValueError("Nonmonotonic bag storage timestamps")
        last = ns
        counts[topic] += 1
        if topic == "/tef4d/status":
            row = json.loads(deserialize_message(data, String).data)
            if row["frame"] != for_check[ns]["frame"]:
                raise ValueError("Bag status does not match the new run")
            stamps.append(ns)
        elif topic == "/tef4d/target_mesh":
            marker = deserialize_message(data, Marker)
            if marker.type != Marker.TRIANGLE_LIST or marker.id != 0 or marker.ns != "target":
                raise ValueError("Unexpected target representation or duplicate ID")
            if len(marker.points) % 3 or len(marker.colors) != len(marker.points):
                raise ValueError("Invalid target triangle/color arrays")
    if stamps != [r["stamp_ns"] for r in expected]:
        raise ValueError("Missing or duplicated frame statuses")
    if counts["/tef4d/target_mesh"] != len(expected):
        raise ValueError("Expected exactly one target mesh marker per input frame")
    if "/tef4d/objects" in counts:
        raise ValueError("Duplicate object-cloud rendering is not allowed")
    return dict(counts=counts, original_timestamps=len(stamps), target_marker_unique=True)


def export_bag(config, run, output):
    import rosbag2_py
    from builtin_interfaces.msg import Time
    from geometry_msgs.msg import Point, TransformStamped
    from rclpy.serialization import serialize_message
    from scipy.spatial.transform import Rotation
    from sensor_msgs.msg import Image as ImageMessage
    from std_msgs.msg import ColorRGBA, String
    from tf2_msgs.msg import TFMessage
    from visualization_msgs.msg import Marker, MarkerArray
    from PIL import Image
    output = Path(output)
    path = output / "scene_bag"
    if path.exists():
        raise FileExistsError(path)
    manifest = json.loads(Path(config["scene_manifest"]).read_text())
    context = {r["frame"]: r for r in manifest["slices"]}
    data_root = Path(run["case"]["dataset"])
    records = [json.loads(x) for x in (data_root / "manifest.jsonl").read_text().splitlines()]
    writer = rosbag2_py.SequentialWriter()
    writer.open(rosbag2_py.StorageOptions(uri=str(path), storage_id="mcap"),
                rosbag2_py.ConverterOptions("cdr", "cdr"))
    topics = [
        ("/tef4d/target_mesh", "visualization_msgs/msg/Marker"),
        ("/tef4d/background_mesh", "visualization_msgs/msg/Marker"),
        ("/tef4d/trajectory", "visualization_msgs/msg/MarkerArray"),
        ("/tef4d/label", "visualization_msgs/msg/Marker"),
        ("/tef4d/status", "std_msgs/msg/String"),
        ("/tef4d/image", "sensor_msgs/msg/Image"),
        ("/tf", "tf2_msgs/msg/TFMessage"),
    ]
    for i, (name, typ) in enumerate(topics):
        writer.create_topic(rosbag2_py.TopicMetadata(id=i, name=name, type=typ, serialization_format="cdr"))

    def stamp(ns):
        return Time(sec=int(ns // 1_000_000_000), nanosec=int(ns % 1_000_000_000))

    def emit(topic, msg, ns):
        writer.write(topic, serialize_message(msg), int(ns))

    def triangles(V, F, rgb, ns, namespace, pose=None):
        marker = Marker()
        marker.header.frame_id = "map"
        marker.header.stamp = stamp(ns)
        marker.ns, marker.id = namespace, 0
        marker.type, marker.action = Marker.TRIANGLE_LIST, Marker.ADD
        marker.pose.orientation.w = 1.
        marker.scale.x = marker.scale.y = marker.scale.z = 1.
        marker.color = ColorRGBA(r=1., g=1., b=1., a=1.)
        if pose is not None:
            T = np.asarray(pose)
            marker.pose.position.x, marker.pose.position.y, marker.pose.position.z = map(float, T[:3, 3])
            q = Rotation.from_matrix(T[:3, :3]).as_quat()
            marker.pose.orientation.x, marker.pose.orientation.y, marker.pose.orientation.z, marker.pose.orientation.w = map(float, q)
        points = [Point(x=float(v[0]), y=float(v[1]), z=float(v[2])) for v in V]
        palette, inverse = np.unique(rgb, axis=0, return_inverse=True)
        entries = [ColorRGBA(r=float(c[0])/255, g=float(c[1])/255, b=float(c[2])/255, a=1.) for c in palette]
        marker.points = [points[int(i)] for i in F.reshape(-1)]
        marker.colors = [entries[int(inverse[i])] for i in F.reshape(-1)]
        return marker

    previous_shape, V, F = None, None, None
    background_key, background_marker = None, None
    previous_mesh_source, background_raw = None, None
    shape_checks, bg_faces = 0, []
    hashes = {}
    radius = float(config.get("background_radius_m", 10))
    for index, row in enumerate(run["rows"]):
        f, ns = row["frame"], row["stamp_ns"]
        revision = row["shape_revision"]
        if not revision["source_frame"] <= revision["available_frame"] <= f:
            raise ValueError("Future mesh revision leaked into playback")
        if previous_shape != revision["path"]:
            V, F = load_mesh(revision["path"])
            hashes[revision["path"]] = digest(revision["path"])
            previous_shape = revision["path"]
        cache_path = Path(config["image_cache"]) / f"{f:04d}.npz"
        rgb = np.tile(np.array([115, 120, 127], np.uint8), (len(V), 1))
        camera = K = vehicle = image = None
        rec = records[f]
        if cache_path.exists() and rec.get("image_path"):
            with np.load(cache_path, allow_pickle=False) as cache:
                if (int(cache["lidar_timestamp_ns"]) != ns
                        or int(cache["image_timestamp_ns"]) != int(rec["image_timestamp_ns"])):
                    raise ValueError("Color-image timestamp mismatch")
                camera, K, vehicle = cache["camera"], cache["K"], cache["vehicle"].astype(bool)
                dt = float(cache["image_dt"])
            image_path = data_root / rec["image_path"]
            image = np.asarray(Image.open(image_path).convert("RGB"))
            if image.shape[:2] != vehicle.shape:
                raise ValueError("Image/semantic mask shape mismatch")
            hashes[str(cache_path)] = digest(cache_path)
            hashes[str(image_path)] = digest(image_path)
            with np.load(config["initial_model"], allow_pickle=False) as model:
                ref = model["reference"]
            T = matrix(np.asarray(row["state"]) + np.asarray(row["velocity"]) * dt, ref)
            rgb = colors(V @ T[:3, :3].T + T[:3, 3], image, camera, K, vehicle)
            message = ImageMessage()
            message.header.frame_id, message.header.stamp = "camera", stamp(int(rec["image_timestamp_ns"]))
            message.height, message.width = image.shape[:2]
            message.encoding, message.step = "rgb8", image.shape[1] * 3
            message.data = image.tobytes()
            emit("/tef4d/image", message, ns)
        emit("/tef4d/target_mesh", triangles(V, F, rgb, ns, "target", row["pose"]), ns)
        shape_checks += 1
        # Context is explicitly background only. Old target/other-object parts
        # are never appended to the new target representation.
        background = [p for p in context[f]["parts"] if p["role"] == "background"]
        if len(background) != 1:
            raise ValueError("Expected exactly one saved background revision")
        part = background[0]
        if int(part["revision"]["available_frame"]) > f:
            raise ValueError("Future background revision")
        key = (part["path"], f // 10)
        background_changed = key != background_key
        if background_changed:
            if previous_mesh_source != part["path"]:
                background_raw = load_mesh(part["path"])
                hashes[part["path"]] = digest(part["path"])
                previous_mesh_source = part["path"]
            BV, BF = crop_mesh(*background_raw, np.asarray(row["pose"])[:3, 3], radius)
            bc = (colors(BV, image, camera, K, ~vehicle) if image is not None else
                  np.tile(np.array([115, 120, 127], np.uint8), (len(BV), 1)))
            background_marker = triangles(BV, BF, bc, ns, "background")
            if not len(BF):
                background_marker.action = Marker.DELETE
            background_key = key
            bg_faces.append(dict(frame=f, faces=len(BF), radius_m=radius, source=part["path"]))
        if background_changed:
            background_marker.header.stamp = stamp(ns)
            emit("/tef4d/background_mesh", background_marker, ns)
        markers = []
        for accepted in (True, False):
            m = Marker()
            m.header.frame_id, m.header.stamp = "map", stamp(ns)
            m.ns, m.id = "trajectory", int(accepted)
            m.type, m.action = Marker.LINE_LIST, Marker.ADD
            m.pose.orientation.w, m.scale.x = 1., .05
            m.color = ColorRGBA(r=.1 if accepted else 1., g=.9 if accepted else .45, b=.25 if accepted else .05, a=1.)
            for j in range(1, index + 1):
                pair = run["rows"][j-1:j+1]
                if all(p["accepted"] for p in pair) != accepted:
                    continue
                for p in pair:
                    xyz = np.asarray(p["pose"])[:3, 3]
                    m.points.append(Point(x=float(xyz[0]), y=float(xyz[1]), z=float(xyz[2])))
            markers.append(m)
        emit("/tef4d/trajectory", MarkerArray(markers=markers), ns)
        label = Marker()
        label.header.frame_id, label.header.stamp = "map", stamp(ns)
        label.ns, label.id, label.type, label.action = "status", 0, Marker.TEXT_VIEW_FACING, Marker.ADD
        xyz = np.asarray(row["pose"])[:3, 3]
        label.pose.position.x, label.pose.position.y, label.pose.position.z = float(xyz[0]), float(xyz[1]), float(xyz[2] + 2)
        label.pose.orientation.w, label.scale.z = 1., .22
        label.color = ColorRGBA(r=1., g=1., b=1., a=1.)
        label.text = (f"TEF experimental 4D | frame {f} | {row['status']}\n"
                      f"Image: {row['image']['reason']} | Adam steps {row['image']['steps']}\n"
                      f"New target mesh; saved Gc background crop ({radius:g} m)\n"
                      "Input-time estimates; NOT a real-time benchmark")
        emit("/tef4d/label", label, ns)
        tf = TransformStamped()
        tf.header.frame_id, tf.header.stamp, tf.child_frame_id = "map", stamp(ns), "inspection_view"
        tf.transform.translation.x, tf.transform.translation.y, tf.transform.translation.z = map(float, xyz)
        tf.transform.rotation.w = 1.
        emit("/tf", TFMessage(transforms=[tf]), ns)
        emit("/tef4d/status", String(data=json.dumps(row)), ns)
        if f % 10 == 9:
            print(f"bag frame {f}: target {len(F)} faces, background {len(background_marker.points)//3} faces", flush=True)
    del writer
    verified = verify_bag(path, run["rows"])
    if any(digest(p) != h for p, h in hashes.items()):
        raise ValueError("A source mesh or image changed during export")
    audit = dict(bag=str(path), storage="mcap", target_triangle_messages=shape_checks,
                 gt_used=False, future_pose_interpolation=False, background="saved Gc cropped triangle mesh",
                 other_objects="not rerun or separately displayed", background_crops=bg_faces,
                 source_sha256=hashes, verification=verified)
    save_json(output / "bag_audit.json", audit)
    template = Path(__file__).with_name("view.rviz")
    (output / "view.rviz").write_text(template.read_text())
    launcher = """#!/usr/bin/env bash
set -eo pipefail
dynamic4d_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source /opt/ros/jazzy/setup.bash
set -u
export ROS_DOMAIN_ID=141
export ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST
export ROS_LOG_DIR="${dynamic4d_dir}/ros_logs"
mkdir -p "${ROS_LOG_DIR}"
rviz2 -d "${dynamic4d_dir}/view.rviz" --ros-args -p use_sim_time:=true &
dynamic4d_rviz_pid=$!
trap 'kill "${dynamic4d_rviz_pid}" 2>/dev/null || true' EXIT INT TERM
ros2 bag play "${dynamic4d_dir}/scene_bag" --clock-topics-all --loop --delay 3 --rate 1.0 --read-ahead-queue-size 32
"""
    (output / "play.sh").write_text(launcher)
    print(f"Verified ROS 2 bag: {path}\nPlay: bash {shlex.quote(str(output / 'play.sh'))}", flush=True)
