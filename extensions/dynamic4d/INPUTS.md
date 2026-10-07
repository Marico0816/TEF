# Dynamic objects (4D): inputs, outputs, dependencies

This optional extension tracks one moving target through a case interval on the given sensor poses. For every frame it:

1. predicts the target's 4-DoF state `[x, y, z, yaw]`;
2. registers the target's returns to the object mesh;
3. optionally refines the pose with image features: 30 Adam steps, accepted only after finite-value, pixel-support, loss-decrease and geometry checks;
4. fuses accepted observations into an object-frame TSDF (3 cm voxels) and extracts the object mesh every `mesh_every` frames;
5. exports everything as a ROS 2 bag for RViz.

The static mapper (`tef_mapping.py`) is not involved.

The front end that produces the target's returns is **not** part of this repository: object detection or segmentation, tracking, and per-return ownership. Its outputs are the inputs below. The code checks every input against the canonical dataset: frame index, timestamps, point order, camera pose and intrinsics.

## Run

```bash
python reconstruct4d.py --config CONFIG.json --output NEW_DIR [--image-mode features|off] [--track-only] [--end FRAME]
                        [--vehicle-heading-guard-deg 25]
```

- `--track-only` skips the bag export.
- `--image-mode off` skips the image refinement and needs no OpenCV.
- `--vehicle-heading-guard-deg C` is an optional safeguard for vehicle targets. It rejects headings outside a cone of C degrees around the reliable travel direction and is off by default.
- The output directory must not exist.
- `bash NEW_DIR/play.sh` plays the bag in RViz (ROS 2 Jazzy).

## The configuration (`config/extensions/dynamic4d_*.json`)

| Key | Meaning |
|---|---|
| `scenes_json` | Case description: JSON whose `case` holds `dataset` (the canonical sequence, with images and camera calibration), `tracks` (front-end tracks, below), `track_id`, `initial` (frame of the initial model), `start`, `end` (exclusive) and `prefix_rows` (track rows that built the initial model) |
| `initial_model` | `.npz`: `vertices`, `faces`, `reference` (3×3 object-frame rotation), `initial_state` `[x, y, z, yaw]`, `initial_velocity` (4), `initial_time` (s since the dataset's first frame), `initial_frame` |
| `partitions` | Directory of `NNNN.npz`, one per frame. `cluster_id` (N,) gives the owner of every return of the frame, in the dataset's point order: the target where it equals `track_id`, ≥ −2 otherwise. Also holds `frame`, `stamp_ns`, `point_sha256` (SHA-256 of `str(points.shape)` + the float64 points) and `source_kind = "full_return_frontend"`. |
| `image_cache` | Directory of `NNNN.npz` (a missing frame means no image). Holds `gray` (H×W uint8), `vehicle` (H×W bool target mask), `camera` (camera_from_world 4×4), `K` (3×3), `image_dt` (image − LiDAR time, s), `lidar_timestamp_ns`, `image_timestamp_ns`. |
| `scene_manifest` | Background for display only. `slices` is a list of `{frame, parts: [{role: "background", path: mesh .npz (vertices, faces), revision: {available_frame}}]}`. A part is never used before its `available_frame`. |
| `mesh_every` | Object mesh extraction interval in frames (default 5) |
| `background_radius_m` | Display crop of the background around the target (default 10 m) |

**Front-end tracks** (`case.tracks`, `.npz`) have one row per (track, frame):

- `track_id`, `frame`, `stamp_ns`, `observed`;
- `object_from_lidar` (R×4×4);
- `point_offsets` (R+1) into `points_lidar` (P×3);
- `track_class` (T×2: track id, class id).

The guard uses Cityscapes class ids 13 / 14 / 15 (car, truck, bus).

## Outputs

| File | Content |
|---|---|
| `run.json` | Every frame: prediction, state, pose, velocity, status (`lidar`, `lidar_image`, `image_only`, `prediction`, `heldout_prediction`, …), registration and image metadata, fused or not, the mesh revision in use, timings; plus input SHA-256 hashes |
| `meshes/*.npz` | The object meshes (object frame) as they became available (`initial.npz`, `target_fNNNN.npz`) |
| `scene_bag/` | ROS 2 MCAP bag with these topics: `/tef4d/target_mesh`, `/tef4d/background_mesh`, `/tef4d/trajectory`, `/tef4d/label`, `/tef4d/status`, `/tef4d/image`, `/tf` |
| `play.sh`, `view.rviz` | Playback with RViz |
| `COMPLETE.json` / `FAILED.json` | Completion record |

Held-out frames (index % 10 == 5) are never fused; they only receive predictions.

## Dependencies

- **Core:** NumPy, SciPy, PyTorch, Open3D (the repository requirements).
- **`--image-mode features`:** OpenCV (`opencv-python`).
- **Bag export:** Pillow, and ROS 2 Jazzy's Python packages (`rosbag2_py`, `rclpy`, `visualization_msgs`, `sensor_msgs`, `tf2_msgs`) with the MCAP storage plugin. Source `/opt/ros/jazzy/setup.bash` first.

The reconstruction runs on the CPU.

## Example case and equivalence

`config/extensions/dynamic4d_kitti0059.json` is the frozen KITTI 0059 case, target 51, frames 180–279, with placeholder inputs under `data/kitti0059_t51/`. Replace these paths with your prepared inputs before running. Relative paths are resolved from the current working directory (run from the repository root); this repository does not download or produce the required front-end caches. On that case `tests/compare_dynamic4d.py` finds this repository identical to the research implementation:

- every `run.json` row (100) and every mesh revision (22);
- all 621 bag messages, decoded field by field;
- frame rows and mesh revisions also match with images off and with the heading guard at 25°; those two checks used `--track-only`. Timing and output-path fields are excluded from the comparison.
