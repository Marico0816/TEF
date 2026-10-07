"""High-level loader that assembles synchronized :class:`FramePacket` values."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .calibration import Calibration, load_calibration
from .data_reader import FrameRecord, read_image, read_manifest, read_point_cloud
from .trajectory import Trajectory


@dataclass(frozen=True, slots=True)
class FramePacket:
    frame_id: str
    lidar_timestamp_raw_ns: int
    lidar_timestamp_ns: int
    image_timestamp_raw_ns: int | None
    image_timestamp_ns: int | None
    points_lidar: np.ndarray
    image_rgb: np.ndarray | None
    world_from_imu: np.ndarray
    world_from_lidar: np.ndarray
    world_from_camera: np.ndarray | None
    lidar_image_delta_ns: int | None
    pose_lidar_nearest_delta_ns: int
    intensity: np.ndarray | None = None
    point_time_offset_ns: np.ndarray | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        points = np.asarray(self.points_lidar, dtype=np.float64)
        if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
            raise ValueError("points_lidar must be a finite array with shape (N, 3)")
        object.__setattr__(self, "points_lidar", np.ascontiguousarray(points))
        for name in ("world_from_imu", "world_from_lidar", "world_from_camera"):
            value = getattr(self, name)
            if value is None:
                continue
            matrix = np.asarray(value, dtype=np.float64)
            if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
                raise ValueError(f"{name} must be a finite 4x4 matrix")
            if not np.allclose(matrix[3], [0, 0, 0, 1], atol=1e-9):
                raise ValueError(f"{name} has an invalid homogeneous last row")
            object.__setattr__(self, name, np.ascontiguousarray(matrix))
        if self.image_rgb is not None:
            image = np.asarray(self.image_rgb)
            if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
                raise ValueError("image_rgb must have shape (H, W, 3) and dtype uint8")
            object.__setattr__(self, "image_rgb", np.ascontiguousarray(image))
        for name in ("intensity", "point_time_offset_ns"):
            value = getattr(self, name)
            if value is None:
                continue
            array = np.asarray(value).reshape(-1)
            if len(array) != len(points):
                raise ValueError(f"{name} length must equal the point count")
            object.__setattr__(self, name, np.ascontiguousarray(array))


class DatasetLoader:
    """Lazy loader for any sequence converted to the canonical GUB format."""

    def __init__(
        self,
        root: str | Path,
        *,
        manifest: str = "manifest.jsonl",
        trajectory: str = "trajectory.txt",
        calibration: str = "calibration.yaml",
        trajectory_timestamp_unit: str = "seconds",
        lidar_time_offset_ns: int = 0,
        image_time_offset_ns: int = 0,
        max_image_delta_ns: int | None = 30_000_000,
        max_pose_gap_ns: int | None = 200_000_000,
        allow_pose_extrapolation: bool = False,
        min_range_m: float = 0.2,
        max_range_m: float | None = None,
        load_images: bool = True,
    ) -> None:
        self.root = Path(root).resolve()
        if not self.root.is_dir():
            raise FileNotFoundError(self.root)
        self.records: list[FrameRecord] = read_manifest(self.root / manifest)
        self.trajectory = Trajectory.load(self.root / trajectory, trajectory_timestamp_unit)
        calibration_path = self.root / calibration
        if not calibration_path.exists() and calibration == "calibration.yaml":
            fallback = self.root / "calibration.json"
            if fallback.exists():
                calibration_path = fallback
        self.calibration: Calibration = load_calibration(calibration_path)
        self.lidar_time_offset_ns = int(lidar_time_offset_ns)
        self.image_time_offset_ns = int(image_time_offset_ns)
        self.max_image_delta_ns = max_image_delta_ns
        self.max_pose_gap_ns = max_pose_gap_ns
        self.allow_pose_extrapolation = bool(allow_pose_extrapolation)
        self.min_range_m = float(min_range_m)
        self.max_range_m = None if max_range_m is None else float(max_range_m)
        self.load_images = bool(load_images)

    def __len__(self) -> int:
        return len(self.records)

    def _resolve(self, relative_path: str) -> Path:
        value = Path(relative_path)
        if value.is_absolute():
            raise ValueError("manifest paths must be relative to the sequence root")
        resolved = (self.root / value).resolve()
        try:
            resolved.relative_to(self.root)
        except ValueError as exc:
            raise ValueError(f"manifest path escapes the sequence root: {relative_path}") from exc
        return resolved

    def __getitem__(self, index: int) -> FramePacket:
        record = self.records[index]
        lidar_raw_time = int(record.lidar_timestamp_ns)
        lidar_time = lidar_raw_time + self.lidar_time_offset_ns
        image_raw_time = None if record.image_timestamp_ns is None else int(record.image_timestamp_ns)
        image_time = None if image_raw_time is None else image_raw_time + self.image_time_offset_ns

        cloud = read_point_cloud(
            self._resolve(record.lidar_path),
            min_range_m=self.min_range_m,
            max_range_m=self.max_range_m,
            point_time_field=record.point_time_field,
            point_time_unit=record.point_time_unit,
        )
        image_rgb = None
        if self.load_images and record.image_path is not None:
            image_rgb = read_image(self._resolve(record.image_path))

        image_delta = None if image_time is None else image_time - lidar_time
        if image_delta is not None and self.max_image_delta_ns is not None:
            if abs(image_delta) > int(self.max_image_delta_ns):
                raise ValueError(
                    f"frame {record.frame_id}: LiDAR-image delta {image_delta} ns exceeds {self.max_image_delta_ns} ns"
                )

        lidar_pose = self.trajectory.pose_at(
            lidar_time,
            max_gap_ns=self.max_pose_gap_ns,
            allow_extrapolation=self.allow_pose_extrapolation,
        )
        world_from_lidar = self.calibration.world_from_lidar(lidar_pose.world_from_imu)
        world_from_camera = None
        if image_time is not None and self.calibration.camera_from_lidar is not None:
            image_pose = self.trajectory.pose_at(
                image_time,
                max_gap_ns=self.max_pose_gap_ns,
                allow_extrapolation=self.allow_pose_extrapolation,
            )
            world_from_camera = self.calibration.world_from_camera(image_pose.world_from_imu)

        return FramePacket(
            frame_id=str(record.frame_id),
            lidar_timestamp_raw_ns=lidar_raw_time,
            lidar_timestamp_ns=lidar_time,
            image_timestamp_raw_ns=image_raw_time,
            image_timestamp_ns=image_time,
            points_lidar=cloud.xyz,
            image_rgb=image_rgb,
            world_from_imu=lidar_pose.world_from_imu,
            world_from_lidar=world_from_lidar,
            world_from_camera=world_from_camera,
            lidar_image_delta_ns=image_delta,
            pose_lidar_nearest_delta_ns=self.trajectory.nearest_delta_ns(lidar_time),
            intensity=cloud.intensity,
            point_time_offset_ns=cloud.point_time_offset_ns,
            metadata=dict(record.metadata),
        )


# Short alias for training/mapping call sites.
Dataset = DatasetLoader
