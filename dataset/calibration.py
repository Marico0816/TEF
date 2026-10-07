"""Sensor calibration loading and explicit frame transformations."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np


def _rotation(value: Any, name: str) -> np.ndarray:
    matrix = np.asarray(value, dtype=np.float64)
    if matrix.size != 9:
        raise ValueError(f"{name} must contain 9 values")
    matrix = matrix.reshape(3, 3)
    if not np.isfinite(matrix).all():
        raise ValueError(f"{name} contains non-finite values")
    if not np.allclose(matrix.T @ matrix, np.eye(3), atol=1e-5):
        raise ValueError(f"{name} is not orthonormal")
    if not np.isclose(np.linalg.det(matrix), 1.0, atol=1e-5):
        raise ValueError(f"{name} determinant is not +1")
    return matrix


def transform_from_rotation_translation(rotation: Any, translation: Any, name: str) -> np.ndarray:
    translation_array = np.asarray(translation, dtype=np.float64).reshape(-1)
    if translation_array.size != 3 or not np.isfinite(translation_array).all():
        raise ValueError(f"{name}.translation must contain 3 finite values")
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = _rotation(rotation, f"{name}.rotation")
    transform[:3, 3] = translation_array
    return transform


def invert_transform(transform: np.ndarray) -> np.ndarray:
    matrix = np.asarray(transform, dtype=np.float64)
    rotation = matrix[:3, :3]
    translation = matrix[:3, 3]
    inverse = np.eye(4, dtype=np.float64)
    inverse[:3, :3] = rotation.T
    inverse[:3, 3] = -(rotation.T @ translation)
    return inverse


@dataclass(frozen=True, slots=True)
class Calibration:
    """Transforms follow target_from_source notation."""

    imu_from_lidar: np.ndarray
    camera_from_lidar: np.ndarray | None = None
    camera_intrinsics: np.ndarray | None = None
    distortion: np.ndarray | None = None

    def __post_init__(self) -> None:
        for name in ("imu_from_lidar", "camera_from_lidar"):
            value = getattr(self, name)
            if value is None:
                continue
            matrix = np.asarray(value, dtype=np.float64)
            if matrix.shape != (4, 4):
                raise ValueError(f"{name} must have shape (4, 4)")
            _rotation(matrix[:3, :3], f"{name}.rotation")
            if not np.allclose(matrix[3], [0, 0, 0, 1], atol=1e-9):
                raise ValueError(f"{name} has an invalid last row")
            object.__setattr__(self, name, np.ascontiguousarray(matrix))
        if self.camera_intrinsics is not None:
            intrinsics = np.asarray(self.camera_intrinsics, dtype=np.float64)
            if intrinsics.shape != (3, 3) or not np.isfinite(intrinsics).all():
                raise ValueError("camera_intrinsics must be a finite 3x3 matrix")
            object.__setattr__(self, "camera_intrinsics", intrinsics)
        if self.distortion is not None:
            object.__setattr__(self, "distortion", np.asarray(self.distortion, dtype=np.float64).reshape(-1))

    def world_from_lidar(self, world_from_imu: np.ndarray) -> np.ndarray:
        return np.asarray(world_from_imu, dtype=np.float64) @ self.imu_from_lidar

    def world_from_camera(self, world_from_imu: np.ndarray) -> np.ndarray | None:
        if self.camera_from_lidar is None:
            return None
        world_from_lidar = self.world_from_lidar(world_from_imu)
        return world_from_lidar @ invert_transform(self.camera_from_lidar)


def _read_mapping(path: Path) -> Mapping[str, Any]:
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".json":
        data = json.loads(text)
    else:
        try:
            import yaml
        except ModuleNotFoundError as exc:
            raise RuntimeError("YAML calibration requires PyYAML; JSON is supported without it") from exc
        data = yaml.safe_load(text)
    if not isinstance(data, Mapping):
        raise ValueError("calibration file must contain a mapping")
    return data


def _block_transform(data: Mapping[str, Any], key: str) -> np.ndarray | None:
    block = data.get(key)
    if block is None:
        return None
    if not isinstance(block, Mapping):
        raise ValueError(f"{key} must be a mapping with rotation and translation")
    return transform_from_rotation_translation(block["rotation"], block["translation"], key)


def load_calibration(path: str | Path) -> Calibration:
    """Load the native schema or FAST-LIVO2 ``extrin_calib`` keys."""

    data = _read_mapping(Path(path))
    imu_from_lidar = _block_transform(data, "imu_from_lidar")
    camera_from_lidar = _block_transform(data, "camera_from_lidar")

    extrinsics = data.get("extrin_calib", data)
    if not isinstance(extrinsics, Mapping):
        raise ValueError("extrin_calib must be a mapping")
    if imu_from_lidar is None and "extrinsic_R" in extrinsics and "extrinsic_T" in extrinsics:
        imu_from_lidar = transform_from_rotation_translation(
            extrinsics["extrinsic_R"], extrinsics["extrinsic_T"], "imu_from_lidar"
        )
    if camera_from_lidar is None and "Rcl" in extrinsics and "Pcl" in extrinsics:
        camera_from_lidar = transform_from_rotation_translation(
            extrinsics["Rcl"], extrinsics["Pcl"], "camera_from_lidar"
        )
    if imu_from_lidar is None:
        raise ValueError("missing imu_from_lidar or FAST-LIVO2 extrinsic_R/extrinsic_T")

    camera = data.get("camera", {})
    intrinsics = None
    distortion = None
    if isinstance(camera, Mapping):
        if "intrinsics" in camera:
            intrinsics = np.asarray(camera["intrinsics"], dtype=np.float64).reshape(3, 3)
        elif all(key in camera for key in ("fx", "fy", "cx", "cy")):
            intrinsics = np.array(
                [[camera["fx"], 0.0, camera["cx"]], [0.0, camera["fy"], camera["cy"]], [0.0, 0.0, 1.0]],
                dtype=np.float64,
            )
        if "distortion" in camera:
            distortion = camera["distortion"]
    return Calibration(imu_from_lidar, camera_from_lidar, intrinsics, distortion)
