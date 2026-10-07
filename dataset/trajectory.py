"""Trajectory input shared by every dataset.

The canonical text format is TUM-style::

    timestamp tx ty tz qx qy qz qw

FAST-LIVO2 ``Log/result/<sequence>.txt`` already uses this layout.  Translation
is linearly interpolated and rotation is interpolated with quaternion SLERP.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path

import numpy as np


def _normalize_quaternion(value: np.ndarray) -> np.ndarray:
    quaternion = np.asarray(value, dtype=np.float64).reshape(4)
    norm = float(np.linalg.norm(quaternion))
    if not np.isfinite(norm) or norm < 1e-12:
        raise ValueError("quaternion has zero or non-finite norm")
    return quaternion / norm


def _quaternion_to_matrix(value: np.ndarray) -> np.ndarray:
    x, y, z, w = _normalize_quaternion(value)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def _slerp(first: np.ndarray, second: np.ndarray, alpha: float) -> np.ndarray:
    q0 = _normalize_quaternion(first)
    q1 = _normalize_quaternion(second)
    dot = float(np.dot(q0, q1))
    if dot < 0.0:
        q1 = -q1
        dot = -dot
    dot = float(np.clip(dot, -1.0, 1.0))
    if dot > 0.9995:
        return _normalize_quaternion(q0 + alpha * (q1 - q0))
    theta = float(np.arccos(dot))
    sin_theta = float(np.sin(theta))
    return (
        np.sin((1.0 - alpha) * theta) / sin_theta * q0
        + np.sin(alpha * theta) / sin_theta * q1
    )


def _timestamp_to_ns(token: str, unit: str) -> int:
    try:
        value = Decimal(token)
    except InvalidOperation as exc:
        raise ValueError(f"invalid trajectory timestamp {token!r}") from exc
    normalized = unit.lower()
    if normalized in {"s", "sec", "second", "seconds"}:
        value *= Decimal(1_000_000_000)
    elif normalized in {"ns", "nanosecond", "nanoseconds"}:
        pass
    elif normalized == "auto":
        # Epoch/relative seconds are far below a contemporary nanosecond stamp.
        if abs(value) < Decimal("1000000000000"):
            value *= Decimal(1_000_000_000)
    else:
        raise ValueError(f"unsupported timestamp unit {unit!r}")
    return int(value.to_integral_value())


@dataclass(frozen=True, slots=True)
class PoseSample:
    timestamp_ns: int
    world_from_imu: np.ndarray


@dataclass(frozen=True, slots=True)
class Trajectory:
    """Timestamped ``world_from_imu`` states."""

    timestamps_ns: np.ndarray
    positions: np.ndarray
    quaternions_xyzw: np.ndarray

    def __post_init__(self) -> None:
        times = np.asarray(self.timestamps_ns, dtype=np.int64).reshape(-1)
        positions = np.asarray(self.positions, dtype=np.float64)
        quaternions = np.asarray(self.quaternions_xyzw, dtype=np.float64)
        count = len(times)
        if count == 0:
            raise ValueError("trajectory is empty")
        if positions.shape != (count, 3) or quaternions.shape != (count, 4):
            raise ValueError("trajectory arrays have inconsistent shapes")
        if count > 1 and np.any(np.diff(times) <= 0):
            raise ValueError("trajectory timestamps must be strictly increasing")
        if not np.isfinite(positions).all():
            raise ValueError("trajectory positions contain non-finite values")
        quaternions = np.vstack([_normalize_quaternion(value) for value in quaternions])
        object.__setattr__(self, "timestamps_ns", np.ascontiguousarray(times))
        object.__setattr__(self, "positions", np.ascontiguousarray(positions))
        object.__setattr__(self, "quaternions_xyzw", np.ascontiguousarray(quaternions))

    @classmethod
    def load(cls, path: str | Path, timestamp_unit: str = "seconds") -> "Trajectory":
        times: list[int] = []
        positions: list[list[float]] = []
        quaternions: list[list[float]] = []
        with Path(path).open("r", encoding="utf-8") as stream:
            for line_number, raw_line in enumerate(stream, start=1):
                line = raw_line.strip()
                if not line or line.startswith("#"):
                    continue
                columns = line.split()
                if len(columns) != 8:
                    raise ValueError(f"trajectory line {line_number} must contain 8 columns")
                times.append(_timestamp_to_ns(columns[0], timestamp_unit))
                values = [float(value) for value in columns[1:]]
                positions.append(values[:3])
                quaternions.append(values[3:])
        return cls(np.asarray(times), np.asarray(positions), np.asarray(quaternions))

    def nearest_delta_ns(self, timestamp_ns: int) -> int:
        timestamp = int(timestamp_ns)
        index = int(np.searchsorted(self.timestamps_ns, timestamp))
        candidates: list[int] = []
        if index < len(self.timestamps_ns):
            candidates.append(abs(int(self.timestamps_ns[index]) - timestamp))
        if index > 0:
            candidates.append(abs(int(self.timestamps_ns[index - 1]) - timestamp))
        return min(candidates)

    def pose_at(
        self,
        timestamp_ns: int,
        *,
        max_gap_ns: int | None = 200_000_000,
        allow_extrapolation: bool = False,
    ) -> PoseSample:
        timestamp = int(timestamp_ns)
        index = int(np.searchsorted(self.timestamps_ns, timestamp))
        if index < len(self.timestamps_ns) and int(self.timestamps_ns[index]) == timestamp:
            position = self.positions[index]
            quaternion = self.quaternions_xyzw[index]
        elif index == 0 or index == len(self.timestamps_ns):
            if not allow_extrapolation:
                raise ValueError(f"timestamp {timestamp} lies outside the trajectory")
            endpoint = 0 if index == 0 else len(self.timestamps_ns) - 1
            position = self.positions[endpoint]
            quaternion = self.quaternions_xyzw[endpoint]
        else:
            left = index - 1
            right = index
            t0 = int(self.timestamps_ns[left])
            t1 = int(self.timestamps_ns[right])
            gap = t1 - t0
            if max_gap_ns is not None and gap > int(max_gap_ns):
                raise ValueError(f"trajectory gap {gap} ns exceeds {max_gap_ns} ns")
            alpha = (timestamp - t0) / gap
            position = (1.0 - alpha) * self.positions[left] + alpha * self.positions[right]
            quaternion = _slerp(self.quaternions_xyzw[left], self.quaternions_xyzw[right], alpha)
        transform = np.eye(4, dtype=np.float64)
        transform[:3, :3] = _quaternion_to_matrix(quaternion)
        transform[:3, 3] = position
        return PoseSample(timestamp, transform)


# Name kept as an explanatory alias: FAST-LIVO2 is one producer of this format.
FastLivo2Trajectory = Trajectory



def deskew_points_to_scan_frame(points_lidar: np.ndarray, offsets_ns, scan_stamp_ns: int, world_from_imu_at, imu_from_lidar: np.ndarray, *, bin_ns: int = 2_000_000, return_origins: bool = False):
    """``world_from_imu_at(ns) -> (4,4)`` is the trajectory; returns ``(points_scan_frame (N,3), bins_used)`` -- with
    ``return_origins`` also the sensor position at each return's own time, in the same scan frame (N,3), so a ray can be
    cast from where the sensor really was when that return was measured."""

    P = np.asarray(points_lidar, dtype=np.float64)
    if offsets_ns is None or len(P) == 0:
        return (P, 0, np.zeros_like(P)) if return_origins else (P, 0)
    off = np.asarray(offsets_ns, dtype=np.int64).reshape(-1)
    if len(off) != len(P):
        raise ValueError("point_time_offset_ns must align with points_lidar")
    imu_from_lidar = np.asarray(imu_from_lidar, dtype=np.float64)
    T0 = np.asarray(world_from_imu_at(int(scan_stamp_ns)), dtype=np.float64) @ imu_from_lidar
    R0t = T0[:3, :3].T; inv0 = np.eye(4); inv0[:3, :3] = R0t; inv0[:3, 3] = -R0t @ T0[:3, 3]
    bins = off // int(bin_ns)
    out = np.empty_like(P); org = np.empty_like(P) if return_origins else None; uniq = np.unique(bins)
    for b in uniq:
        sel = bins == b
        Tb = np.asarray(world_from_imu_at(int(scan_stamp_ns) + int(np.median(off[sel]))), dtype=np.float64) @ imu_from_lidar
        M = inv0 @ Tb  # lidar(t0) <- lidar(t_b)
        out[sel] = P[sel] @ M[:3, :3].T + M[:3, 3]
        if return_origins:
            org[sel] = M[:3, 3]
    if return_origins:
        return np.ascontiguousarray(out), int(len(uniq)), np.ascontiguousarray(org)
    return np.ascontiguousarray(out), int(len(uniq))


def trajectory_pose_fn(trajectory):
    """Adapter: ``ns -> world_from_imu (4,4)`` from a Trajectory (allowing extrapolation at the scan ends)."""

    def fn(ns: int):
        ps = trajectory.pose_at(int(ns), allow_extrapolation=True)
        for name in ("world_from_imu", "matrix", "transform"):
            if hasattr(ps, name):
                return np.asarray(getattr(ps, name), dtype=np.float64)
        from scipy.spatial.transform import Rotation
        T = np.eye(4); T[:3, :3] = Rotation.from_quat(ps.quaternion_xyzw).as_matrix(); T[:3, 3] = ps.position
        return T
    return fn
