"""Low-level readers for the canonical GUB dataset format.

This module only turns files into typed in-memory values.  It does not perform
pose interpolation, sensor synchronization, or world-frame transformations;
those belong to :mod:`dataset_loader`.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, BinaryIO

import numpy as np


@dataclass(frozen=True, slots=True)
class FrameRecord:
    """One row of ``manifest.jsonl``."""

    frame_id: str
    lidar_path: str
    lidar_timestamp_ns: int
    image_path: str | None = None
    image_timestamp_ns: int | None = None
    point_time_field: str | None = None
    point_time_unit: str = "nanoseconds"
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class PointCloudData:
    xyz: np.ndarray
    intensity: np.ndarray | None
    point_time_offset_ns: np.ndarray | None


_PCD_DTYPES = {
    ("F", 4): "<f4", ("F", 8): "<f8",
    ("I", 1): "i1", ("I", 2): "<i2", ("I", 4): "<i4", ("I", 8): "<i8",
    ("U", 1): "u1", ("U", 2): "<u2", ("U", 4): "<u4", ("U", 8): "<u8",
}


def _read_pcd_header(stream: BinaryIO) -> dict[str, list[str]]:
    header: dict[str, list[str]] = {}
    while True:
        raw = stream.readline()
        if not raw:
            raise ValueError("PCD header ended before DATA")
        line = raw.decode("ascii").strip()
        if not line or line.startswith("#"):
            continue
        key, *values = line.split()
        header[key.upper()] = values
        if key.upper() == "DATA":
            return header


def _read_pcd_columns(path: Path) -> dict[str, np.ndarray]:
    with path.open("rb") as stream:
        header = _read_pcd_header(stream)
        fields = header.get("FIELDS", header.get("FIELD"))
        if fields is None:
            raise ValueError(f"PCD has no FIELDS: {path}")
        sizes = [int(value) for value in header["SIZE"]]
        types = [value.upper() for value in header["TYPE"]]
        counts = [int(value) for value in header.get("COUNT", ["1"] * len(fields))]
        if not (len(fields) == len(sizes) == len(types) == len(counts)):
            raise ValueError(f"inconsistent PCD field metadata: {path}")
        point_count = int(header.get("POINTS", header.get("WIDTH", ["0"]))[0])
        data_kind = header["DATA"][0].lower()
        if data_kind == "binary":
            layout = []
            for name, size, code, count in zip(fields, sizes, types, counts):
                try:
                    scalar = _PCD_DTYPES[(code, size)]
                except KeyError as exc:
                    raise ValueError(f"unsupported PCD scalar type {code}{size}") from exc
                layout.append((name, scalar) if count == 1 else (name, scalar, (count,)))
            records = np.fromfile(stream, dtype=np.dtype(layout), count=point_count)
            if len(records) != point_count:
                raise ValueError(f"PCD contains {len(records)} records, expected {point_count}")
            return {name: np.asarray(records[name]) for name in fields}
        if data_kind == "ascii":
            values = np.loadtxt(stream, dtype=np.float64, ndmin=2)
            if len(values) != point_count:
                raise ValueError(f"PCD contains {len(values)} rows, expected {point_count}")
            if values.shape[1] != sum(counts):
                raise ValueError(f"PCD has {values.shape[1]} columns, expected {sum(counts)}")
            result: dict[str, np.ndarray] = {}
            offset = 0
            for name, count in zip(fields, counts):
                block = values[:, offset : offset + count]
                result[name] = block[:, 0] if count == 1 else block
                offset += count
            return result
        if data_kind == "binary_compressed":
            raise ValueError("binary_compressed PCD is not supported yet")
        raise ValueError(f"unsupported PCD DATA mode {data_kind!r}")


def read_point_cloud(
    path: str | Path,
    *,
    min_range_m: float = 0.2,
    max_range_m: float | None = None,
    point_time_field: str | None = None,
    point_time_unit: str = "nanoseconds",
) -> PointCloudData:
    """Read ASCII/binary PCD and preserve optional intensity/point time."""

    columns = _read_pcd_columns(Path(path))
    missing = [name for name in ("x", "y", "z") if name not in columns]
    if missing:
        raise ValueError(f"PCD is missing fields {missing}: {path}")
    xyz = np.column_stack([columns["x"], columns["y"], columns["z"]]).astype(np.float64)
    ranges = np.linalg.norm(xyz, axis=1)
    valid = np.isfinite(xyz).all(axis=1) & (ranges >= float(min_range_m))
    if max_range_m is not None:
        valid &= ranges <= float(max_range_m)
    if not np.any(valid):
        raise ValueError(f"no valid points remain after filtering: {path}")

    intensity = None
    for name in ("intensity", "reflectivity"):
        if name in columns:
            intensity = np.asarray(columns[name])[valid]
            break

    if point_time_field is None and "offset_time" in columns:
        point_time_field = "offset_time"
    point_offsets = None
    if point_time_field is not None:
        if point_time_field not in columns:
            raise ValueError(f"point time field {point_time_field!r} is absent from {path}")
        scales = {
            "seconds": 1e9,
            "milliseconds": 1e6,
            "microseconds": 1e3,
            "nanoseconds": 1.0,
        }
        try:
            scale = scales[point_time_unit.lower()]
        except KeyError as exc:
            raise ValueError(f"unsupported point_time_unit {point_time_unit!r}") from exc
        point_offsets = np.rint(np.asarray(columns[point_time_field], dtype=np.float64)[valid] * scale).astype(np.int64)
    return PointCloudData(np.ascontiguousarray(xyz[valid]), intensity, point_offsets)


def read_image(path: str | Path) -> np.ndarray:
    """Decode JPG/PNG into contiguous uint8 RGB."""

    try:
        from PIL import Image
    except ModuleNotFoundError as exc:
        raise RuntimeError("reading JPG/PNG images requires Pillow") from exc
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"), dtype=np.uint8).copy()


def read_manifest(path: str | Path) -> list[FrameRecord]:
    """Read the canonical JSON-lines frame index."""

    records: list[FrameRecord] = []
    with Path(path).open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                records.append(FrameRecord(**json.loads(line)))
            except (json.JSONDecodeError, TypeError, ValueError) as exc:
                raise ValueError(f"invalid manifest row {line_number}: {path}") from exc
    if not records:
        raise ValueError(f"manifest is empty: {path}")
    times = np.asarray([record.lidar_timestamp_ns for record in records], dtype=np.int64)
    if len(times) > 1 and np.any(np.diff(times) <= 0):
        raise ValueError("manifest LiDAR timestamps must be strictly increasing")
    return records
