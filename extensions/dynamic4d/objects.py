"""Object tracking primitives of the dynamic extension, vendored from ``dynobj`` (unchanged arithmetic):
4-DoF point-to-plane registration against the object's mesh, the cached front-end tracks, the initial object model and
the rigid-motion helpers.  The object state is [x, y, z, yaw] on top of a fixed reference rotation R = rz(yaw) @ reference."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np


# -- rigid-motion helpers ----------------------------------------------------------------------------------------------
def rz(yaw: float) -> np.ndarray:
    """Rotation about z."""
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def deterministic_points(points: np.ndarray, maximum: int = 1024) -> np.ndarray:
    """Evenly spaced subsample in storage order; identity below the cap."""
    if len(points) <= maximum:
        return points
    return points[np.linspace(0, len(points) - 1, maximum).astype(int)]


# -- inputs --------------------------------------------------------------------------------------------------------------
class CachedTracks:
    """Front-end tracks (``*_tracks.npz``): one row per (track, frame) with the returns the front end assigned to it.
    Arrays: ``frame``, ``track_id``, ``observed``, ``stamp_ns``, ``object_from_lidar`` (R,4,4), ``point_offsets`` (R+1),
    ``points_lidar`` (P,3)."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.z = dict(np.load(self.path))

    def row_points(self, j: int) -> np.ndarray:
        a, e = self.z["point_offsets"][j:j + 2]
        return np.asarray(self.z["points_lidar"][a:e], np.float64)

    def stamp_ns(self, j: int) -> int:
        return int(self.z["stamp_ns"][j])

    def object_from_lidar(self, j: int) -> np.ndarray:
        return np.asarray(self.z["object_from_lidar"][j], np.float64)


@dataclass
class InitialModel:
    """The object's mesh and motion state before the reconstructed interval (``model.npz``: ``vertices``, ``faces``,
    ``reference`` (3,3), ``initial_state`` [x,y,z,yaw], ``initial_velocity`` (4), ``initial_time`` s, ``initial_frame``)."""
    vertices: np.ndarray
    faces: np.ndarray
    reference: np.ndarray
    state: np.ndarray
    velocity: np.ndarray
    time: float
    frame: int

    @classmethod
    def load(cls, path: str | Path) -> "InitialModel":
        m = dict(np.load(path))
        return cls(vertices=np.asarray(m["vertices"], float), faces=np.asarray(m["faces"]), reference=np.asarray(m["reference"], float),
                   state=np.asarray(m["initial_state"], float), velocity=np.asarray(m["initial_velocity"], float),
                   time=float(m["initial_time"]), frame=int(m["initial_frame"]))


# -- registration --------------------------------------------------------------------------------------------------------
@dataclass
class RegistrationConfig:
    min_points: int = 30
    min_pairs: int = 30
    iterations: int = 8
    correspondence_max_m: float = 0.5
    huber_m: float = 0.05
    step_clip_m: float = 0.8
    step_clip_deg: float = 12.0
    rcond: float = 1e-4
    subsample: int = 1024


@dataclass
class RegistrationResult:
    accepted: bool = False
    reason: str = "insufficient_points"
    pairs: int = 0
    rmse_before: float | None = None
    rmse_after: float | None = None

    def as_dict(self) -> dict:
        return asdict(self)


class PointToPlaneRegistration:
    """4-DoF IRLS point-to-plane registration of world points against the fixed object mesh (object frame)."""

    def __init__(self, vertices: np.ndarray, faces: np.ndarray, reference: np.ndarray, cfg: RegistrationConfig | None = None):
        import open3d as o3d
        self.o3d = o3d
        self.cfg = cfg or RegistrationConfig()
        self.reference = np.asarray(reference, np.float64)
        self.scene = o3d.t.geometry.RaycastingScene(nthreads=4)
        self.scene.add_triangles(o3d.core.Tensor(np.asarray(vertices, np.float32)), o3d.core.Tensor(np.asarray(faces, np.uint32)))

    def correspond(self, state: np.ndarray, points: np.ndarray):
        """Closest mesh points (object frame) and normals for world points under ``state``."""
        R = rz(state[3]) @ self.reference
        q = (points - state[:3]) @ R
        got = self.scene.compute_closest_points(self.o3d.core.Tensor(q.astype(np.float32)), nthreads=4)
        closest = got["points"].numpy().astype(float)
        normal = got["primitive_normals"].numpy().astype(float)
        return closest, normal, np.linalg.norm(q - closest, axis=1)

    def rmse(self, state: np.ndarray, points: np.ndarray) -> float:
        if not len(points):
            return float("inf")
        _, _, dist = self.correspond(state, points)
        return float(np.sqrt(np.mean(np.minimum(dist, self.cfg.correspondence_max_m) ** 2)))

    def refine(self, predicted: np.ndarray, points: np.ndarray) -> tuple[np.ndarray, RegistrationResult]:
        """Refine the predicted state; returns (state, result).  Rejections return the prediction unchanged."""
        c = self.cfg
        state = predicted.copy()
        meta = RegistrationResult()
        if len(points) < c.min_points:
            return state, meta
        for _ in range(c.iterations):
            q, normal, distance = self.correspond(state, points)
            keep = np.isfinite(distance) & (distance < c.correspondence_max_m)
            meta.pairs = int(keep.sum())
            if keep.sum() < c.min_pairs:
                return predicted.copy(), meta
            q, normal, p = q[keep], normal[keep], points[keep]
            R = rz(state[3]) @ self.reference
            rotated = q @ R.T
            nw = normal @ R.T
            residual = np.sum(nw * (p - state[:3] - rotated), axis=1)
            yaw_derivative = np.column_stack((-rotated[:, 1], rotated[:, 0], np.zeros(len(q))))
            A = np.column_stack((nw, np.sum(nw * yaw_derivative, axis=1)))
            weights = np.sqrt(np.minimum(1.0, c.huber_m / np.maximum(abs(residual), 1e-9)))
            # Truncated SVD leaves geometry-unobserved directions at the prediction.
            delta = np.linalg.lstsq(A * weights[:, None], residual * weights, rcond=c.rcond)[0]
            total = state + delta - predicted
            total[:3] = np.clip(total[:3], -c.step_clip_m, c.step_clip_m)
            total[3] = np.clip(total[3], -np.deg2rad(c.step_clip_deg), np.deg2rad(c.step_clip_deg))
            updated = predicted + total
            if np.linalg.norm(updated - state) < 1e-5:
                state = updated
                break
            state = updated
        old, new = self.rmse(predicted, points), self.rmse(state, points)
        if not np.isfinite(state).all() or new > old + 1e-6:
            return predicted.copy(), RegistrationResult(False, "geometry_worsened", meta.pairs, old, new)
        return state, RegistrationResult(True, "geometry", meta.pairs, old, new)
