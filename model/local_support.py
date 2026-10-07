"""Persistent local-support statistics (the dashed "local support" store of the pipeline figure).

For every voxel at three scales (0.25 / 0.5 / 1.0 m) TEF keeps the running sums

    n, sum(p - c), sum((p - c)(p - c)^T)        (c = voxel centre, so float32 stays accurate)

of the returns seen so far.  Step II updates them with the current scan, then queries a local frame per return: the finest
level with at least ``minimum_points`` points and a dispersion ratio sqrt(lambda0 / lambda2) <= ``maximum_planarity`` gives
the normal (smallest-eigenvalue direction), two tangent axes and extents (square roots of the other eigenvalues, floored),
the thickness tau and a count confidence n / (n + minimum_points).  Returns that no level matches keep the
ray-orthogonal default frame supplied by the caller.
"""
from __future__ import annotations

import numpy as np
import torch

from model.sparse_field import pack_keys_t


def sym_eig3_t(cov: torch.Tensor):
    """Closed-form eigendecomposition of a batch of symmetric 3x3 matrices (ascending values, column eigenvectors).

    Pure elementwise tensor ops: no cuSOLVER call, no workspace (``torch.linalg.eigh`` requested a 10 GB workspace for
    24k tiny matrices once the process was large) and about 10x faster for this shape.  Eigenvalues by the
    trigonometric formula; eigenvectors as the largest cross product of two rows of (A - lambda I), the middle one
    completed by the cross product so the frame is orthonormal even for repeated eigenvalues.
    """

    A = cov.to(torch.float64)
    a11, a22, a33 = A[:, 0, 0], A[:, 1, 1], A[:, 2, 2]
    a12, a13, a23 = A[:, 0, 1], A[:, 0, 2], A[:, 1, 2]
    p1 = a12 * a12 + a13 * a13 + a23 * a23
    q = (a11 + a22 + a33) / 3.0
    p2 = (a11 - q) ** 2 + (a22 - q) ** 2 + (a33 - q) ** 2 + 2.0 * p1
    p = torch.sqrt(torch.clamp(p2 / 6.0, min=0.0))
    safe_p = torch.where(p > 1e-300, p, torch.ones_like(p))
    b11, b22, b33 = (a11 - q) / safe_p, (a22 - q) / safe_p, (a33 - q) / safe_p
    b12, b13, b23 = a12 / safe_p, a13 / safe_p, a23 / safe_p
    detB = b11 * (b22 * b33 - b23 * b23) - b12 * (b12 * b33 - b23 * b13) + b13 * (b12 * b23 - b22 * b13)
    r = torch.clamp(detB / 2.0, -1.0, 1.0)
    phi = torch.acos(r) / 3.0
    e_hi = q + 2.0 * p * torch.cos(phi)
    e_lo = q + 2.0 * p * torch.cos(phi + 2.0 * torch.pi / 3.0)
    e_mid = 3.0 * q - e_hi - e_lo
    diag = p <= 1e-300
    values = torch.stack([torch.where(diag, torch.minimum(torch.minimum(a11, a22), a33), e_lo),
                          torch.where(diag, a11 + a22 + a33 - torch.maximum(torch.maximum(a11, a22), a33) - torch.minimum(torch.minimum(a11, a22), a33), e_mid),
                          torch.where(diag, torch.maximum(torch.maximum(a11, a22), a33), e_hi)], dim=1)

    def eigvec(lam):
        M = A - lam[:, None, None] * torch.eye(3, dtype=A.dtype, device=A.device)
        r0, r1, r2 = M[:, 0], M[:, 1], M[:, 2]
        c01, c12, c20 = torch.cross(r0, r1, dim=1), torch.cross(r1, r2, dim=1), torch.cross(r2, r0, dim=1)
        n01, n12, n20 = (c01 * c01).sum(1), (c12 * c12).sum(1), (c20 * c20).sum(1)
        best = torch.where((n01 >= n12)[:, None] & (n01 >= n20)[:, None], c01, torch.where((n12 >= n20)[:, None], c12, c20))
        nrm = torch.linalg.norm(best, dim=1, keepdim=True)
        return best / nrm.clamp_min(1e-300), nrm.reshape(-1)

    v_lo, n_lo = eigvec(values[:, 0])
    v_hi, n_hi = eigvec(values[:, 2])
    # degenerate directions (repeated eigenvalue -> zero cross products): fall back to any unit vector orthogonal to the other
    fallback = torch.zeros_like(v_lo); fallback[:, 0] = 1.0
    v_lo = torch.where((n_lo < 1e-12)[:, None], fallback, v_lo)
    alt = torch.cross(v_lo, torch.where((v_lo[:, 0].abs() < 0.9)[:, None], fallback, torch.tensor([0.0, 1.0, 0.0], dtype=A.dtype, device=A.device).expand_as(v_lo)), dim=1)
    alt = alt / torch.linalg.norm(alt, dim=1, keepdim=True).clamp_min(1e-300)
    v_hi = torch.where((n_hi < 1e-12)[:, None], alt, v_hi)
    v_hi = v_hi - (v_hi * v_lo).sum(1, keepdim=True) * v_lo  # exact orthogonality
    v_hi = v_hi / torch.linalg.norm(v_hi, dim=1, keepdim=True).clamp_min(1e-300)
    v_mid = torch.cross(v_lo, v_hi, dim=1)
    vectors = torch.stack([v_lo, v_mid, v_hi], dim=2)  # columns, ascending like torch.linalg.eigh
    return values, vectors


class LocalSupport:
    """Multi-scale running per-voxel PCA over the points seen so far (device-resident, causal; research name StreamingBandField)."""

    def __init__(self, voxel_sizes_m=(0.25, 0.5, 1.0), *, minimum_points: int = 10, device: str = "cuda",
                 maximum_planarity: float = 0.35, tau_floor_m: float = 0.02) -> None:
        self.voxel_sizes = tuple(float(v) for v in voxel_sizes_m)
        if not self.voxel_sizes or any(v <= 0 for v in self.voxel_sizes):
            raise ValueError("voxel_sizes_m must be positive")
        if tuple(sorted(self.voxel_sizes)) != self.voxel_sizes:
            raise ValueError("voxel_sizes_m must be increasing (finest first)")
        self.minimum_points = int(minimum_points)
        self.maximum_planarity = float(maximum_planarity)
        self.tau_floor = float(tau_floor_m)
        self.device = torch.device(device)
        self.levels = [{"keys": torch.zeros(0, dtype=torch.int64, device=self.device),
                        "n": torch.zeros(0, dtype=torch.float32, device=self.device),
                        "s": torch.zeros((0, 3), dtype=torch.float32, device=self.device),
                        "m2": torch.zeros((0, 6), dtype=torch.float32, device=self.device)} for _ in self.voxel_sizes]

    # -- accumulation -------------------------------------------------------
    @staticmethod
    def _outer6(d: torch.Tensor) -> torch.Tensor:
        return torch.stack([d[:, 0] * d[:, 0], d[:, 1] * d[:, 1], d[:, 2] * d[:, 2],
                            d[:, 0] * d[:, 1], d[:, 0] * d[:, 2], d[:, 1] * d[:, 2]], dim=1)

    def update(self, points_world: torch.Tensor) -> None:
        """Fold one frame's returns (N,3) into every level."""

        p = points_world if isinstance(points_world, torch.Tensor) else torch.from_numpy(np.ascontiguousarray(points_world))
        p = p.to(self.device, torch.float64)
        if len(p) == 0:
            return
        for level, voxel in zip(self.levels, self.voxel_sizes):
            coords = torch.floor(p / voxel).to(torch.int64)
            keys = pack_keys_t(coords)
            delta = (p - (coords.to(torch.float64) + 0.5) * voxel).to(torch.float32)  # centred: float32 is exact enough
            uniq, inverse = torch.unique(keys, return_inverse=True)
            n = torch.zeros(len(uniq), dtype=torch.float32, device=self.device).index_add_(0, inverse, torch.ones(len(p), dtype=torch.float32, device=self.device))
            s = torch.zeros((len(uniq), 3), dtype=torch.float32, device=self.device).index_add_(0, inverse, delta)
            m2 = torch.zeros((len(uniq), 6), dtype=torch.float32, device=self.device).index_add_(0, inverse, self._outer6(delta))
            self._merge(level, uniq, n, s, m2)

    def _merge(self, level: dict, keys: torch.Tensor, n: torch.Tensor, s: torch.Tensor, m2: torch.Tensor) -> None:
        if not len(level["keys"]):
            level.update(keys=keys, n=n, s=s, m2=m2)
            return
        pos = torch.searchsorted(level["keys"], keys)
        pos_clamped = pos.clamp(max=len(level["keys"]) - 1)
        hit = level["keys"][pos_clamped] == keys
        if hit.any():
            rows = pos_clamped[hit]
            level["n"].index_add_(0, rows, n[hit]); level["s"].index_add_(0, rows, s[hit]); level["m2"].index_add_(0, rows, m2[hit])
        if (~hit).any():
            merged_keys = torch.cat([level["keys"], keys[~hit]])
            order = torch.argsort(merged_keys)
            level["keys"] = merged_keys[order]
            level["n"] = torch.cat([level["n"], n[~hit]])[order]
            level["s"] = torch.cat([level["s"], s[~hit]])[order]
            level["m2"] = torch.cat([level["m2"], m2[~hit]])[order]

    # -- query --------------------------------------------------------------
    def _eig(self, rows: torch.Tensor, level: dict):
        n = level["n"][rows].to(torch.float64).clamp_min(1.0)
        mean = level["s"][rows].to(torch.float64) / n[:, None]
        m2 = level["m2"][rows].to(torch.float64) / n[:, None]
        cov = torch.zeros((len(rows), 3, 3), dtype=torch.float64, device=self.device)
        cov[:, 0, 0] = m2[:, 0] - mean[:, 0] ** 2
        cov[:, 1, 1] = m2[:, 1] - mean[:, 1] ** 2
        cov[:, 2, 2] = m2[:, 2] - mean[:, 2] ** 2
        cov[:, 0, 1] = cov[:, 1, 0] = m2[:, 3] - mean[:, 0] * mean[:, 1]
        cov[:, 0, 2] = cov[:, 2, 0] = m2[:, 4] - mean[:, 0] * mean[:, 2]
        cov[:, 1, 2] = cov[:, 2, 1] = m2[:, 5] - mean[:, 1] * mean[:, 2]
        values, vectors = sym_eig3_t(cov)  # closed form: no cuSOLVER workspace, ~10x faster than linalg.eigh here
        return values.clamp_min(0.0), vectors, n

    def query(self, points_world, *, fallback_scale_m: float, fallback_tau_m: float, ray_basis_world, ray_normals_world) -> dict:
        """Local frame per point: ``tangent_basis_world`` (N,2,3), ``tangent_scales_m`` (N,2), ``tau_m``, ``confidence``,
        ``matched`` and ``endpoint_normal_world``.  Unmatched points keep the default frame: the ray-orthogonal basis
        ``ray_basis_world``, the sensor-facing normal ``ray_normals_world``, isotropic extent ``fallback_scale_m`` and
        thickness ``fallback_tau_m``."""

        q = points_world.to(self.device, torch.float64)
        count = len(q)
        basis = ray_basis_world.to(self.device, torch.float64).clone()
        normal = ray_normals_world.to(self.device, torch.float64).clone()
        scales = torch.full((count, 2), float(fallback_scale_m), dtype=torch.float64, device=self.device)
        tau = torch.full((count,), float(fallback_tau_m), dtype=torch.float64, device=self.device)
        confidence = torch.ones(count, dtype=torch.float64, device=self.device)
        matched = torch.zeros(count, dtype=torch.bool, device=self.device)
        if count == 0:
            return self._as_numpy(basis, scales, tau, confidence, matched, normal)
        for level, voxel in zip(self.levels, self.voxel_sizes):  # finest first; a coarser level only fills what is still unmatched
            todo = (~matched).nonzero(as_tuple=True)[0]
            if not len(todo) or not len(level["keys"]):
                continue
            coords = torch.floor(q[todo] / voxel).to(torch.int64)
            keys = pack_keys_t(coords)
            pos = torch.searchsorted(level["keys"], keys).clamp(max=len(level["keys"]) - 1)
            hit = level["keys"][pos] == keys
            rows = pos[hit]
            if not len(rows):
                continue
            enough = level["n"][rows] >= self.minimum_points
            rows = rows[enough]; target = todo[hit][enough]
            if not len(rows):
                continue
            values, vectors, n = self._eig(rows, level)
            planarity = torch.sqrt(values[:, 0] / values[:, 2].clamp_min(1e-12))
            good = (values[:, 2] > 1e-9) & (planarity <= self.maximum_planarity)
            target, values, vectors, n = target[good], values[good], vectors[good], n[good]
            if not len(target):
                continue
            new_normal = vectors[:, :, 0]
            flip = (new_normal * normal[target]).sum(1) < 0      # keep the sensor-facing hemisphere
            normal[target] = torch.where(flip[:, None], -new_normal, new_normal)
            basis[target] = torch.stack([vectors[:, :, 2], vectors[:, :, 1]], dim=1)  # largest extent first
            scales[target] = torch.sqrt(values[:, [2, 1]]).clamp_min(0.25 * float(fallback_scale_m))
            tau[target] = torch.sqrt(values[:, 0]).clamp_min(self.tau_floor)
            # saturating count confidence only: planarity already gates matching, so it must not be charged twice
            # (this matches the scale of the offline band confidences, median ~0.85, which weight the footprint samples)
            confidence[target] = (n / (n + float(self.minimum_points))).clamp(0.0, 1.0)
            matched[target] = True
        return self._as_numpy(basis, scales, tau, confidence, matched, normal)

    @staticmethod
    def _as_numpy(basis, scales, tau, confidence, matched, normal) -> dict:
        return {"tangent_basis_world": basis.cpu().numpy(), "tangent_scales_m": scales.cpu().numpy(), "tau_m": tau.cpu().numpy(),
                "confidence": confidence.cpu().numpy(), "matched": matched.cpu().numpy(), "endpoint_normal_world": normal.cpu().numpy(),
                "queried_rays": int(len(matched)), "matched_rays": int(matched.sum())}
