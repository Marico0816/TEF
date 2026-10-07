"""Causal feature tracking and bounded 30-step Adam optimization of xyz/yaw.

Adapted from the saved image_gap/image_every_frame experiments. This optimizes
four state variables, not network weights. New anchors never supervise the
same frame that creates them. No annotation or future frame is read.
"""
from __future__ import annotations
import numpy as np
import torch


def rotation(yaw):
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[c, -s, 0.], [s, c, 0.], [0., 0., 1.]])


def matrix(state, reference):
    out = np.eye(4)
    out[:3, :3] = rotation(state[3]) @ reference
    out[:3, 3] = state[:3]
    return out


def inverse(T):
    out = np.eye(4)
    out[:3, :3] = T[:3, :3].T
    out[:3, 3] = -out[:3, :3] @ T[:3, 3]
    return out


def project(points, T, K):
    cam = points @ T[:3, :3].T + T[:3, 3]
    pix = cam @ K.T
    return pix[:, :2] / np.maximum(cam[:, 2:3], 1e-9), cam[:, 2]


def tensor(value):
    return torch.as_tensor(value, dtype=torch.float64)


def pose_torch(state, reference):
    c, s = torch.cos(state[3]), torch.sin(state[3])
    zero, one = state.new_tensor(0.), state.new_tensor(1.)
    R = torch.stack([c, -s, zero, s, c, zero, zero, zero, one]).reshape(3, 3)
    return R @ reference, state[:3]


def project_torch(state, points, C, K, reference):
    R, t = pose_torch(state, reference)
    cam = (points @ R.T + t) @ C[:3, :3].T + C[:3, 3]
    pix = cam @ K.T
    return pix[:, :2] / cam[:, 2:3].clamp_min(1e-6)


def refine(base, prediction, velocity, points, anchors, pixels, frame, geometry, joint):
    """Return the original base exactly unless finite, image, and geometry gates pass."""
    base = np.asarray(base, float)
    meta = dict(attempted=False, accepted=False, reason="insufficient_matches",
                matches=len(pixels), steps=0, mode="joint" if joint else "image_only")
    if len(pixels) < 8:
        return base.copy(), meta
    C, K, dt = frame["camera"], frame["K"], float(frame["image_dt"])
    uv, depth = project(anchors, C @ matrix(base + velocity * dt, geometry.reference), K)
    valid = ((depth > .5) & np.isfinite(uv).all(1) & np.isfinite(anchors).all(1)
             & np.isfinite(pixels).all(1) & (np.linalg.norm(uv - pixels, axis=1) <= 60.))
    q, target, initial_uv = anchors[valid], pixels[valid], uv[valid]
    meta["matches"] = len(target)
    if len(target) < 8 or np.any(target.std(0) < 2.):
        return base.copy(), dict(meta, reason="weak_support")
    geometry_args = None
    if joint:
        gq, gn, distance = geometry.correspond(base, points)
        keep = (np.isfinite(distance) & (distance < .5) & np.isfinite(gn).all(1)
                & (np.linalg.norm(gn, axis=1) > .5))
        meta["lidar_pairs"] = int(keep.sum())
        if keep.sum() < 30:
            return base.copy(), dict(meta, reason="geometry_factor_missing")
        geometry_args = tuple(map(tensor, (gq[keep], gn[keep], points[keep])))
    b, pred, advance, qt, pt, Ct, Kt, ref = map(tensor,
        (base, prediction, velocity * dt, q, target, C, K, geometry.reference))
    scales = tensor([.5, .5, .5, np.deg2rad(10)])
    x = torch.zeros(4, dtype=torch.float64, requires_grad=True)
    optimizer = torch.optim.Adam([x], lr=.08)
    best, best_loss, first = base.copy(), float("inf"), None
    before_parts = after_parts = None
    meta["attempted"] = True
    for step in range(31):
        optimizer.zero_grad()
        normalized = torch.tanh(x)
        pose = b + scales * normalized
        projected = project_torch(pose + advance, qt, Ct, Kt, ref)
        image_loss = torch.nn.functional.smooth_l1_loss((projected - pt) / 3., torch.zeros_like(pt))
        geometric_loss = pose.sum() * 0.
        if geometry_args is not None:
            gq, gn, gp = geometry_args
            R, t = pose_torch(pose, ref)
            residual = ((gq @ R.T + t - gp) * (gn @ R.T)).sum(1) / .05
            geometric_loss = torch.nn.functional.smooth_l1_loss(residual, torch.zeros_like(residual))
            prior_loss = .02 * ((pose - pred) / scales).square().mean()
        else:
            prior_loss = .02 * normalized.square().mean()
        loss = image_loss + geometric_loss + prior_loss
        value = float(loss.detach())
        parts = [float(v.detach()) for v in (image_loss, geometric_loss, prior_loss)]
        if first is None:
            first, before_parts = value, parts
        if np.isfinite(value) and value < best_loss:
            best_loss, best, after_parts = value, pose.detach().numpy().copy(), parts
        if step == 30:
            break
        loss.backward()
        if x.grad is None or not torch.isfinite(x.grad).all():
            return base.copy(), dict(meta, reason="nonfinite_gradient")
        optimizer.step()
        meta["steps"] += 1
    uv, depth = project(q, C @ matrix(best + velocity * dt, geometry.reference), K)
    before = np.linalg.norm(initial_uv - target, axis=1)
    after = np.linalg.norm(uv - target, axis=1)
    inliers = (after <= 4.) & (depth > .5)
    image_ok = (inliers.sum() >= 8 and inliers.mean() >= .6 and np.median(after) <= 3.
                and np.median(before) - np.median(after) >= .1)
    geometry_ok = True
    if len(points):
        before_g, after_g = geometry.rmse(base, points), geometry.rmse(best, points)
        geometry_ok = np.isfinite(after_g) and after_g <= before_g + .01
        meta.update(geometry_before_m=float(before_g), geometry_after_m=float(after_g))
    descent = np.isfinite(best).all() and np.isfinite(best_loss) and best_loss < first
    accepted = bool(image_ok and geometry_ok and descent)
    reason = ("image_gradient" if accepted else "image_quality_rejected" if not image_ok
              else "geometry_guard_rejected" if not geometry_ok else "no_descent")
    meta.update(accepted=accepted, reason=reason, loss_before=first, loss_after=best_loss,
                components_before=before_parts, components_after=after_parts,
                median_before_px=float(np.median(before)), median_after_px=float(np.median(after)),
                inliers=int(inliers.sum()), candidate=best.tolist())
    return best if accepted else base.copy(), meta


class FeatureTrack:
    """Forward/backward LK with semantic support and mesh-intersection anchors."""
    def __init__(self):
        self.gray = None
        self.stamp = None
        self.anchors = np.empty((0, 3))
        self.pixels = np.empty((0, 2))

    def reset(self):
        self.gray, self.stamp = None, None
        self.anchors = np.empty((0, 3))
        self.pixels = np.empty((0, 2))

    def advance(self, frame):
        import cv2
        stamp = int(frame["image_timestamp_ns"])
        if self.stamp is not None and stamp <= self.stamp:
            raise ValueError("Image timestamps must increase; duplicate images cannot be reused as new evidence")
        if self.stamp is not None and stamp - self.stamp > 350_000_000:
            self.reset()
        if self.gray is None or not len(self.pixels):
            return dict(input=0, kept=0)
        p = self.pixels.astype(np.float32).reshape(-1, 1, 2)
        kw = dict(winSize=(21, 21), maxLevel=3,
                  criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, .01))
        q, status, error = cv2.calcOpticalFlowPyrLK(self.gray, frame["gray"], p, None, **kw)
        if q is None:
            self.reset()
            return dict(input=len(p), kept=0)
        back, bs, _ = cv2.calcOpticalFlowPyrLK(frame["gray"], self.gray, q, None, **kw)
        if back is None:
            self.reset()
            return dict(input=len(p), kept=0)
        q, back = q[:, 0], back[:, 0]
        H, W = frame["gray"].shape
        finite = np.isfinite(q).all(1) & np.isfinite(back).all(1)
        ij = np.rint(np.nan_to_num(q)).astype(int)
        sem = frame["vehicle"][np.clip(ij[:, 1], 0, H - 1), np.clip(ij[:, 0], 0, W - 1)]
        keep = (finite & status[:, 0].astype(bool) & bs[:, 0].astype(bool)
                & (np.linalg.norm(back - p[:, 0], axis=1) <= 1.5) & (error[:, 0] <= 25.)
                & (q[:, 0] >= 0) & (q[:, 0] < W) & (q[:, 1] >= 0) & (q[:, 1] < H) & sem)
        self.anchors, self.pixels = self.anchors[keep], q[keep].astype(float)
        return dict(input=len(p), kept=int(keep.sum()))

    def seed(self, geometry, pose, velocity, points, frame):
        import cv2
        import open3d as o3d
        from scipy.spatial import cKDTree
        C, K = frame["camera"], frame["K"]
        W = matrix(pose, geometry.reference)
        WI = matrix(pose + velocity * float(frame["image_dt"]), geometry.reference)
        local = (points - W[:3, 3]) @ W[:3, :3]
        uv, depth = project(local, C @ WI, K)
        H, width = frame["gray"].shape
        valid = ((depth > .5) & np.isfinite(uv).all(1) & (uv[:, 0] >= 0) & (uv[:, 0] < width)
                 & (uv[:, 1] >= 0) & (uv[:, 1] < H))
        uv, depth = uv[valid], depth[valid]
        self.anchors, self.pixels = np.empty((0, 3)), np.empty((0, 2))
        if len(uv) < 3:
            return dict(anchors=0, reason="no_support")
        mask = np.zeros_like(frame["gray"])
        cv2.fillConvexPoly(mask, cv2.convexHull(np.rint(uv).astype(np.int32)), 255)
        mask[~frame["vehicle"].astype(bool)] = 0
        mask = cv2.erode(mask, np.ones((3, 3), np.uint8))
        corners = cv2.goodFeaturesToTrack(frame["gray"], 200, .01, 5., mask=mask, blockSize=7)
        if corners is None:
            return dict(anchors=0, reason="no_corners")
        pixels = corners[:, 0].astype(float)
        distance, neighbor = cKDTree(uv).query(pixels)
        CO = C @ WI
        origin = inverse(CO)[:3, 3]
        directions = (np.c_[pixels, np.ones(len(pixels))] @ np.linalg.inv(K).T) @ CO[:3, :3]
        rays = np.c_[np.broadcast_to(origin, directions.shape), directions].astype(np.float32)
        hits = geometry.scene.cast_rays(o3d.core.Tensor(rays), nthreads=4)["t_hit"].numpy()
        good = (np.isfinite(hits) & (hits > .5) & (distance <= 12.) & (abs(hits - depth[neighbor]) <= .5))
        self.anchors, self.pixels = origin + directions[good] * hits[good, None], pixels[good]
        return dict(anchors=int(good.sum()), reason="refreshed")

    def finish(self, frame):
        self.gray = frame["gray"]
        self.stamp = int(frame["image_timestamp_ns"])
