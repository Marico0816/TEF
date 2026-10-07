"""Optional heading guard for vehicle targets (``--vehicle-heading-guard-deg``; off by default).

A vehicle driving faster than ``V_MIN`` points along its direction of travel, up to a fixed per-track offset between the
object frame and the travel direction.  The guard compares each candidate heading with a travel direction estimated only
from *reliable* past positions and gives every frame exactly one outcome:

* ``valid``        accepted, unsaturated registration consistent with the travel direction (fused, feeds the motion model)
* ``restart``      an inconsistent candidate was re-registered from the travel-aligned heading and that result is reliable
* ``reacquired``   the first reliable registration after a constrained stretch; tracking and shape fusion resume here
* ``image_only``   an accepted, consistent image-only correction (as without the guard: committed, never fused)
* ``prediction``   an uncommitted prediction consistent with the travel direction (as without the guard)
* ``enter``        an inconsistent candidate that could not be restarted: the displayed, saved and next-start state is the
                   prediction re-anchored to the travel direction, with zero yaw rate and horizontal travel velocity
* ``constrained``  every later frame until a reliable registration: the anchored prediction, committed, never fused

A registration is reliable only if dynobj accepts it, it is not stopped by dynobj's step clip (saturated), and its heading
is consistent.  The travel direction is a least-squares fit over the reliable positions of the last ``WINDOW_S`` seconds
(never the prediction or the velocity state); it only changes when a reliable position is added.  Non-vehicle classes and
tracks that start slower than ``V_MIN`` are left unguarded.
"""
from __future__ import annotations

import numpy as np

VEHICLE_CLASSES = (13, 14, 15)          # Cityscapes car, truck, bus (the frontend tracker's class ids, as gubmap.vehicle_refine)
V_MIN, WINDOW_S, MIN_SPAN_S, SAT_TOL = 2.0, 1.0, 0.3, 1e-9
EVENTS = ("restart", "enter", "reacquire", "saturated", "image_discard")
FUSED, COMMITTED = ("valid", "restart", "reacquired"), ("valid", "restart", "reacquired", "image_only", "enter", "constrained")


def wrap(angle):
    return float(np.arctan2(np.sin(angle), np.cos(angle)))


def track_class(tracks, track_id):
    """Frontend class id of a cached track, or None."""
    if "track_class" not in getattr(tracks, "z", {}):
        return None
    return {int(a): int(b) for a, b in np.asarray(tracks.z["track_class"]).reshape(-1, 2)}.get(int(track_id))


class VehicleHeadingGuard:
    def __init__(self, cone_deg, class_id, model, step_clip_deg, step_clip_m):
        self.cone, self.class_id = np.radians(float(cone_deg)), class_id
        self.step_yaw, self.step_xyz = np.radians(float(step_clip_deg)), float(step_clip_m)
        v0 = np.asarray(model.velocity, float)[:2]
        self.v0_speed = float(np.hypot(*v0))
        self.enabled, self.disabled_reason = True, None
        if class_id not in VEHICLE_CLASSES:
            self.enabled, self.disabled_reason = False, "class_unknown" if class_id is None else f"class_{class_id}"
        elif self.v0_speed < V_MIN:
            self.enabled, self.disabled_reason = False, "slow_start"
        self.offset = wrap(model.state[3] - np.arctan2(v0[1], v0[0])) if self.enabled else None
        self.start_t = float(model.time)
        self.history = [(float(model.time), *np.asarray(model.state, float)[:3])]
        self.travel, self.travel_t = v0.copy(), float(model.time)
        self.travel_source, self.travel_n, self.travel_span = "initial", 1, 0.0
        self.mode = "tracking"
        self.event_counts = {e: 0 for e in EVENTS}
        self.early = []

    # ------------------------------------------------------------------ travel estimate (reliable history only)
    def course(self):
        return float(np.arctan2(self.travel[1], self.travel[0]))

    def speed(self):
        return float(np.hypot(*self.travel))

    def velocity(self):
        return np.array([self.travel[0], self.travel[1], 0., 0.])

    def append(self, t, xyz):
        self.history.append((float(t), *np.asarray(xyz, float)[:3]))
        h = np.asarray([r for r in self.history if r[0] >= self.history[-1][0] - WINDOW_S])
        if len(h) >= 2 and h[-1, 0] - h[0, 0] >= MIN_SPAN_S:
            dt = h[:, 0] - h[:, 0].mean()
            self.travel = (dt @ (h[:, 1:3] - h[:, 1:3].mean(0))) / (dt @ dt)
            self.travel_t, self.travel_source, self.travel_n, self.travel_span = float(t), "fit", len(h), float(h[-1, 0] - h[0, 0])
        else:
            self.travel_source = "kept"

    # ------------------------------------------------------------------ checks
    def applicable(self):
        return self.enabled and self.speed() >= V_MIN

    def deviation(self, state, course):
        return wrap(state[3] - self.offset - course) if self.enabled else None

    def consistent(self, state, course):
        return not self.applicable() or abs(self.deviation(state, course)) <= self.cone

    def anchor(self, state, course):
        out = np.asarray(state, float).copy()
        out[3] = state[3] + wrap(self.offset + course - state[3])
        return out

    def saturated(self, result, start):
        yaw = abs(wrap(result[3] - start[3])) >= self.step_yaw - SAT_TOL
        xyz = float(np.max(np.abs(np.asarray(result[:3]) - np.asarray(start[:3])))) >= self.step_xyz - SAT_TOL
        return bool(yaw), bool(xyz)

    # ------------------------------------------------------------------ per-frame classification (before any image step)
    def classify(self, heldout, reg, geo, predicted, points_w, geometry):
        """(outcome, lidar_state, info).  ``reg`` / ``geo`` are dynobj's registration from ``predicted`` (ignored when held out)."""
        course = self.course()
        deg = lambda a: None if a is None else float(np.degrees(a))
        info = dict(check_applicable=self.applicable(), course_deg=deg(course), speed_mps=self.speed(), travel_source=self.travel_source,
                    travel_n=self.travel_n, travel_span_s=self.travel_span, travel_velocity=self.velocity().tolist(), events=[],
                    registration=None, prediction_deviation_deg=deg(self.deviation(predicted, course)), trigger_deviation_deg=None,
                    restart=dict(attempted=False))
        accepted = bool(not heldout and geo["accepted"])
        sat = False
        if not heldout:
            sy, sx = self.saturated(reg, predicted)
            sat = sy or sx
            info["registration"] = dict(saturated_yaw=sy, saturated_xyz=sx, deviation_deg=deg(self.deviation(reg, course)))
        if self.mode == "constrained":
            if accepted and not sat and self.consistent(reg, course):
                info["events"].append("reacquire")
                return "reacquired", reg.copy(), info
            if accepted and sat:
                info["events"].append("saturated")
            return "constrained", None, info
        if accepted and not self.consistent(reg, course):
            info["trigger_deviation_deg"] = deg(self.deviation(reg, course))
            return self._restart(predicted, points_w, geometry, course, geo, True, info)
        if accepted and not sat:
            return "valid", reg.copy(), info
        if accepted and sat:
            info["events"].append("saturated")
        if self.consistent(predicted, course):
            return "prediction", None, info
        info["trigger_deviation_deg"] = deg(self.deviation(predicted, course))
        if not heldout and len(points_w) >= geometry.cfg.min_points:
            return self._restart(predicted, points_w, geometry, course, geo, accepted, info)
        info["events"].append("enter")
        return "enter", None, info

    def _restart(self, predicted, points_w, geometry, course, geo, original_accepted, info):
        hyp = self.anchor(predicted, course)
        state, result = geometry.refine(hyp, points_w)
        sy, sx = self.saturated(state, hyp)
        ok = (result.accepted and not (sy or sx) and self.consistent(state, course)
              and (not original_accepted or result.rmse_after <= geo["rmse_after"] + 1e-6))
        info["restart"] = dict(attempted=True, accepted=bool(result.accepted), saturated_yaw=sy, saturated_xyz=sx, pairs=int(result.pairs),
                               rmse_before=result.rmse_before, rmse_after=result.rmse_after,
                               deviation_deg=float(np.degrees(self.deviation(state, course))), start_state=hyp.tolist(), state=state.tolist())
        if ok:
            info["events"].append("restart")
            return "restart", state.copy(), info
        info["events"].append("enter")
        return "enter", None, info

    def record(self):
        return dict(cone_deg=float(np.degrees(self.cone)), v_min=V_MIN, window_s=WINDOW_S, min_span_s=MIN_SPAN_S,
                    step_clip_deg=float(np.degrees(self.step_yaw)), step_clip_m=self.step_xyz, class_id=self.class_id, enabled=self.enabled,
                    disabled_reason=self.disabled_reason, offset_deg=None if self.offset is None else float(np.degrees(self.offset)),
                    v0_speed_mps=self.v0_speed, early_median_deviation_deg=float(np.degrees(np.median(self.early))) if self.early else None,
                    event_counts=dict(self.event_counts))
