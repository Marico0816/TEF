"""Paced replay (online extension): a scan is read only once its timestamp has "arrived" at ``replay_speed`` x real time,
and the arrival, start and end of every scan are recorded.  ``latency_summary`` turns a run summary into the online
figures: output latency (mesh ready - arrival of the block's last scan), processing lag behind arrival, backlog."""
from __future__ import annotations

import time

import numpy as np


class Pacer:
    """Arrival schedule of the selected frames; ``wait`` blocks until frame ``seq`` has arrived."""

    def __init__(self, dataset, frames, speed: float):
        if not float(speed) > 0:
            raise ValueError("replay_speed must be positive")
        self.speed = float(speed)
        self.t0 = time.time()
        stamps = np.array([int(dataset.records[int(i)].lidar_timestamp_ns) for i in frames], dtype=np.int64)
        self.arrival = self.t0 + (stamps - stamps[0]) * 1e-9 / self.speed
        self.record = {"replay_speed": self.speed, "t0_wall": self.t0, "first_stamp_ns": int(stamps[0]), "frames": []}
        self._start = None

    def wait(self, seq: int):
        delay = self.arrival[seq] - time.time()
        if delay > 0:
            time.sleep(delay)
        self._start = time.time()

    def done(self, seq: int, index: int, stamp_ns: int, block: int):
        t_end = time.time()
        self.record["frames"].append({"seq": seq, "index": int(index), "stamp_ns": int(stamp_ns), "block": int(block),
                                      "arrival_wall": float(self.arrival[seq]), "start_wall": self._start, "end_wall": t_end,
                                      "lag_s": self._start - float(self.arrival[seq]),
                                      "backlog_frames": int(np.searchsorted(self.arrival, self._start, side="right") - seq - 1)})


def latency_summary(summary: dict) -> dict:
    """Latency of every output that has a ``mesh_ready_wall`` (the final extraction is reported separately), processing
    lag and backlog of a paced run (``summary`` = the run summary written with ``--timing-json``)."""
    online = summary.get("online")
    if not online or not online.get("frames"):
        return {}
    frames = online["frames"]
    last_arrival = {}
    for f in frames:
        last_arrival[f["block"]] = max(last_arrival.get(f["block"], -1e18), f["arrival_wall"])
    lat = [r["mesh_ready_wall"] - last_arrival[r["block"]] for r in summary["blocks_detail"]
           if r.get("mesh_ready_wall") and r["block"] in last_arrival]
    lag = np.array([f["lag_s"] for f in frames])
    third = max(1, len(lag) // 3)
    q = lambda x, p: float(np.percentile(x, p)) if len(x) else None   # noqa: E731
    return {"outputs": len(lat), "latency_s": {"median": q(lat, 50), "p95": q(lat, 95), "max": float(max(lat)) if lat else None},
            "lag_s": {"first_third_median": float(np.median(lag[:third])), "last_third_median": float(np.median(lag[-third:])),
                      "max": float(lag.max())},
            "max_backlog_frames": int(max(f["backlog_frames"] for f in frames))}
