"""Small helpers: frame selection, device synchronisation, timers."""
from __future__ import annotations

import re
import time

import torch


def resolve_frames(spec: str) -> list[int]:
    """``train:START-STOP`` (held-out frames, index % 10 == 5, excluded), ``all:START-STOP`` or explicit indices."""

    match = re.fullmatch(r"(train|all):(\d+)-(\d+)", spec.strip())
    if match:
        mode, lo, hi = match.group(1), int(match.group(2)), int(match.group(3))
        if hi <= lo:
            raise ValueError("frame range needs STOP > START (STOP is exclusive)")
        indices = [i for i in range(lo, hi) if mode == "all" or i % 10 != 5]
    else:
        parts = spec.replace(",", " ").split()
        if not parts or not all(re.fullmatch(r"\d+", p) for p in parts):
            raise ValueError("use train:START-STOP, all:START-STOP or a list of nonnegative frame indices")
        indices = [int(p) for p in parts]
    if not indices:
        raise ValueError("the frame selection is empty")
    if indices != sorted(set(indices)):
        raise ValueError("frame indices must be unique and increasing")
    return indices


def synchronize(device) -> None:
    if torch.device(device).type == "cuda":
        torch.cuda.current_stream().synchronize()


class Timer:
    """``with timer("name"): ...`` accumulates synchronised wall time per name into ``timer.row``."""

    def __init__(self, device):
        self.device, self.row = device, {}

    def __call__(self, name: str):
        timer = self

        class _Section:
            def __enter__(self):
                self.t = time.time()

            def __exit__(self, *exc):
                synchronize(timer.device)
                timer.row[name] = timer.row.get(name, 0.0) + time.time() - self.t

        return _Section()
