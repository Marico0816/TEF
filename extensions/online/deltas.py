"""Incremental mesh output: one delta file per extraction, holding the new geometry of every re-extracted block and the
ids of the blocks that became empty.  Applying the deltas in block order reproduces the mesh of each output time
(research ``mesh_io.py``, unchanged)."""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np


def write_mesh_delta(directory, block: int, changed_blocks, blocks: dict, stamp_ns: int) -> Path:
    """Write ``delta_<block>.npz`` (to a temporary name, then renamed: a delta is complete or absent)."""
    d = Path(directory)
    d.mkdir(parents=True, exist_ok=True)
    ids = [int(b) for b in changed_blocks]
    present = [b for b in ids if b in blocks]
    V = [np.asarray(blocks[b][0], np.float32) for b in present]
    F = [np.asarray(blocks[b][1], np.int32) for b in present]
    tmp, out = d / f"delta_{int(block):05d}.part.npz", d / f"delta_{int(block):05d}.npz"
    np.savez(tmp, chunk=np.int64(block), stamp_ns=np.int64(stamp_ns), ids=np.asarray(present, np.int64),
             removed=np.asarray([b for b in ids if b not in blocks], np.int64),
             nv=np.asarray([len(v) for v in V], np.int64), nf=np.asarray([len(f) for f in F], np.int64),
             V=np.concatenate(V) if V else np.zeros((0, 3), np.float32), F=np.concatenate(F) if F else np.zeros((0, 3), np.int32))
    os.replace(tmp, out)
    return out


def apply_mesh_delta(blocks: dict, path) -> int:
    """Apply one delta file to ``blocks`` (block id -> (vertices, faces)) in place; returns its block index."""
    with np.load(path) as z:
        V, F = z["V"], z["F"]
        for b in z["removed"]:
            blocks.pop(int(b), None)
        v0 = f0 = 0
        for b, nv, nf in zip(z["ids"], z["nv"], z["nf"]):
            blocks[int(b)] = (V[v0:v0 + nv].copy(), F[f0:f0 + nf].copy())
            v0 += int(nv)
            f0 += int(nf)
        return int(z["chunk"])


def assemble_blocks(blocks: dict):
    """(vertices, faces) of ``blocks`` in ascending id order (as ``Mesher.assemble``)."""
    vs, fs, off = [], [], 0
    for b in sorted(blocks):
        v, f = blocks[b][:2]
        vs.append(v)
        fs.append(f + off)
        off += len(v)
    if not vs:
        return np.zeros((0, 3), np.float32), np.zeros((0, 3), np.int32)
    return np.concatenate(vs), np.concatenate(fs).astype(np.int32)


def delta_files(directory):
    return sorted(p for p in Path(directory).glob("delta_*.npz") if not p.name.endswith(".part.npz"))


def load_mesh_deltas(directory, upto_block: int):
    """Mesh (vertices, faces) after applying every delta with block index <= ``upto_block``."""
    blocks = {}
    for path in delta_files(directory):
        if int(path.stem.split("_")[1]) > int(upto_block):
            break
        apply_mesh_delta(blocks, path)
    return assemble_blocks(blocks)
