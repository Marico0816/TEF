"""Optional extension: online output (off in the paper configuration).

Adds to the six steps, without changing what they compute:
  * ``incremental_remesh``: at every output only the blocks whose nodes moved since their meshes were extracted are
    re-extracted, with a one-voxel halo of context (``incremental.py``); the final extraction is unchanged;
  * ``background_remesh``: that extraction runs on a worker thread and CUDA stream while the next block is integrated;
  * ``region_candidates``: pass-vote candidates are restricted to the blocks the block's rays can reach (same votes);
  * ``mesh_delta_dir``: every output writes a mesh delta file (``deltas.py``);
  * ``replay_speed``: scans are read only when their timestamp has "arrived" (``replay.py``), and the latency of every
    output is recorded.
Enable with ``config/extensions/online.yaml`` (per-block output, incremental + background remeshing, region candidates).
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import contextlib
import time

import torch

from extensions.online.deltas import write_mesh_delta
from extensions.online.incremental import MeshReference, corner_blocks, halo_rows, ray_region_blocks, store_block_faces
from utils.mesher import extract_surface_nets_t


class OnlineOutput:
    """Mesh reference, background extraction and delta output of the online extension."""

    def __init__(self, cfg, mesher, device: str):
        self.cfg, self.mesher, self.device, self.dev = cfg, mesher, device, torch.device(device)
        self.ref = MeshReference(cfg.remesh_reuse_tol, cfg.remesh_gate_tol, cfg.min_node_weight) if cfg.incremental_remesh else None
        self.pool = ThreadPoolExecutor(max_workers=1) if cfg.background_remesh else None
        self.stream = torch.cuda.Stream(device=self.dev) if self.pool is not None and self.dev.type == "cuda" else None
        self.pending = None          # (future, block row) of a background extraction

    @property
    def incremental(self) -> bool:
        return self.ref is not None

    # -- III: the persistent field changed its rows ------------------------------------------------------------------
    def after_merge(self, field, old_inv):
        if self.ref is not None:
            self.ref.after_merge(len(field), old_inv, field.device, field.dtype)

    def vote_region(self, views, voxel: float, clearance_m: float):
        """Blocks the block's rays can reach (pass-vote candidates), or None without ``region_candidates``."""
        if not self.cfg.region_candidates:
            return None
        return ray_region_blocks(views, voxel, self.cfg.block_voxels, clearance_m, self.dev)

    # -- VI: which blocks, and the extraction job ---------------------------------------------------------------------
    def schedule(self, field, coords, block_of, c: int, final: bool):
        """Incremental remeshing: the blocks to extract now and, for a non-final output, the job state
        ``(dirty rows, current sdf, current weight)``; the final extraction takes every block (as the paper path)."""
        mesher = self.mesher
        if not mesher.due(c, final):
            return torch.zeros(0, dtype=torch.int64, device=self.dev), None
        evicted = [] if mesher.pending_blocks is None else [mesher.pending_blocks]
        mesher.pending_blocks = None
        if final:
            return torch.unique(torch.cat([block_of] + evicted)), None
        sdf, weight = field.current_sdf(), field.extraction_weight()
        dirty = self.ref.dirty_rows(sdf, weight)
        blocks = torch.unique(torch.cat([corner_blocks(coords, dirty, self.cfg.block_voxels)] + evicted))
        return blocks, (dirty, sdf, weight)

    def launch(self, c: int, blocks, coords, block_of, job, stamp_ns: int, row: dict):
        """Gather the one-voxel context of ``blocks`` (copies: the field keeps changing), commit the reference and run the
        extraction inline or on the background thread.  The block row is completed when the job is done."""
        dirty, sdf, weight = job
        ext = halo_rows(coords, block_of, blocks, self.cfg.block_voxels) if len(blocks) else torch.zeros(0, dtype=torch.int64, device=self.dev)
        inputs = (c, blocks, coords[ext], sdf[ext], weight[ext], int(stamp_ns))
        self.ref.commit(dirty, sdf, weight)
        row["extract_nodes"] = int(len(ext))
        if self.pool is None:
            row.update(self._remesh_job(*inputs, None))
            return
        event = None
        if self.dev.type == "cuda":
            event = torch.cuda.Event()
            event.record()
        self.pending = (self.pool.submit(self._remesh_job, *inputs, event), row)

    def join(self):
        """Wait for the background extraction of an earlier block and complete its row."""
        if self.pending is not None:
            future, row = self.pending
            self.pending = None
            t = time.time()
            row.update(future.result())
            row["mesh_wait_s"] = time.time() - t

    def _remesh_job(self, c, blocks, xc, xs, xw, stamp_ns, event):
        mesher = self.mesher
        t0 = time.time()
        stream = self.stream if event is not None else None
        with torch.cuda.stream(stream) if stream is not None else contextlib.nullcontext():
            if stream is not None:
                stream.wait_event(event)
            out = None
            if len(xc):
                out = extract_surface_nets_t(xc, xs, xw, mesher.voxel, device=self.device, cube_batch=mesher.cube_batch,
                                             cube_candidates="edges", return_tensors=True, **mesher.extract_kw)
            changed = blocks.cpu().numpy()
            for b in changed:
                mesher.blocks.pop(int(b), None)
            n = 0
            if out is not None and len(out[1]):
                n = store_block_faces(mesher.blocks, out[0], out[1], out[2], out[3], blocks, mesher.block_voxels, self.dev)
            if self.dev.type == "cuda":
                torch.cuda.current_stream().synchronize()
        result = {"region_faces": int(n), "extract_job_s": time.time() - t0, "delta_s": 0.0}
        if self.cfg.mesh_delta_dir:
            t_d = time.time()
            write_mesh_delta(self.cfg.mesh_delta_dir, c, changed, mesher.blocks, stamp_ns)
            result["delta_s"] = time.time() - t_d
        result["mesh_faces"] = int(sum(len(b[1]) for b in mesher.blocks.values()))
        result["mesh_ready_wall"] = time.time()
        return result

    def after_extract(self, c: int, blocks, stamp_ns: int, row: dict):
        """Periodic (non-incremental) output: write its delta and mark the mesh as ready."""
        if self.cfg.mesh_delta_dir:
            t_d = time.time()
            write_mesh_delta(self.cfg.mesh_delta_dir, c, blocks.cpu().numpy(), self.mesher.blocks, int(stamp_ns))
            row["delta_s"] = time.time() - t_d
        row["mesh_ready_wall"] = time.time()

    # -- eviction, end of run -------------------------------------------------------------------------------------------
    def evict(self, far):
        if self.ref is not None:
            self.ref.evict(far)

    def close(self):
        self.join()
        if self.pool is not None:
            self.pool.shutdown(wait=True)
