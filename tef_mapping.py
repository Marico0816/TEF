#!/usr/bin/env python3
"""TEF: temporal evidence fusion of given-pose LiDAR scans into a triangle mesh.

    python tef_mapping.py config/tef_t2.yaml --dataset DIR --frames train:0-600 --output outputs/t2.ply

Per scan    I.   read and preprocess the scan
            II.  local support (multi-scale running PCA) and ray / lateral samples
            III. accumulate the samples into the block-local field of the current temporal block
Per block   III. merge the block (bounded block weight, one hit vote per node) and count pass votes
            IV.  conflict target from the fused value and the hit / pass counts
            V.   regularised solve on the region the block touched
            VI.  surface extraction (periodic or final), eviction of far blocks
Optional extensions (off in the paper configuration) are enabled by config overlays, e.g.
    python tef_mapping.py config/tef_t2.yaml config/extensions/online.yaml ...
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import time

import numpy as np
import torch

from dataset import DatasetLoader
from dataset.trajectory import trajectory_pose_fn
from model.sparse_field import SparseField, block_keys, unpack_keys_t
from utils.block_fusion import BlockFusion
from utils.config import Config
from utils.conflict_target import conflict_target
from utils.mesher import Mesher
from utils.preprocess import prepare_frame
from utils.regularizer import damped_jacobi, solve_region
from utils.sampler import Sampler, deterministic_subsample_indices
from utils.tools import Timer, resolve_frames


class TEFMapper:
    """The map (persistent sparse field) and the modules of the six steps."""

    def __init__(self, cfg: Config):
        self.cfg, self.device = cfg, cfg.device
        self.sampler = Sampler(cfg, cfg.device)                                    # II
        self.fusion = BlockFusion(cfg, cfg.device)                                 # III
        self.field = SparseField(cfg.voxel_m, cfg.truncation_m, device=cfg.device)  # the map
        self.mesher = Mesher(cfg, cfg.device)                                      # VI
        self.online = None                                                         # optional extension
        if cfg.incremental_remesh or cfg.region_candidates or cfg.mesh_delta_dir or cfg.replay_speed:
            from extensions.online import OnlineOutput
            self.online = OnlineOutput(cfg, self.mesher, cfg.device)
        self.frame_rows, self.block_rows, self.input_budget = [], [], {}

    # ------------------------------------------------------------------------------------------------------------------
    # per scan: II + III (accumulate)
    # ------------------------------------------------------------------------------------------------------------------
    def integrate_scan(self, seq: int, frame, block_id: int, points, neighbours):
        cfg, fusion = self.cfg, self.fusion
        t0 = time.time()
        flush_s = 0.0
        if block_id != fusion.block_id:
            if fusion.block_field is not None:
                self.process_block()                    # the previous temporal block is complete: steps III-VI
                flush_s = time.time() - t0
            fusion.begin_block(block_id)
        T_wl = np.asarray(frame.world_from_lidar, dtype=np.float64)
        selected = deterministic_subsample_indices(len(points), cfg.max_rays_per_frame)
        fusion.add_rays(T_wl[:3, 3], np.asarray(points[selected], dtype=np.float64) @ T_wl[:3, :3].T + T_wl[:3, 3])
        timer = Timer(self.device)

        # II. Local support and samples
        with timer("support_s"):
            support = self.sampler.local_frame(points, T_wl, selected)
        with timer("sample_s"):
            positions, values, weights = self.sampler.samples(points, T_wl, support, selected, neighbours)

        # III. Accumulate the scan into the block-local field (one frame bit per scan)
        with timer("splat_s"):
            fusion.accumulate(positions, values, weights, fusion.next_frame_bit())

        row = {"frame": int(seq), "block": int(block_id), **timer.row, "flush_s": flush_s,
               "frame_total_s": time.time() - t0, "block_nodes": len(fusion.block_field)}
        self.frame_rows.append(row)
        return row

    # ------------------------------------------------------------------------------------------------------------------
    # per block: III (merge, votes) -> IV -> V -> VI
    # ------------------------------------------------------------------------------------------------------------------
    def process_block(self, final: bool = False):
        cfg, field, fusion, mesher, dev = self.cfg, self.field, self.fusion, self.mesher, torch.device(self.device)
        online = self.online
        c, views = int(fusion.block_id), fusion.block_views
        timer = Timer(self.device)
        t0 = time.time()
        row = {"block": c, "frames": len(views), "data_stamp_ns": int(fusion.block_last_stamp_ns)}

        # III.1 Temporal-block merge: bounded block weight, one hit vote per touched node
        with timer("merge_s"):
            n_new, old_inv, touched = fusion.merge_block(field, c)
        if online is not None:
            online.after_merge(field, old_inv)

        stats = {"candidates": 0, "voted": 0, "newly_voted": 0}
        iters, max_update, n_solve = 0, 0.0, 0
        changed_rows = torch.zeros(0, dtype=torch.int64, device=dev)
        if cfg.evidence:
            # III.2 Pass votes: block rays that traverse nodes inside the fused surface (at most one vote per block)
            with timer("votes_s"):
                region = online.vote_region(views, fusion.voxel, fusion.alpha_clearance) if online is not None else None
                coords, raw, block_of, newly, stats = fusion.count_passes(field, c, old_inv, touched, views, region_blocks=region)

            with timer("solve_s"):
                sub_idx, interior = solve_region(block_of, touched, newly, cfg.block_voxels, cfg.solve_margin_blocks)
                s_obs = raw[sub_idx]
                s_init = field.current_sdf()[sub_idx]
                del coords, raw, newly                  # low-memory solve (recomputed below)
                w = field.weight_sum[sub_idx]
                h, p = (x.to(s_obs.dtype) for x in fusion.evidence(field, sub_idx))

                # IV. Conflict target s~ and data strength c from (s0, w, h, p)
                target, c_data = conflict_target(s_obs, w, h, p, pass_weight=cfg.pass_weight,
                                                 persistence_blocks=cfg.persistence_blocks, free_target_m=cfg.truncation_m,
                                                 free_target_mode=cfg.free_target, constant_target=cfg.constant_target)
                del s_obs, w, h, p

                # V. Regularised solve on the touched blocks (+ one fixed boundary block), warm-started from the field
                upd = interior[sub_idx]
                s, iters, max_update = damped_jacobi(field.keys[sub_idx], target, c_data, s_init, lam=cfg.lam,
                                                     iterations=cfg.iterations, damping=cfg.damping, update_mask=upd,
                                                     tolerance=cfg.solve_tol)
                del target, c_data
                coords = unpack_keys_t(field.keys)
                changed_rows = sub_idx[upd & ((s - s_init).abs() > cfg.remesh_eps_m)]
                field.solved[sub_idx[upd]] = s[upd]
                n_solve = int(len(sub_idx))
                del s, s_init, upd, sub_idx, interior
        else:
            # P2 (pure fusion): the fused field is the map, no evidence, no target, no solve
            coords = unpack_keys_t(field.keys)
            block_of = block_keys(coords, cfg.block_voxels)

        # VI. Surface extraction of the changed blocks (every extract_every_blocks blocks, and at the end)
        with timer("extract_s"):
            job = None
            if online is not None and online.incremental:
                blocks, job = online.schedule(field, coords, block_of, c, final)    # extension: blocks whose nodes moved
            else:
                blocks = mesher.schedule(block_of, touched, changed_rows, c, final)
            if online is not None:
                online.join()                           # a background extraction must finish before the store changes
            if job is not None:
                online.launch(c, blocks, coords, block_of, job, fusion.block_last_stamp_ns, row)
            else:
                if final and cfg.free_before_final:
                    fusion.release()
                    if dev.type == "cuda":
                        torch.cuda.empty_cache()
                row["region_faces"] = self._extract(coords, block_of, blocks, final)
                if online is not None and mesher.due(c, final) and not final:
                    online.after_extract(c, blocks, fusion.block_last_stamp_ns, row)

        # Eviction (long runs): blocks far from the sensor leave device memory; they return in the final extraction
        n_evicted = 0
        if not final and len(views):
            far = mesher.evict(field, coords, block_of, views[-1][0])
            if far is not None:
                fusion.after_evict(far)
                if online is not None:
                    online.evict(far)
                n_evicted = int(far.sum())
        mesher.evict_stats["peak_resident"] = max(mesher.evict_stats["peak_resident"], len(field))

        row.update({"new_nodes": int(n_new), "field_nodes": len(field), **stats,
                    "solve_nodes": n_solve, "solve_iters": int(iters), "solve_max_update": float(max_update),
                    "changed_nodes": int(len(changed_rows)), "extract_blocks": int(len(blocks)),
                    "evicted_nodes": n_evicted, **timer.row, "block_total_s": time.time() - t0})
        row["algorithm_s"] = row["merge_s"] + row.get("votes_s", 0.0) + row.get("solve_s", 0.0)
        if dev.type == "cuda":
            row["cuda_max_allocated_gib"] = round(torch.cuda.max_memory_allocated() / 2 ** 30, 3)
        self.block_rows.append(row)
        print("[block {block}] {frames} frames, +{new_nodes} nodes ({field_nodes} total), voted {voted}/{candidates} "
              "(+{newly_voted} new), solve {solve_nodes} nodes/{solve_iters} it ({changed_nodes} moved) | "
              "{block_total_s:.2f}s".format(**row), flush=True)
        if dev.type == "cuda":
            torch.cuda.empty_cache()
        return row

    def _extract(self, coords, block_of, blocks, final: bool):
        """VI with one retry at a quarter of the cube batch when the final (non-tiled) extraction runs out of memory."""
        try:
            return self.mesher.extract(self.field, coords, block_of, blocks, final)
        except torch.OutOfMemoryError:
            if not final:
                raise
            torch.cuda.empty_cache()
            self.mesher.cube_batch = max(self.mesher.cube_batch // 4, 100_000)
            return self.mesher.extract(self.field, coords, block_of, blocks, final)


def run_tef(cfg: Config, dataset_dir, frames, output):
    started = time.time()
    dataset = DatasetLoader(Path(dataset_dir), min_range_m=cfg.min_range_m, max_range_m=cfg.max_range_m, load_images=False)
    if frames[0] < 0 or frames[-1] >= len(dataset):
        raise ValueError(f"frame indices must lie in [0, {len(dataset)})")
    mapper = TEFMapper(cfg)
    pose_fn = trajectory_pose_fn(dataset.trajectory) if cfg.deskew else None
    block_ns = int(float(cfg.block_seconds) * 1e9)
    pacer = None
    if cfg.replay_speed:                    # online extension: a scan is read only once its timestamp has arrived
        from extensions.online.replay import Pacer
        pacer = Pacer(dataset, frames, cfg.replay_speed)
    pool = ThreadPoolExecutor(max_workers=1) if cfg.prefetch and pacer is None else None
    first_stamp = last_stamp = None
    try:
        pending = pool.submit(prepare_frame, dataset, frames[0], cfg, pose_fn, mapper.input_budget) if pool else None
        for seq, index in enumerate(frames):
            if pacer is not None:
                pacer.wait(seq)
            # I. Load and preprocess the scan (deskew, input subset, neighbour table); the next one is prepared meanwhile
            frame, points, neighbours = pending.result() if pool else prepare_frame(dataset, index, cfg, pose_fn, mapper.input_budget)
            if pool and seq + 1 < len(frames):
                pending = pool.submit(prepare_frame, dataset, frames[seq + 1], cfg, pose_fn, mapper.input_budget)
            stamp = int(frame.lidar_timestamp_ns)
            first_stamp = stamp if first_stamp is None else first_stamp
            block_id = int((stamp - first_stamp) // block_ns)

            # II + III. Samples of this scan into the current temporal block (a new block first processes the last one)
            row = mapper.integrate_scan(seq, frame, block_id, points, neighbours)
            mapper.fusion.block_last_stamp_ns = stamp
            last_stamp = stamp
            if pacer is not None:
                pacer.done(seq, index, stamp, block_id)
            if seq % 10 == 0:
                print(f"[frame {seq}] block {block_id}: {row['block_nodes']} block nodes", flush=True)
    finally:
        if pool:
            pool.shutdown(wait=True)

    # The last block, then the final extraction over every block
    if mapper.fusion.block_field is not None:
        mapper.sampler.release()
        if torch.device(cfg.device).type == "cuda":
            torch.cuda.empty_cache()
        mapper.process_block(final=True)
    if mapper.online is not None:
        mapper.online.close()
    n_vertices, n_faces = mapper.mesher.write_ply(output)
    print(f"[mesh] {n_vertices} vertices / {n_faces} faces -> {output}", flush=True)

    frames_r, blocks_r = mapper.frame_rows, mapper.block_rows
    wall = time.time() - started
    extraction = sum(r.get("extract_s", 0.0) for r in blocks_r)
    summary = {"output": str(output), "frames": len(frames_r), "blocks": len(blocks_r), "wall_s": wall,
               "algorithm_s": wall - extraction, "extraction_s": extraction,
               "data_span_s": (last_stamp - first_stamp) * 1e-9 if last_stamp is not None else 0.0,
               "field_nodes": len(mapper.field), "vertices": int(n_vertices), "faces": int(n_faces),
               "device": cfg.device, "input_budget": mapper.input_budget, "local_support": mapper.sampler.stats,
               "eviction": mapper.mesher.evict_stats, "config": cfg.as_dict(),
               "per_frame_ms": {k: 1000 * float(np.mean([r[k] for r in frames_r])) for k in
                                ("support_s", "sample_s", "splat_s", "frame_total_s")} if frames_r else {},
               "per_block_s": {k: float(np.mean([r.get(k, 0.0) for r in blocks_r])) for k in
                               ("merge_s", "votes_s", "solve_s", "extract_s", "algorithm_s", "block_total_s")} if blocks_r else {},
               "blocks_detail": blocks_r}
    if mapper.fusion.units is not None:
        summary["evidence_units"] = mapper.fusion.units.summary()
    if pacer is not None:
        from extensions.online.replay import latency_summary
        summary["online"] = pacer.record
        summary["latency"] = latency_summary(summary)
        print(f"[online] {json.dumps(summary['latency'])}", flush=True)
    if cfg.timing_json:
        path = Path(cfg.timing_json)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description="TEF mapping (given poses) -> triangle mesh")
    parser.add_argument("config", nargs="+", help="config/tef_t2.yaml or config/tef_p2.yaml, then optional extension overlays "
                                                  "(config/extensions/*.yaml; later files override earlier ones)")
    parser.add_argument("--dataset", required=True, help="canonical dataset directory")
    parser.add_argument("--frames", required=True, help="train:START-STOP (excludes held-out frames, index %% 10 == 5), all:START-STOP or indices")
    parser.add_argument("--output", required=True, help="output mesh (.ply)")
    parser.add_argument("--device", choices=("cuda", "cpu"), default=None, help="override the config device")
    parser.add_argument("--timing-json", default=None, help="write per-frame / per-block timings and the run summary")
    parser.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE", help="override config keys (ablations)")
    args = parser.parse_args(argv)
    overrides = list(args.set) + ([f"device={args.device}"] if args.device else []) + ([f"timing_json={args.timing_json}"] if args.timing_json else [])
    cfg = Config.load(args.config, overrides)
    if torch.device(cfg.device).type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA is not available; pass --device cpu")
    return run_tef(cfg, args.dataset, resolve_frames(args.frames), args.output)


if __name__ == "__main__":
    main()
