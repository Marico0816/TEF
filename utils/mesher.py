"""Step VI -- surface extraction.

Surface Nets on the current field (regularised value where solved, fused value elsewhere), gated by the node weight w:
  * nodes with w < w_min are ignored; a cube needs >= 3 observed corners, a sign change (an unobserved corner counts as
    positive) and a mean corner probability exp(-s^2 / 2*0.38^2) * (1 - exp(-w/2)) >= 0.1; its vertex is the mean of the
    edge crossings; every sign-changing lattice edge closes a quad of the four cubes around it (split into two triangles,
    degenerate or stretched triangles dropped);
  * faces are owned by the 64-voxel block of the node whose +axis edge produced them, so a block can be re-extracted
    (with one block of context) without touching the rest of the mesh;
  * extraction runs every ``extract_every_blocks`` temporal blocks on the blocks that changed (0 = once, at the end);
  * blocks farther than ``evict_distance_m`` from the sensor are moved out of device memory (the archive) and re-enter
    only in the final extraction, which runs in spatial tiles of at most ``extract_tile_nodes`` nodes.
Research ``mesh.py`` / ``mesh_io.py`` / the extraction part of ``fusion.py`` restricted to the paper settings (one-frame
crossings, observed-edge rule off, no per-node colours), unchanged otherwise.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from model.sparse_field import _CUBE_CORNERS, LatticeTable, block_keys, dilate_blocks, pack_keys_t, unpack_keys_t

_EDGE_CORNERS = np.array(
    [(0, 1), (2, 3), (4, 5), (6, 7), (0, 2), (1, 3), (4, 6), (5, 7), (0, 4), (1, 5), (2, 6), (3, 7)], dtype=np.int64
)
# lattice offsets of the four cubes sharing the +axis edge that starts at a node
_EDGE_CUBES = {
    0: np.array([(0, -1, -1), (0, 0, -1), (0, -1, 0), (0, 0, 0)], dtype=np.int64),
    1: np.array([(-1, 0, -1), (0, 0, -1), (-1, 0, 0), (0, 0, 0)], dtype=np.int64),
    2: np.array([(-1, -1, 0), (0, -1, 0), (-1, 0, 0), (0, 0, 0)], dtype=np.int64),
}


def _crossing_edge_cubes(table, coords, val, batch: int = 4_000_000):
    """Packed keys of every cube that has an edge with a sign change (an unobserved end counts as positive, as in the
    extraction): the four cubes around each axis edge from a negative node to an absent or non-negative neighbour."""
    dev = coords.device
    neg = torch.nonzero(val < 0.0).reshape(-1)
    parts = [torch.zeros(0, dtype=torch.int64, device=dev)]
    for axis in range(3):
        step = torch.zeros(3, dtype=torch.int64, device=dev); step[axis] = 1
        offs = torch.from_numpy(_EDGE_CUBES[axis]).to(dev)
        for sign in (1, -1):
            for s in range(0, len(neg), batch):
                r = neg[s:s + batch]
                other = coords[r] + sign * step
                nb = table.lookup(pack_keys_t(other))
                cross = (nb < 0) | ~(val[torch.where(nb >= 0, nb, torch.zeros_like(nb))] < 0.0)
                start = coords[r][cross] if sign == 1 else other[cross]
                parts.append(torch.unique(pack_keys_t((start[:, None, :] + offs[None, :, :]).reshape(-1, 3))))
    return torch.unique(torch.cat(parts))


def extract_surface_nets_t(coords, tsdf, weights, voxel_size_m: float, *, minimum_node_weight: float = 0.20,
                           minimum_observed_corners: int = 3, minimum_cube_probability: float = 0.10,
                           maximum_edge_factor: float = 3.0, device: str = "cuda", cube_batch: int = 4_000_000,
                           cube_candidates: str = "nodes", return_tensors: bool = False):
    """Return ``(vertices (V,3) float64, faces (F,3) int32, confidence (V,) float32, owner (F,) int64)`` -- ``owner`` is the
    packed key of the node whose +axis edge produced each face.

    ``cube_candidates="edges"`` (online extension) builds the candidate cubes from the edges with a sign change instead of
    from every cube touching a near node and applies the near-corner condition per cube: only cubes with a crossing edge
    can produce a vertex, so the mesh is identical while far fewer cubes are evaluated.  ``return_tensors`` keeps the
    arrays on the device (float64 vertices, int64 faces, float32 confidence, int64 owners)."""

    dev = torch.device(device)
    coords = coords.to(dev, torch.int64).reshape(-1, 3)
    val = tsdf.to(dev, torch.float64)
    wgt = weights.to(dev, torch.float64)
    keep = wgt >= float(minimum_node_weight)
    coords, val, wgt = coords[keep], val[keep], wgt[keep]
    empty = (np.empty((0, 3), np.float64), np.empty((0, 3), np.int32), np.empty(0, np.float32), np.empty(0, np.int64))
    if len(coords) == 0:
        return empty
    table = LatticeTable(coords, device)
    corners = torch.from_numpy(_CUBE_CORNERS).to(dev)  # (8,3)
    near = val.abs() <= 1.0
    near_coords = coords[near]
    if len(near_coords) == 0:
        return empty
    if cube_candidates not in ("nodes", "edges"):
        raise ValueError("cube_candidates must be 'nodes' or 'edges'")
    edge_mode = cube_candidates == "edges"
    if edge_mode:
        cubes = _crossing_edge_cubes(table, coords, val)
    else:
        cubes = torch.unique(pack_keys_t((near_coords[:, None, :] - corners[None, :, :]).reshape(-1, 3)))
    cube_xyz = unpack_keys_t(cubes)  # (M,3)
    edges = torch.from_numpy(_EDGE_CORNERS).to(dev)
    corner_f = corners.to(torch.float64)

    vert_parts, conf_parts, cube_parts = [], [], []
    for start in range(0, len(cubes), cube_batch):
        cx = cube_xyz[start:start + cube_batch]
        m = len(cx)
        corner_keys = pack_keys_t((cx[:, None, :] + corners[None, :, :]).reshape(-1, 3))
        row = table.lookup(corner_keys).reshape(m, 8)
        observed = row >= 0
        safe = torch.where(observed, row, torch.zeros_like(row))
        cval = torch.where(observed, val[safe], torch.ones_like(safe, dtype=torch.float64))
        cw = torch.where(observed, wgt[safe], torch.zeros_like(safe, dtype=torch.float64))
        ok = observed.sum(dim=1) >= int(minimum_observed_corners)
        if edge_mode:
            ok &= (observed & (cval.abs() <= 1.0)).any(dim=1)          # the node-mode candidate condition
        neg = cval < 0.0
        ok &= neg.any(dim=1) & (~neg).any(dim=1)
        # edge crossings
        va = cval[:, edges[:, 0]]; vb = cval[:, edges[:, 1]]  # (m,12)
        cross = (va < 0.0) != (vb < 0.0)
        den = va - vb
        tt = torch.where(den.abs() < 1e-12, torch.full_like(va, 0.5), torch.clamp(va / torch.where(den.abs() < 1e-12, torch.ones_like(den), den), 0.0, 1.0))
        pa = corner_f[edges[:, 0]]; pb = corner_f[edges[:, 1]]  # (12,3)
        inter = pa[None, :, :] + tt[:, :, None] * (pb - pa)[None, :, :]  # (m,12,3)
        n_cross = cross.sum(dim=1)
        ok &= n_cross > 0
        # cube probability over observed corners
        prob = torch.exp(-0.5 * (cval / 0.38) ** 2) * (1.0 - torch.exp(-cw / 2.0))
        prob = (prob * observed).sum(dim=1) / observed.sum(dim=1).clamp_min(1).to(torch.float64)
        ok &= prob >= float(minimum_cube_probability)
        crossf = cross.to(torch.float64)
        local = (inter * crossf[:, :, None]).sum(dim=1) / n_cross.clamp_min(1).to(torch.float64)[:, None]
        vert_parts.append(((cx.to(torch.float64) + local) * float(voxel_size_m))[ok])
        conf_parts.append(prob[ok])
        cube_parts.append(cubes[start:start + cube_batch][ok])
    vertices = torch.cat(vert_parts)
    confidence = torch.cat(conf_parts)
    cube_keys = torch.cat(cube_parts)
    if len(vertices) == 0:
        return empty
    cube_table = LatticeTable(unpack_keys_t(cube_keys), device)  # rows = vertex ids

    faces_parts, owner_parts = [], []
    maximum_edge = float(maximum_edge_factor) * float(voxel_size_m)
    packed_nodes = pack_keys_t(coords)
    for axis in range(3):
        step = torch.zeros(3, dtype=torch.int64, device=dev); step[axis] = 1
        nb = table.lookup(pack_keys_t(coords + step[None, :]))
        has = nb >= 0
        nb_safe = torch.where(has, nb, torch.zeros_like(nb))
        sign_change = has & ((val < 0.0) != (val[nb_safe] < 0.0))
        idx = torch.nonzero(sign_change).reshape(-1)
        if len(idx) == 0:
            continue
        base = coords[idx]  # (E,3)
        offs = torch.from_numpy(_EDGE_CUBES[axis]).to(dev)  # (4,3)
        quad = cube_table.lookup(pack_keys_t((base[:, None, :] + offs[None, :, :]).reshape(-1, 3))).reshape(-1, 4)
        good = (quad >= 0).all(dim=1)
        quad = quad[good]
        owner = packed_nodes[idx][good]
        a, b, c, d = quad[:, 0], quad[:, 1], quad[:, 2], quad[:, 3]
        for tri in ((a, b, d), (a, d, c)):
            f = torch.stack(tri, dim=1)
            p0, p1, p2 = vertices[f[:, 0]], vertices[f[:, 1]], vertices[f[:, 2]]
            e = torch.stack([(p1 - p0).norm(dim=1), (p2 - p1).norm(dim=1), (p0 - p2).norm(dim=1)], dim=1)
            area2 = torch.cross(p1 - p0, p2 - p0, dim=1).norm(dim=1)
            emax = e.max(dim=1).values; emin = e.min(dim=1).values
            keep_f = (emax <= maximum_edge) & (emin > 1e-5) & (emax / emin <= 8.0) & (area2 > 1e-8)
            faces_parts.append(f[keep_f])
            owner_parts.append(owner[keep_f])
    faces = torch.cat(faces_parts) if faces_parts else torch.zeros((0, 3), dtype=torch.int64, device=dev)
    owners = torch.cat(owner_parts) if owner_parts else torch.zeros(0, dtype=torch.int64, device=dev)
    if return_tensors:
        return vertices, faces.reshape(-1, 3), confidence.to(torch.float32), owners
    return (vertices.cpu().numpy().astype(np.float64), faces.cpu().numpy().astype(np.int32).reshape(-1, 3),
            confidence.cpu().numpy().astype(np.float32), owners.cpu().numpy())


def confidence_colors(confidence: np.ndarray) -> np.ndarray:
    """Display colours from the cube confidence (blue = low, green = high); they never affect geometry."""

    values = np.clip(np.asarray(confidence, dtype=np.float64), 0.0, 1.0)
    low = np.asarray([45.0, 74.0, 128.0])
    high = np.asarray([68.0, 204.0, 174.0])
    return np.rint(low[None, :] * (1.0 - values[:, None]) + high[None, :] * values[:, None]).astype(np.uint8)


class Mesher:
    """Block-wise mesh store: which blocks to re-extract, the extraction itself, eviction to the archive and the output."""

    def __init__(self, cfg, device: str):
        self.cfg, self.device, self.dev = cfg, device, torch.device(device)
        self.voxel = cfg.voxel_m
        self.block_voxels = int(cfg.block_voxels)
        self.extract_kw = dict(minimum_node_weight=cfg.min_node_weight, minimum_observed_corners=cfg.min_observed_corners,
                               minimum_cube_probability=cfg.min_cube_probability, maximum_edge_factor=cfg.max_edge_factor)
        self.cube_batch = int(cfg.cube_batch)
        self.blocks = {}               # block key -> (vertices, faces, confidence) of the faces the block owns
        self.pending_blocks = None     # blocks changed since their last extraction (+ evicted blocks)
        self.archive = []              # evicted nodes: packed keys, current value, weight (CPU)
        self.evict_stats = {"evicted": 0, "peak_resident": 0, "blocks": 0}

    # -- which blocks -------------------------------------------------------------------------------------------------
    def due(self, c: int, final: bool) -> bool:
        """Is block ``c`` an output time (every ``extract_every_blocks`` blocks, and the final flush)?"""
        every = int(self.cfg.extract_every_blocks)
        return final or (every > 0 and (c + 1) % every == 0)

    def schedule(self, block_of, touched_rows, changed_rows, c: int, final: bool):
        """Record the blocks touched by the merge or moved by the solve; return the blocks to extract now."""
        do_extract = self.due(c, final)
        pending = torch.unique(torch.cat([block_of[touched_rows], block_of[changed_rows]]))
        self.pending_blocks = pending if self.pending_blocks is None else torch.unique(torch.cat([self.pending_blocks, pending]))
        blocks = self.pending_blocks if do_extract else torch.zeros(0, dtype=torch.int64, device=self.dev)
        if do_extract:
            self.pending_blocks = None
        return blocks

    # -- extraction -----------------------------------------------------------------------------------------------------
    def extract(self, field, coords, block_of, blocks, final: bool):
        """Re-extract ``blocks`` (with one block of context); the final extraction also re-extracts every archived block.
        Returns the number of faces stored."""
        if len(blocks) == 0:
            return 0
        sdf_now = field.current_sdf()
        if final and self.dev.type == "cuda":
            torch.cuda.empty_cache()
        if final and (self.archive or self.cfg.tiled_final):
            return self._extract_tiled_final(blocks, coords, block_of, sdf_now, field.extraction_weight())
        ext_idx = torch.nonzero(torch.isin(block_of, dilate_blocks(blocks, self.block_voxels, 1))).reshape(-1)
        v, f, conf, owner = extract_surface_nets_t(coords[ext_idx], sdf_now[ext_idx], field.extraction_weight()[ext_idx],
                                                   self.voxel, device=self.device, cube_batch=self.cube_batch, **self.extract_kw)
        for b in blocks.cpu().numpy():
            self.blocks.pop(int(b), None)
        return self._store_block_faces(v, f, conf, owner, blocks) if len(f) else 0

    def _store_block_faces(self, v, f, conf, owner, keep_blocks):
        """Split an extraction's faces by owner block and store those owned by ``keep_blocks`` (own vertex copies per block)."""
        owner_block = block_keys(unpack_keys_t(torch.from_numpy(owner).to(self.dev)), self.block_voxels)
        keep = torch.isin(owner_block, keep_blocks.to(self.dev)).cpu().numpy()
        f_keep = f[keep]
        ob = owner_block.cpu().numpy()[keep]
        order = np.argsort(ob, kind="stable")
        f_keep, ob = f_keep[order], ob[order]
        uniq, starts = np.unique(ob, return_index=True)
        ends = np.append(starts[1:], len(ob))
        n = 0
        for b, lo, hi in zip(uniq, starts, ends):
            fb = f_keep[lo:hi]
            used, local = np.unique(fb.reshape(-1), return_inverse=True)
            self.blocks[int(b)] = (v[used], local.reshape(-1, 3).astype(np.int32), conf[used])
            n += len(fb)
        return n

    def _extract_tiled_final(self, blocks_now, coords, block_of, sdf_now, weight):
        """Final extraction over resident + archived nodes in spatial tiles of at most ``extract_tile_nodes`` nodes (context
        included); a tile that runs out of memory is split in two."""
        dev = self.dev
        A = self.archive
        if A:
            ak = torch.cat([a_["keys"] for a_ in A]); asdf = torch.cat([a_["sdf"] for a_ in A]); aw = torch.cat([a_["weight"] for a_ in A])
        else:
            ak = torch.zeros(0, dtype=torch.int64); asdf = torch.zeros(0, dtype=sdf_now.dtype); aw = torch.zeros(0, dtype=weight.dtype)
        acoords = unpack_keys_t(ak)
        ablock = block_keys(acoords, self.block_voxels)
        blocks = torch.unique(torch.cat([blocks_now.cpu(), torch.unique(ablock)]))
        for b in blocks.numpy():
            self.blocks.pop(int(b), None)
        rb_cpu = block_of.cpu()
        cnt = torch.zeros(len(blocks), dtype=torch.int64)
        for u_, c_ in (torch.unique(rb_cpu, return_counts=True), torch.unique(ablock, return_counts=True)):
            pos = torch.searchsorted(blocks, u_).clamp(max=len(blocks) - 1)
            ok = blocks[pos] == u_
            cnt.index_add_(0, pos[ok], c_[ok])
        cap = int(self.cfg.extract_tile_nodes)
        tile_id = torch.div(torch.cumsum(cnt, 0) - 1, max(cap // 3, 1), rounding_mode="floor")
        n_faces = n_tiles = 0

        def run(G):
            nonlocal n_faces, n_tiles
            ctx = dilate_blocks(G.to(dev), self.block_voxels, 1)
            r_idx = torch.nonzero(torch.isin(block_of, ctx)).reshape(-1)
            a_sel = torch.isin(ablock, ctx.cpu())
            xc = torch.cat([coords[r_idx], acoords[a_sel].to(dev, coords.dtype)])
            xs = torch.cat([sdf_now[r_idx], asdf[a_sel].to(dev, sdf_now.dtype)])
            xw = torch.cat([weight[r_idx], aw[a_sel].to(dev, weight.dtype)])
            try:
                v, f, conf, owner = extract_surface_nets_t(xc, xs, xw, self.voxel, device=self.device, cube_batch=self.cube_batch, **self.extract_kw)
            except torch.OutOfMemoryError:
                if len(G) < 2:
                    raise
                v = None
            del xc, xs, xw
            if dev.type == "cuda":
                torch.cuda.empty_cache()
            if v is None:                 # out of memory: split the tile in two
                h = len(G) // 2
                run(G[:h])
                run(G[h:])
                return
            n_tiles += 1
            if len(f):
                n_faces += self._store_block_faces(v, f, conf, owner, G)

        for t_ in torch.unique(tile_id):
            run(blocks[tile_id == t_])
        print(f"[mesh] final extraction: {len(ak)} archived + {len(block_of)} resident nodes in {n_tiles} tiles, {n_faces} faces", flush=True)
        self.archive = []
        return n_faces

    # -- eviction -------------------------------------------------------------------------------------------------------
    def evict(self, field, coords, block_of, sensor_origin):
        """Move the nodes of blocks farther than ``evict_distance_m`` from the sensor to the CPU archive; their blocks are
        re-extracted at the end.  Returns the boolean row mask of the evicted nodes (None when nothing was evicted)."""
        D = float(self.cfg.evict_distance_m or 0.0)
        if D <= 0 or len(field) == 0:
            return None
        p_s = torch.as_tensor(np.asarray(sensor_origin, dtype=np.float64).reshape(3), device=self.dev)
        B = self.block_voxels
        centre = (torch.div(coords, B, rounding_mode="floor").to(torch.float64) + 0.5) * (B * self.voxel)
        far = torch.linalg.norm(centre - p_s[None, :], dim=1) > D
        if not bool(far.any()):
            return None
        far_blocks = torch.unique(block_of[far])
        self.pending_blocks = far_blocks if self.pending_blocks is None else torch.unique(torch.cat([self.pending_blocks, far_blocks]))
        out = field.evict(far)
        self.archive.append({"keys": out["keys"], "sdf": out["sdf"], "weight": out["weight"]})
        self.evict_stats["evicted"] += int(far.sum())
        self.evict_stats["blocks"] += 1
        if self.dev.type == "cuda":
            torch.cuda.empty_cache()
        return far

    # -- output ---------------------------------------------------------------------------------------------------------
    def assemble(self):
        """(vertices, faces, confidence) of the whole mesh, blocks in ascending key order."""
        if not self.blocks:
            return np.zeros((0, 3)), np.zeros((0, 3), np.int32), np.zeros(0, np.float32)
        vs, fs, cs, offset = [], [], [], 0
        for b in sorted(self.blocks):
            v, f, c = self.blocks[b]
            vs.append(v); fs.append(f + offset); cs.append(c)
            offset += len(v)
        return np.concatenate(vs), np.concatenate(fs).astype(np.int32), np.concatenate(cs)

    def write_ply(self, path):
        """Binary PLY with confidence vertex colours (Open3D writer, as the research pipeline)."""
        import open3d as o3d
        v, f, conf = self.assemble()
        mesh = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(v), o3d.utility.Vector3iVector(f.astype(np.int32)))
        mesh.vertex_colors = o3d.utility.Vector3dVector(confidence_colors(conf).astype(np.float64) / 255.0)
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        if not o3d.io.write_triangle_mesh(str(path), mesh, write_ascii=False):
            raise RuntimeError(f"could not write mesh: {path}")
        return len(v), len(f)
