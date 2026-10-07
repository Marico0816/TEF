"""Configuration: one flat set of parameters, read from a sectioned YAML file (``config/*.yaml``) plus ``--set`` overrides.

The YAML sections only group the keys by pipeline step; every key is unique across sections.  ``truncation_m`` is
derived (``truncation_factor * voxel_m``).  Defaults are the frozen T2 values; ``config/tef_p2.yaml`` sets the P2 control.
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field

import yaml

SECTIONS = ("data", "preprocess", "sampler", "fusion", "conflict", "solver", "mesher", "online", "output")


@dataclass
class Config:
    # -- data -----------------------------------------------------------------------------------------------------------
    min_range_m: float = 0.5                  # returns outside [min, max] range are dropped when a scan is read
    max_range_m: float = 50.0
    device: str = "cuda"

    # -- I. preprocessing -------------------------------------------------------------------------------------------------
    deskew: bool = True                       # motion-compensate each return to the scan time (no-op without point times)
    subsample_first: bool = True              # keep the deterministic 12 000-return subset of the shared input protocol
    max_rays_per_frame: int = 12000
    input_keep_mask: str | None = None        # DIR of per-frame bool masks from an external filter (control experiment)
    prefetch: bool = True                     # read + preprocess the next scan on a CPU thread

    # -- II. local support and samples --------------------------------------------------------------------------------
    voxel_m: float = 0.04                     # lattice spacing
    truncation_factor: float = 2.0            # truncation f = truncation_factor * voxel_m (0.08 m)
    samples_per_ray: int = 5                  # along-ray samples per return
    normal_projected_sdf: bool = True         # signed distance measured along the local normal
    local_support: bool = True                # multi-scale running PCA frame; False = default frame for every return
    local_support_voxels_m: list = field(default_factory=lambda: [0.25, 0.5, 1.0])
    local_support_min_points: int = 30
    local_support_max_planarity: float = 0.15
    normal_truncation_sigma: float = 3.0      # per-return truncation = sigma * tau for returns with a PCA frame
    per_return_truncation: bool = True        # False = the constant truncation for every return
    footprint_extent_sigma: float = 1.0       # lateral disc radius in units of the tangential scale
    max_footprint_samples_per_ray: int = 25   # 1 = no lateral samples (on-ray samples only)
    max_sample_step_factor: float = 1.0
    data_support_limit: bool = True           # clip the lateral disc to the scan's own neighbouring returns
    support_neighbors: int = 32               # k-NN of the data-support limits
    match_maximum_normal_angle_deg: float = 55.0
    support_normal_offset_sigma: float = 2.5
    support_maximum_normal_slope: float = 0.35
    support_connectivity_factor: float = 2.5
    support_boundary_margin_factor: float = 0.5

    # -- III. temporal blocks --------------------------------------------------------------------------------------------
    block_seconds: float = 1.0                # temporal block length
    block_weight: str = "max"                 # max = bounded block weight (T2); sum = sample-weighted fusion (P2)
    count_unit: str = "block"                 # block (paper) | frame | ray (counting-unit control)
    free_space_ray_radius_factor: float = 1.5
    free_space_endpoint_clearance_factor: float = 2.5
    free_space_min_rays_per_frame: int = 1

    # -- IV. conflict target ---------------------------------------------------------------------------------------------
    evidence: bool = True                     # False = pure fusion (P2): no pass votes, no target, no solve
    free_target: str = "mirror"               # mirror (T2) | truncation | constant
    constant_target: float = 1.0              # free-space value of the constant target (normalised units)
    pass_weight: float = 1.0                  # beta
    persistence_blocks: int = 3               # H

    # -- V. regularised solve ----------------------------------------------------------------------------------------------
    lam: float = 0.3                          # lambda
    iterations: int = 30
    damping: float = 0.8
    solve_tol: float = 1e-4
    block_voxels: int = 64                    # block size (solve regions, extraction, eviction)
    solve_margin_blocks: int = 1

    # -- VI. extraction ---------------------------------------------------------------------------------------------------
    min_node_weight: float = 0.10             # w_min of the extraction gate
    min_observed_corners: int = 3
    min_cube_probability: float = 0.1
    max_edge_factor: float = 3.0
    cube_batch: int = 500000
    remesh_eps_m: float = 1e-4                # a solved node moving more than this marks its block for re-extraction
    extract_every_blocks: int = 0             # 0 = extract once at the end
    tiled_final: bool = True
    extract_tile_nodes: int = 20000000
    free_before_final: bool = True            # release solver / evidence state before the final extraction
    evict_distance_m: float = 90.0            # 0 = keep every node resident

    # -- optional extension: online output (extensions/online; all off in the paper configuration) -------------------
    incremental_remesh: bool = False          # re-extract only blocks whose nodes moved since their meshes were extracted
    remesh_reuse_tol: float = 0.0125          # value change (normalised units) that dirties a node
    remesh_gate_tol: float = 0.05             # change of the weight factor 1 - exp(-w/2) that dirties a node
    background_remesh: bool = False           # run that extraction on a worker thread / CUDA stream
    region_candidates: bool = False           # pass-vote candidates only in blocks the rays can reach (same votes)
    mesh_delta_dir: str | None = None         # write one mesh delta file per output
    replay_speed: float | None = None         # read each scan only when its timestamp has arrived (x real time)

    # -- output -----------------------------------------------------------------------------------------------------------
    timing_json: str | None = None

    @property
    def truncation_m(self) -> float:
        return float(self.truncation_factor) * float(self.voxel_m)

    def validate(self) -> "Config":
        if self.block_weight not in ("max", "sum"):
            raise ValueError("block_weight must be max or sum")
        if self.count_unit not in ("block", "frame", "ray"):
            raise ValueError("count_unit must be block, frame or ray")
        if self.free_target not in ("mirror", "truncation", "constant"):
            raise ValueError("free_target must be mirror, truncation or constant")
        if self.count_unit != "block" and not self.evidence:
            raise ValueError("count_unit frame|ray needs the evidence path (evidence: true)")
        for name in ("voxel_m", "truncation_factor", "block_seconds", "normal_truncation_sigma", "max_rays_per_frame"):
            if not getattr(self, name) > 0:
                raise ValueError(f"{name} must be positive")
        if self.input_keep_mask and not self.subsample_first:
            raise ValueError("input_keep_mask needs subsample_first")
        if self.background_remesh and not self.incremental_remesh:
            raise ValueError("background_remesh needs incremental_remesh")
        if self.replay_speed is not None and not self.replay_speed > 0:
            raise ValueError("replay_speed must be positive")
        return self

    # -- loading ------------------------------------------------------------------------------------------------------------
    @classmethod
    def load(cls, paths, overrides=()) -> "Config":
        """``paths``: one YAML file or a list; later files override earlier ones (base config + extension overlays)."""
        cfg = cls()
        known = {f_.name for f_ in dataclasses.fields(cls)}
        for path in ([paths] if isinstance(paths, (str, bytes)) or hasattr(paths, "__fspath__") else list(paths)):
            with open(path) as f:
                doc = yaml.safe_load(f) or {}
            for section, values in doc.items():
                if section not in SECTIONS:
                    raise ValueError(f"{path}: unknown section '{section}' (expected one of {', '.join(SECTIONS)})")
                for key, value in (values or {}).items():
                    if key not in known:
                        raise ValueError(f"{path}: unknown key '{section}.{key}'")
                    setattr(cfg, key, value)
        for item in overrides:
            key, _, text = item.partition("=")
            if key not in known or not _:
                raise ValueError(f"--set expects KEY=VALUE with a known key, got '{item}'")
            setattr(cfg, key, yaml.safe_load(text))
        return cfg.validate()

    def as_dict(self) -> dict:
        out = dataclasses.asdict(self)
        out["truncation_m"] = self.truncation_m
        return out
