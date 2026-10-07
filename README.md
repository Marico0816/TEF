# TEF: Temporal Evidence Fusion for Given-Pose LiDAR Surface Reconstruction

[Project page](https://marico0816.github.io/tef-project/) | [Validation](docs/validation.md) | [Publication notes](docs/publication.md) | [License](LICENSE.md)

TEF fuses LiDAR scans with known sensor poses into a triangle mesh. Scans are grouped into 1 s temporal blocks. Within a block, ray and lateral samples build a block-local signed-distance field. Each finished block is merged into a persistent sparse field, where it counts as surface-hit and free-space-pass evidence. Where the two kinds of evidence conflict, they set the target of a local regularised solve. Surface Nets then extracts the mesh. TEF does not train a network or estimate poses.

This repository is the step-organised implementation of the paper configuration. Each of the six steps in the method figure lives in its own module. Its output is byte-identical to the research implementation on the CPU and in deterministic GPU mode: see [Equivalence with the research code](#equivalence-with-the-research-code).

## Pipeline

| Step | What it does | Code |
|---|---|---|
| I. Scan preprocessing | Deskews each return to the scan time with the given trajectory. Keeps the deterministic 12 000-return subset shared by every method. Builds the scan's neighbour table. | [`utils/preprocess.py`](utils/preprocess.py) |
| II. Local support and samples | Updates multi-scale running PCA statistics (0.25 / 0.5 / 1.0 m). Queries a local frame per return, falling back to a ray-orthogonal default frame. Generates along-ray samples with normal-projected signed distances, plus a lateral tangent disc clipped to the scan's own support. | [`model/local_support.py`](model/local_support.py), [`utils/sampler.py`](utils/sampler.py) |
| III. Temporal-block merge | Splats each scan into the block-local field. When a block ends, the field is merged with a bounded block weight, so each node gets at most one hit vote per block. Nodes inside the fused surface that the block's rays traverse get at most one pass vote per block. | [`utils/block_fusion.py`](utils/block_fusion.py), [`model/sparse_field.py`](model/sparse_field.py) |
| IV. Conflict target | Forms c_hit = h/H and c_pass = βp/H. A mirror target on the interior side of the fused field supports the surface where h > βp and free space where h < βp. | [`utils/conflict_target.py`](utils/conflict_target.py) |
| V. Regularised solve | Runs damped Jacobi on the blocks the merge touched (plus one fixed boundary block), with a degree-normalised six-neighbour smoothness term, warm-started from the field. | [`utils/regularizer.py`](utils/regularizer.py) |
| VI. Surface extraction | Runs Surface Nets on the current field with weight and cube-probability gates. Faces are owned by 64-voxel blocks. Extraction is periodic or final and tiled. Far blocks are evicted to the CPU and re-enter the final extraction. | [`utils/mesher.py`](utils/mesher.py) |

[`tef_mapping.py`](tef_mapping.py) is the main loop. Steps I–III run per scan. When a new block starts, the previous block goes through III–VI. The section comments in the loop follow the step numbers above.

## Installation

Clone this repository and run the commands below from its root:

```bash
git clone https://github.com/Marico0816/TEF.git
cd TEF
```

Python 3.12 is recommended; the recorded runs use the versions in [`requirements.txt`](requirements.txt).

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt   # use a compatible CUDA build of PyTorch for GPU mapping
```

The core mapper and demo need no ROS installation. The Newer College bag converter additionally needs `rosbags` (`pip install rosbags`). The optional 4D extension uses OpenCV, Pillow and ROS 2 Jazzy; see [its input and dependency guide](extensions/dynamic4d/INPUTS.md).

### Sanity test (no dataset needed)

```bash
python scripts/demo_tiny_scene.py --out outputs/demo --method t2 --device cpu
```

The demo writes 12 synthetic scans of a floor inside a cylindrical wall and maps them. It then checks the floor and wall errors of the mesh (T2: 289 468 vertices, floor 1.24 cm, wall 0.09 cm; the research implementation writes the same PLY byte for byte).

## Data

The input is a canonical sequence directory with these parts:

- `manifest.jsonl`: one line per frame;
- per-frame point clouds (binary PCD with per-point time offsets);
- `calibration.json` with `imu_from_lidar`;
- `trajectory.txt`: the given poses, TUM format;
- `dataset.json`.

The readers are in [`dataset/`](dataset/). The converters for the paper's datasets are in [`dataset/converter/`](dataset/converter/):

```bash
python dataset/converter/export_newer_college.py --help   # Newer College 2020 (Ouster OS1-64)
python dataset/converter/export_oxford_spires.py --help   # Oxford Spires
python dataset/converter/export_common_input.py --help    # the shared deskewed 12 000-return input for baselines
```

## Run

```bash
# T2: the paper configuration
python tef_mapping.py config/tef_t2.yaml --dataset /path/to/canonical --frames train:0-600 --output outputs/T2.ply

# P2: the shared-frontend pure-fusion control (steps IV-V off, sample-weighted blocks)
python tef_mapping.py config/tef_p2.yaml --dataset /path/to/canonical --frames train:0-600 --output outputs/P2.ply
```

- `train:0-600` selects frames 0–599 except the held-out frames (index % 10 == 5).
- `--timing-json` writes per-frame and per-block timings and the run summary.
- `--device cpu` runs everything on the CPU, which is slow but exact.

### Ablations

Every ablation in the paper is a config key, set with `--set KEY=VALUE`:

| Ablation | Override |
|---|---|
| Default frame for every return (no PCA frame) | `local_support=false` |
| No lateral samples (on-ray samples only) | `max_footprint_samples_per_ray=1` |
| Constant truncation instead of per-return truncation | `per_return_truncation=false` |
| Sample-weighted blocks instead of the bounded block weight | `block_weight=sum` |
| No pass evidence | `pass_weight=0` |
| Constant free-space target (carving control) | `free_target=constant constant_target=1.0` |
| Counting unit frame / ray instead of block | `count_unit=frame` / `count_unit=ray` |
| Extraction weight threshold | `min_node_weight=0.2` |
| Keep every node resident / extract without tiles | `evict_distance_m=0` / `tiled_final=false` |

## Evaluation

The scripts in [`eval/`](eval/) are the paper's scorers; only their import paths changed.

```bash
# accuracy, completeness, F-score and recall against the reference surface (fixed mask)
python eval/evaluate_mesh_gt.py outputs/T2.ply --gt /path/to/reference.ply --dataset /path/to/canonical \
  --frames 0-600:skip5mod10 --mask-cache outputs/mask.npz --region-erode 1.0 --output outputs/T2.gt.json
# free / agree / missing fractions of the held-out rays
python eval/evaluate_mesh_fair.py outputs/T2.ply /path/to/canonical --train-start 0 --heldout-start 5 --stop 600 \
  --deskew-heldout --skip-mesh-components --output outputs/T2.rays.json
# paired 10 m block bootstrap between two meshes
python eval/paired_gt_bootstrap.py --help
```

## Optional extensions

Two optional extensions are available. Both are off in the paper configuration and are not even imported unless enabled. Each is enabled by its own overlay config or entry point.

| Extension | What it adds | Enable | Extra dependencies |
|---|---|---|---|
| Online output ([`extensions/online/`](extensions/online/)) | One mesh output per temporal block. Only blocks whose nodes moved are re-extracted (incremental remeshing), on a background thread. Vote candidates are limited to the blocks the rays reach (same votes). Writes a mesh delta file per output; paced replay reports output latency. | overlay [`config/extensions/online.yaml`](config/extensions/online.yaml) | none |
| Dynamic objects, 4D ([`extensions/dynamic4d/`](extensions/dynamic4d/)) | Tracks one moving target with an image-refined 4-DoF pose. Builds the target's object-frame TSDF mesh and exports a ROS 2 bag for RViz. | `python reconstruct4d.py --config config/extensions/dynamic4d_*.json` | OpenCV for image refinement; ROS 2 Jazzy and Pillow for the bag |

```bash
# online output at the recorded scan rate, with per-block mesh deltas and a latency report
python tef_mapping.py config/tef_t2.yaml config/extensions/online.yaml --dataset DIR --frames train:0-600 \
    --output outputs/T2_online.ply --set replay_speed=1.0 mesh_delta_dir=outputs/T2_online_deltas
# dynamic objects: track, fuse and export a bag (inputs: extensions/dynamic4d/INPUTS.md)
source /opt/ros/jazzy/setup.bash
python reconstruct4d.py --config config/extensions/dynamic4d_kitti0059.json --output outputs/kitti0059_t51
```

What each extension does and does not establish:

- **Online output:**
  - The final mesh is the paper-path final mesh, because the final extraction re-extracts every block.
  - On the Newer College development segment (60 s, given poses, 12 000 returns per scan), output latency had a median of 0.33 s and a P95 of 0.76 s in the recorded 2026-10-07 extension check.
  - Latency grows with the map. The research records show it above 1 s on longer sequences, so a 1 s output is not established in general.
- **Dynamic objects:** this extension needs a front end that is not included, namely object tracks, per-return ownership and semantic image masks; see [INPUTS.md](extensions/dynamic4d/INPUTS.md). It was evaluated on single clips only.

## Equivalence with the research code

[`tests/compare_research.py`](tests/compare_research.py) runs the research mapper (frozen paper flags plus formal-run flags) and this repository on the same frames, then compares the meshes and the per-block counts.

```bash
python tests/compare_research.py --research-root /path/to/research-checkout --case t2 \
    --dataset /path/to/canonical --frames train:0-60 --device cpu
```

These comparison scripts require a separate research checkout and prepared data; they are not standalone unit tests. Use the synthetic demo above for a dataset-free check.

The cases cover T2, P2, every ablation above, the eviction / tiled / periodic-extraction path and the online extension. The verification of 2026-10-06/07 found:

- **CPU:** 26 / 26 runs give byte-identical PLY files and identical per-block counts. That is 13 cases on 54 frames each of Newer College and Oxford Spires Keble.
- **GPU with `--deterministic`** (`torch.use_deterministic_algorithms`): byte-identical meshes on the full paper segments for T2 and P2 (10 / 10 runs), on the development segment and the four paper segments.
- **GPU in default mode:** the scatter-adds use float32 / float64 atomics, so repeated runs of either implementation differ by a few to about a hundred vertices. Scored against the frozen paper runs with the earlier reproducibility tolerances, 7 / 8 runs pass. The eighth (Newer College S2, P2) failed the predeclared recall and completeness tolerances. Follow-up evaluation with seeds 0–3 found overlapping ranges, and deterministic mapping was byte-identical; these diagnostics do not change the original 7 / 8 pass count.
- **Scorers:** the scorers in `eval/` match the paper's scorers exactly.
- **Online extension:** compared against the research flags (`--incremental-remesh --region-candidates --background-remesh --mesh-delta-dir`).
  - CPU: 12 / 12 runs give byte-identical final PLY files and identical per-block counts, and every mesh delta file has identical arrays.
  - GPU with `--deterministic`: 3 / 3 runs are identical, each including all 59 delta files.
- **Dynamic extension:** [`tests/compare_dynamic4d.py`](tests/compare_dynamic4d.py) checks three settings on the KITTI 0059 case: image features with the bag, images off, and the heading guard. All three match the research implementation after excluding timing and output-path fields. Each contains 100 frame rows and 22 mesh revisions; the image-enabled bag case also matches all 621 decoded messages. See [validation details](docs/validation.md).

Not included in this publication:

- learned fusion and normal diagnostics;
- research sample replay;
- the research GPU / CPU / disk tiered-storage backend;
- pose perturbation;
- colour accumulators;
- the observed-edge and multi-frame crossing extraction rules.

Incremental remeshing and timestamp-paced replay are included in the optional online extension. The core still archives distant blocks to CPU memory for final extraction; this is separate from the omitted disk-paging backend.

## Repository layout

```text
TEF/
├── tef_mapping.py          # main loop: I-III per scan, III-VI per temporal block
├── reconstruct4d.py        # optional extension: dynamic objects (4D)
├── config/                 # tef_t2.yaml (paper), tef_p2.yaml (pure-fusion control)
│   └── extensions/         # online.yaml (overlay), dynamic4d_*.json (4D cases)
├── dataset/                # canonical sequence readers, trajectory interpolation and deskew
│   └── converter/          # Newer College / Oxford Spires converters, shared baseline input
├── model/
│   ├── sparse_field.py     # sparse lattice field: packed keys, merge, bounded block weight, eviction
│   └── local_support.py    # multi-scale running PCA (local frames)
├── utils/
│   ├── preprocess.py       # I
│   ├── sampler.py          # II
│   ├── block_fusion.py     # III
│   ├── conflict_target.py  # IV
│   ├── regularizer.py      # V
│   ├── mesher.py           # VI
│   ├── config.py           # YAML config + --set overrides
│   └── tools.py            # frame selection, timers
├── extensions/             # optional, off in the paper configuration
│   ├── online/             # incremental / background remeshing, mesh deltas, paced replay
│   └── dynamic4d/          # target tracking, object TSDF, ROS 2 bag (INPUTS.md: inputs and formats)
├── eval/                   # paper scorers (reference surface, held-out rays, paired bootstrap)
├── scripts/                # demo_tiny_scene.py: synthetic sanity test
└── tests/                  # equivalence checks against the research implementation (mapper, 4D)
```

## Citation and license

Formal citation and the public paper URL will be added when available. No open-source reuse license has been selected; the existing publication notice is retained in [LICENSE.md](LICENSE.md). Dataset and external input terms continue to apply.
