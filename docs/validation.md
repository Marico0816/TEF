# Validation of the step-organised implementation

This page summarizes the completed local equivalence checks of 2026-10-06/07.
They compare the reorganised implementation with the research implementation;
they do not measure a new algorithm or establish superiority over other methods.
The source reports and their hashes are recorded in `../publication_manifest.json`.

## Core mapper and optional extensions

| Check | Recorded result |
|---|---|
| Core CPU: 13 cases on two datasets, 54 selected frames each | 26/26: byte-identical PLY files and identical per-block statistics |
| Core deterministic GPU: development segment and four paper segments, T2 and P2 | 10/10: byte-identical meshes and identical per-block statistics |
| Online CPU: six cases on two datasets | 12/12: identical final meshes, block statistics and mesh-delta arrays |
| Online deterministic GPU: DEV T2, DEV P2 and Keble T2 | 3/3: identical final results and all 59 delta files per case |
| Dynamic 4D: KITTI 0059 target 51, three settings | Matching 100 frame rows and 22 mesh revisions per setting |
| Dynamic 4D bag: image-enabled setting | All 621 messages match after decoding, with timing/output-path fields excluded |
| Reference and ray scorers on the checked Keble T2 mesh | Nine reported metrics match exactly |

The dynamic settings were image features with bag export, images off with
tracking only, and a 25-degree heading guard with tracking only. The guard
branch ran but no rejection event occurred. This is a single-clip check, not
validation across diverse object categories or motions. Raw bag bytes need not
match because serialization padding may differ.

CPU core and online checks total 38 cases; deterministic GPU core and online
checks total 13. Earlier repetitions of the same cases should not be counted
as additional independent evidence.

## Default GPU mode

Default GPU scatter-add operations are nondeterministic. Under the predeclared
comparison tolerances against the frozen paper runs, 7/8 runs passed. The
Newer College S2 P2 run failed two checks:

- recall at 10 cm: -0.11 percentage points, outside the +/-0.05 tolerance;
- completeness: +0.90 cm, outside the +/-0.72 cm tolerance.

Follow-up scoring with seeds 0-3 found overlapping ranges between the clean,
frozen and earlier research meshes. Deterministic mapping produced identical
meshes. These checks support the numerical-equivalence interpretation but do
not retroactively turn the original 7/8 result into 8/8.

## Recorded timestamp-paced replay

Each case contains 60 seconds of sensor data and 59 mesh outputs. These are
descriptive results from the development workstation, not universal latency
guarantees.

| Segment | Median latency | P95 latency | Maximum latency |
|---|---:|---:|---:|
| Newer College development | 0.33 s | 0.76 s | 0.83 s |
| Spires Keble | 0.58 s | 0.64 s | 0.67 s |
| Spires Observatory | 0.63 s | 0.82 s | 0.85 s |

The research records show increasing latency on longer sequences. These short
replays do not establish real-time performance for arbitrary map sizes.

## Reproduction

The standalone synthetic demo requires no downloaded data:

```bash
python scripts/demo_tiny_scene.py --out outputs/demo --method t2 --device cpu
```

The research-comparison scripts require the separate research checkout and
prepared datasets. The dynamic extension additionally requires the front-end
inputs described in [INPUTS.md](../extensions/dynamic4d/INPUTS.md).

Large datasets, reference scans, experiment output meshes, bag files and cached
object inputs are not included in this source publication.
