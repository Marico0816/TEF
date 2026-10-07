# Source publication: 2026-10-07

This publication replaces the repository's introduction-only landing README
with the completed, step-organised TEF implementation. The source snapshot is
the verified `TEF_clean` tree, including the optional online and dynamic-object
extensions. All Python implementation and comparison files are copied byte for
byte. The T2/P2 and online numerical configurations are unchanged.

Publication-only adjustments:

- Replace development-machine absolute paths in the dynamic example JSON with
  documented placeholders under `data/kitti0059_t51/`.
- Update the README to distinguish optional online replay from omitted research
  features and to report the completed verification accurately.
- Retain the existing source-publication license notice, without choosing a
  new reuse license.
- Add English validation and publication notes and exclude generated files,
  datasets, caches and local environments.

The default T2 still uses the frozen PCA-plus-fallback configuration. Later
sampling or dwell-clearing experiments do not change this publication's
algorithm or parameters. GPU/CPU/disk tiered paging is not included; the core's
CPU archival of distant blocks remains available.

`../publication_manifest.json` records source and publication hashes, including
the original verification-report hashes. Paths are relative to the source or
repository, without development-machine home directories.

See [validation.md](validation.md) for the completed equivalence results and
their limitations. The standalone synthetic demo and entry-point checks are
also run on this exact publication snapshot before upload. This packaging pass
does not rerun the full real-data experiment suite or test a fresh dependency
installation.

The older, unpublished 2026-10-02 local release commit is retained locally and
is not mixed into this reorganised source snapshot.

## Checks on the uploaded snapshot

- All 42 Python files parse and match the verified source byte for byte.
- All 12 documented command-line entry points return successfully with `--help`.
- T2 and P2 YAML configurations load, and the dynamic example schema is valid.
- No Chinese text, development-machine absolute home paths or credential patterns
  were found in the publication inventory; Markdown file links resolve.
- The CPU T2 synthetic demo passes: 289,468 vertices, 514,057 triangles,
  floor median error 1.24 cm, wall median error 0.09 cm (5 cm limits).

Checks use the existing Python 3.12 environment. ROS bag reconstruction and
full real-data equivalence are covered by the recorded earlier checks, not
rerun during publication.
