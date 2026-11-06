# icesat2_topobathy — agent orientation

Research codebase comparing **ICESat-2** photon data (ATL03 / ATL08 / ATL24)
with **airborne lidar (ALS)**. The reusable code is the installable package
**`phoreal`** (`src/phoreal/`, version 4.x, the successor to the older PhoREAL).
Experiment-specific code lives in `projects/`.

## Environment

uv + Python 3.12: `uv sync --all-extras`, `uv run pytest`. Do not use the base
conda env: it has the **old** phoreal 3.31.1 egg, which other repos
(`~/dev/atl18_gridding`) still depend on.

## Layout

| Path | Purpose |
|---|---|
| `src/phoreal/io/` | `readers`, `processing`, `metadata`: parse ATL03/08/24 HDF5 into photon DataFrames |
| `src/phoreal/geo/` | `datum_transforms` (NAD83↔WGS84, geoid lookups), `geographic_utils` (UTM zone, distances) |
| `src/phoreal/als/` | `swath` (ALS swath along a track), `las_extent` / `tif_extent` / `reference_extent` / `area` (tile extents, target lookups), `lidar_processing_draft` (WIP) |
| `src/phoreal/analysis.py`, `agg_configs.py` | Aggregation stats and the ATL/ATL08/ALS recipes |
| `src/phoreal/geolocation.py` | Grid interpolation + MAE cost for offset optimization |
| `src/phoreal/plotting.py` | Along-track plots (often imported `as plotter`) |
| `src/phoreal/config.py` | `load_config(path)` for per-folder `config.yaml` |
| `scripts/run_pipeline.py` | Canonical config-driven pipeline (was `main.py`). `create_als_swath.py` is an older near-duplicate |
| `examples/` | Demo notebooks, `quick_plot.py` |
| `projects/footprint/` | Building-footprint geolocation work (ForestSat26): `building_processing.py`, batch/offset scripts, notebooks |
| `projects/fl_mangrove_topobathy/` | Config for `scripts/run_pipeline.py` (Florida AOI) |
| `projects/atl24_labels/` | `label_check.py`: ATL24 label CSV QA |
| `projects/planet_timelapse/` | Unrelated Planet Labs download/timelapse utility |

Dependency rule: `projects/` may import `phoreal`, but `phoreal` never imports from `projects/`.

## Rules

- **No machine paths in tracked files** (public repo). Paths go in a
  gitignored `config.yaml`, with a tracked `config.example.yaml` template next to it.
  Library functions take paths as arguments.
- Notebooks: outputs are stripped on commit by the nbstripout filter
  (`.gitattributes`). Run notebooks from their own folder.
- `local/` (gitignored) keeps the old `__main__` run-history blocks removed
  from `phoreal/als/*`.

## Known issues (not yet fixed)

- `als/las_extent`, `tif_extent` and `reference_extent` are copy-paste forks
  (`tif_extent` ≡ `reference_extent` apart from the removed `__main__`).
  Merging them is pending.
- `phoreal/als/swath.py` has a leftover `import pdb; pdb.set_trace()` in the
  function that creates and transforms the swath.
- `phoreal/io/metadata.py` lazily imports `icesatUtils`, an internal module that
  isn't installed.
