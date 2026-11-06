# icesat2_topobathy — agent orientation

Research codebase comparing **ICESat-2** satellite lidar (ATL03 photon data,
ATL08 land/canopy product, ATL24 bathymetry product) against **airborne
lidar (ALS)** reference data, for a Florida mangrove/coastal topobathy
accuracy study. Current focus (the `footprint` work, now merged to `main`)
is estimating and correcting **geolocation offsets** between ICESat-2 and
ALS using building rooftops/footprints as control features.

## Core pipeline

- **`main.py`** — the canonical pipeline. Per ATL03/ATL08/ATL24 file triple
  and ground track: reads/aligns photon data (`utils.readers`) → filters to
  a spatial extent → geoid-corrects heights (EGM2008) → aggregates photons
  into along-track segments (`utils.analysis`, driven by the
  `ATL_AGG_CONFIG`/`ATL08_AGG_CONFIG`/`ALS_AGG_CONFIG` recipes at the top of
  the file) → pulls the matching ALS swath for the same corridor → merges →
  writes parquet. Config-driven via `config.yaml`, with CLI overrides.
- **`create_als_swath.py`** — an older near-duplicate of part of `main.py`'s
  logic. Prefer `main.py` as the source of truth unless told otherwise.

## `utils/` module map

| File | Purpose |
|---|---|
| `readers.py`, `processing.py` | Parse ATL03/08/24 HDF5 into merged photon-level DataFrames |
| `analysis.py` | Aggregation stats (percentiles, canopy cover, PAI, etc.) |
| `datum_transforms.py`, `geographic_utils.py` | NAD83↔WGS84, geoid height lookups, UTM zone selection, geodesic distance |
| `create_las_swath.py`, `create_las_area.py`, `create_las_extent.py`, `create_tif_extent.py`, `reference_extent.py` | Extract ALS point-cloud swaths/extents along an ICESat-2 ground track |
| `geolocation.py` | Grid interpolation + MAE cost function for geolocation-offset optimization |
| `building_processing.py` | Finds buildings intersected by an IS2 track, fits edges (RANSAC/DBSCAN) for footprint-based offset correction |
| `metadata.py` | File-matching / attribute helpers (also does a lazy `import icesatUtils` — see Gotchas) |
| `plotter.py` | Along-track classification plots |
| `lidar_processing_draft.py` | Explicitly a draft/WIP file — not part of the stable pipeline |

## Standalone scripts

- `perform_geolocation_offset.py` / `batch_process_footprints.py` — solve
  for the IS2↔ALS geolocation offset via `scipy.optimize` (single file vs.
  batch across many).
- `label_check.py` — QA scan of ATL24 bathymetry training-label CSVs.
- `planet_lab_download.py` / `planet_lab_timelapse.py` — unrelated side
  utility: downloads Planet Labs imagery and builds a timelapse video.
- `quick_plot.py` — ad hoc plotting sandbox.

## Notebooks (interactive front end)

`demo.ipynb`, `demo_ATL08_30m_segment.ipynb`, `footprint_analysis.ipynb`,
`footprint_esf.ipynb`, `footprint_plots.ipynb`, `rvg_example.ipynb`.

## Config/data

`config.yaml` points to absolute, machine-specific network-drive paths
(Florida AOI, Everglades extent, geoid grids) — not portable off this
machine as-is.

## Gotchas

- `utils/metadata.py` does `import icesatUtils` inside two functions — a
  local/internal module, not on PyPI, so it's **not** in `requirements.txt`.
  Only breaks if those specific functions are called.
- `requirements.txt` and `.gitignore` were previously stale/missing; both
  have been fixed (17 third-party packages + jupyter deps; ignores
  `__pycache__/`, `.ipynb_checkpoints/`, and generated data artifacts). Don't
  redo this — check them if a new import is added instead.
- Working tree accumulates generated artifacts (`.gpkg`, `.kml`, `.pqt`,
  `outputs/`, `data/`, `als_swaths/`) — these are gitignored outputs, not
  source, and shouldn't be treated as reference data.
