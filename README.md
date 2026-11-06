# icesat2_topobathy / `phoreal`

Tools for comparing ICESat-2 photon data (ATL03, ATL08, ATL24) with airborne
lidar (ALS): reading and aligning photons, geoid/datum transforms, ALS swath
extraction along a ground track, along-track aggregation, and geolocation-offset
estimation.

The reusable code is the installable `phoreal` package in `src/`. Experiment code
lives in `projects/`.

## Setup (uv, Python 3.12)

```bash
uv sync --all-extras          # creates .venv with phoreal (editable) + all extras
uv run nbstripout --install --attributes .gitattributes   # once per clone
uv run pytest                 # import smoke test
```

Extras: `projects` (scikit-learn, fiona), `planet` (requests, opencv, Pillow),
`notebooks` (jupyter). Use `uv sync` with no extras for the library alone.

## Layout

```
src/phoreal/            reusable library
  io/                   readers, processing, metadata (ATL03/08/24 HDF5 -> DataFrames)
  geo/                  datum_transforms, geographic_utils
  als/                  swath (ALS along a track), las/tif/reference extent, area
  analysis.py           aggregation statistics
  agg_configs.py        ATL/ATL08/ALS aggregation recipes
  geolocation.py        interpolation + MAE cost for offset optimization
  plotting.py           along-track plots
  config.py             load_config() for per-folder config.yaml
scripts/                generic CLIs (run_pipeline.py, create_als_swath.py)
examples/               demo notebooks, quick_plot.py
projects/<name>/        experiment code and notebooks for one study
tests/
```

## Paths and configs

Nothing tracked contains machine-specific paths. Each `projects/<name>/` and
`examples/` folder has a tracked **`config.example.yaml`**. Copy it to
**`config.yaml`** (gitignored) and fill in your paths. Notebooks call
`load_config('config.yaml')`, so run them from their own folder.
Each project's `data/` and `outputs/` folders are also gitignored.

```bash
uv run python scripts/run_pipeline.py --config projects/fl_mangrove_topobathy/config.yaml
```

Notebook outputs are stripped on commit by the nbstripout git filter; your
local copies keep them.
