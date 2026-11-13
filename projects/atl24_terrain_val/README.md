# ATL24 vs ATL08 terrain validation against 3DEP

Do ATL24 terrain photons (next ATL24 release, `class_ph == 2`) read low on
slopes the way ATL08 ground photons do? Both classifiers are compared on the
same ATL03 20 m geolocation segments, against the USGS 3DEP 1 m lidar DTM.
Background: `docs/atl08_atl24_terrain_val.txt` (brief) and
`docs/ATL08 Slope Bias.pdf` (D. Shean, ATL08 slope bias).

## Data notes

- The ATL24 CSVs (one per beam) contain **every** ATL03 photon in ATL03 order,
  so they join to ATL03 row for row. `phoreal.io.readers.read_atl24_csv`
  checks time (`GPS_SECONDS - 1198800018 = delta_time`) and `h_ph` for every
  photon and raises on a mismatch.
- `class_ph`: 0 unclassified, 1 other, 2 terrain, 40 bathymetry, 41 sea
  surface. `atl24_prob_0..4` correspond to these classes in that order
  (argmax matches `class_ph` for about 99% of photons).
- Heights are compared in the 3DEP frame: photons go from ITRF2014 (granule
  epoch) to NAD83(2011) UTM + NAVD88 (GEOID18) with pyproj
  (`phoreal.geo.dem.itrf_to_dem_frame`). The geoid grid comes from the PROJ CDN.
- Snow is **not** removed. On snow-covered segments, height − 3DEP includes
  snow depth. The ATL08 `segment_snowcover` flag is carried along for
  subsetting.

## Run (from this folder)

```bash
cp config.example.yaml config.yaml          # then set paths
uv run python build_photons.py              # photon parquet per beam + QA tables
uv run python stage_3dep.py                 # 3DEP 1 m corridor tiles (HTTP, COG windows)
uv run python build_segments.py             # per-segment heights / dz / slope / coverage
uv run python analyze.py                    # metrics tables + figures in outputs/results/
```

Outputs go to `work_dir` (default `outputs/`) and the DEM to `dem_dir`.
Both are gitignored.

## Estimators (all in `segments.parquet`)

| key | photons / product | columns |
|---|---|---|
| `a08` | ATL08 ground (`atl08_class == 1`) | `n_a08`, `dz_a08` (median photon dz), `dzc_a08` (median h − DEM at centre) |
| `a24` | ATL24 terrain (`class_ph == 2`) | `n_a24`, `dz_a24`, `dzc_a24` |
| `sig` | ATL03 land signal, `signal_conf_ph[:,0] >= 4` | `n_sig`, `dz_sig`, `dzc_sig` |
| official ATL08 | `h_te_best_fit_20m`, 100 m `h_te_best_fit/median/interp` | `dz_atl08_*` |

Coverage per segment: `n_both`, `n_a08_only`, `n_a24_only` (photon counts).
Slope and aspect come from a 40 m plane fit on the DTM. `orientation` is
along / across / oblique (downslope direction within 30° of the track or of
its perpendicular).
