# Poland 500-granule run: first findings

ICESat-2 R007 residual horizontal geolocation error against the 2025 Poland ALS
1 m DTM. Run 2026-11-10 16:16 → 2026-11-11 08:53 (about 16.5 h, 18 workers).
Code: branch `geolocation-poland` (`d859ef1`). Outputs live under the run's
`work_dir` (`offsets/offsets.parquet`, `offsets/review/`); see
`config.example.yaml` for the layout.

These are first-pass numbers from the batch output. Nothing below has had a
targeted follow-up yet; open questions are listed at the end.

## What was run

- **Selection:** 1,569 ATL03/ATL08 pairs whose footprints cross the ALS
  coverage (Oct 2018 to present). ATL08 screening kept granules with at least 2
  strong beams having ≥ 2 km of ground photons on the ALS. A greedy
  stratified pick chose 500 of them:
  - 125 per spacecraft orientation × ascending/descending
  - day and night about even
  - about 61 per year for 2019–2026 and 14 in 2018
  - spread across the 137 ALS surveys
- **ATL03:** Harmony trajectory subsets clipped to the ALS bbox. 3 of 500
  came back with no photon groups (see *Data issues*). These were replaced by
  full-granule downloads subset locally.
- **Reference surface:** the 1 m ALS DTM, bilinearly sampled at the nodes of
  an along-/cross-track grid (±20 m cross-track). It was validated against the
  LAZ ground points on the 10-granule pilot:
  - heights: −0.06 m bias, 0.04 m SD
  - windows both runs accept: offsets correlate at r = 0.98 / 0.975
  - all windows: r = 0.86 along-track; the rest of the disagreement is in
    weakly constrained windows
- **Estimator:** MAE between ground photons and the surface, minimised over a
  ±10 m brute-force grid (0.5 m) followed by a local refine. Each window gets
  diagnostics: boundary hit, second minimum, Hessian, `dz_opt`.
  - Uncertainty: block bootstrap with 30 replicates, local Nelder–Mead from the
    solution. Blocks are 1 km for segment windows and 250 m for 2 km windows;
    1 km windows get none.
- **Windows per beam:**
  - *segment:* a contiguous ALS stretch, split at gaps > 1 km and capped at
    20 km; median 11.5 km and 5,200 photons when accepted
  - fixed 2 km and 1 km windows
- **Sign convention:** error = reported IS2 position − true position, i.e. the
  negative of the shift that best aligns photons to the ALS. AT is positive in
  the direction of travel, XT positive to the left.

## Output volume and acceptance

| Window | Windows | Accepted | Median bootstrap σ AT / XT | Robust SD of accepted error AT / XT |
|---|---|---|---|---|
| segment | 26,566 | 58% | 0.67 / 0.79 m | 2.28 / 2.74 m |
| 2 km | 127,915 | 39% | 0.98 / 1.04 m | 2.43 / 2.81 m |
| 1 km | 238,131 | 44% | – | 2.83 / 3.21 m |

The run covers 500 granules and 2,946 beams. Accepted segment windows come from
493 granules (15,308 windows). The rest of this note uses those unless stated.

Reasons segment windows were rejected (a window can have several flags):

| Flag | Share of segment windows |
|---|---|
| weak constraint (flat cost surface) | 17% |
| ambiguous minimum (competing second minimum) | 16% |
| too few photons | 13% |
| no solution | 13% |
| bootstrap replicates hit the search edge | 4% |
| solution on the search boundary | 3% |
| degraded POD/PPD flag | 0.2% |

Acceptance is higher for strong beams (64%) than weak ones (51%), and higher at
night (63%) than in daylight (53%). Terrain slope barely matters: most windows
are < 1° and acceptance is 66–74% in every slope class.

Vertically the DTM and photons agree: median `dz_opt` is +0.03 m.

## Findings

### 1. A small, consistent offset to the east and north

Over all accepted segment windows, the median error is **+0.57 m east and
+0.33 m north**. Taking the mean per granule, then the median over the 493
granules, gives **E +0.60 m (95% CI 0.48–0.74)** and **N +0.27 m (0.13–0.42)**.
The CI comes from a bootstrap over granules.

The offset is geographic, not tied to the instrument frame. Ascending and
descending passes give similar E/N medians (E +0.67 / +0.49, N +0.24 / +0.43).
Their along-/cross-track medians, by contrast, change sign:

| | AT | XT |
|---|---|---|
| ascending | +0.15 | −0.61 |
| descending | −0.48 | +0.47 |

The offset is not uniform across direction × orientation: N ranges from −0.10
to +0.86 m across the four groups. So it is mixed with a component that depends
on geometry.

**Hypothesis to check (unverified):** the size (~0.65 m) and direction (NE)
are close to the Eurasian plate's motion in ITRF. The plate moves roughly 2–3
cm/yr toward the NE, which accumulates to tens of centimetres to about a
metre between a plate-fixed frame (ETRF89/ETRF2000) and ITRF at the
observation epoch. That is the signature you would see if the ALS were really
in PL-ETRF2000 but labelled WGS84/ITRF. ICESat-2 reports ITRF2014 at the
observation epoch.
- *Against:* the expected time trend (~+2 cm/yr in E and N) is not visible
  (fitted E −2.5 cm/yr, N +2.1 cm/yr), but at this noise level a trend that
  small could not be detected.
- *How to check:* confirm the ALS frame and epoch from the provider's metadata
  (GUGiK, PL-ETRF2000 vs ITRF), or apply the ETRF2000→ITRF2014 transform at
  2025.x and re-fit a subset.

### 2. A single-beam error is much larger than the bootstrap says

For passes with ≥ 4 accepted beams, the beams of the same pass scatter by
**1.7 m AT and 2.0 m XT** (median within-pass SD). The median bootstrap σ for
the same windows is 0.55 / 0.66 m. That is about 3× smaller, which repeats the
pilot result (1.8–2.1 m vs ~0.7 m) at scale.

So the block bootstrap captures photon-sampling noise but not the dominant
error. Possible causes include terrain and DTM structure the fit absorbs,
under-constrained directions on flat ground, and real spot-to-spot pointing
differences. Until this is understood, per-beam σ should be inflated (≈3×) or
taken from the empirical scatter, not from the bootstrap.

### 3. There is also a real pass-level component

The robust SD of the pass means is **1.8 m AT, 1.9 m XT**. If beams scattered
independently around the truth, averaging 4–6 of them would leave only about
0.8 m. Subtracting that in quadrature leaves a **common pass-level error of
roughly 1.5–1.7 m** per axis. This is the component a joint per-pass fit would
estimate.

### 4. Spot-dependent offsets exist but are small

After removing each pass mean, the median offsets per spot are within about
±0.6 m:

| Spot | AT backward | AT forward | XT backward | XT forward |
|---|---|---|---|---|
| 1 (strong) | +0.41 | −0.58 | −0.14 | −0.08 |
| 2 (weak) | −0.47 | +0.21 | −0.33 | +0.30 |
| 3 (strong) | +0.09 | +0.31 | +1.13 | −0.24 |
| 4 (weak) | −0.64 | +0.53 | −0.81 | +0.13 |
| 5 (strong) | +0.60 | −0.36 | +0.36 | +0.32 |
| 6 (weak) | −0.22 | +0.09 | −0.30 | −0.23 |

The AT pattern mostly reverses between orientations, as expected when the spot
layout is mirrored. The difference between spots 1 and 4 changes by year and
orientation (for forward: +2.6 m in 2018, −0.6 to −2.4 m afterwards).
- Removing a spot × orientation × year median reduces the within-pass RMS only
  from 1.53 to 1.32 m (AT) and from 1.72 to 1.54 m (XT).
- So most of the scatter between beams is **not** a fixed per-spot bias.
- 2018 has only 14 granules (478 windows). The large early-mission spot pattern
  seen in the first version of the pass figure is real for those passes but
  should not be generalised.

### 5. Weak dependence on orientation, illumination, terrain and year

- **Orientation:** median AT is −0.49 m backward vs +0.27 m forward; XT is
  −0.45 vs −0.32 m. In E/N: backward +0.44 / +0.30, forward +0.77 / +0.36.
- **Day vs night:** medians are within 0.2 m. Robust SD is 2.2 vs 2.4 m (AT)
  and 2.8 vs 2.6 m (XT). Daytime background lowers acceptance but not accuracy.
- **Terrain slope:** bootstrap σ and spread show no clear trend. The site is
  mostly flat (90% of accepted windows are < 1°), which limits what this data
  set can say about slope.
- **Year:** E and N medians stay within about 0.5 m from 2019 to 2026. The
  robust SD of E rises from 1.8 m (2019) to 3.0 m (2025–26). This is not yet
  explained: it may reflect which surveys later passes cross rather than the
  instrument.

## Data issues found during the run

- **Empty Harmony subsets:** 3 of 500 ATL03 subsets (`20190731235409_05180402`,
  `20251019235955_05412906`, `20260614235843_00153202`) came back with no photon
  data (no `heights`/`geolocation` groups). They were exactly 56 or 64 MiB, and
  re-requesting gave identical files. All three granules passed the ATL08
  screen, and full-granule downloads subset locally have photons on all 6
  beams. The acquisition step should check for photon groups and fall back
  automatically; at present this only surfaces as offset errors.
- **Runtime:** cost is about 4 worker-seconds per window and national passes
  produce 1,000–2,200 windows each. The early rate (14 granules/h) gave a
  36 h projection, but the actual run took 16.5 h because the large passes
  were front-loaded. The rough cost breakdown suggests 1 km windows are only
  ~15–20% of compute; this has not been profiled.

## Caveats

- One ALS reference (2025, flat lowland and foothills). This is ICESat-2 error
  relative to that DTM, including any DTM or frame error (see Finding 1).
- The DTM fills ground under buildings and canopy; filled areas weaken
  constraints and explain most of the DTM-vs-LAZ disagreement in the pilot.
- `ph_uncorrelated_error` mapping to spots is still unverified and is not used
  here.
- Acceptance thresholds are the pilot's and have not been tuned for national
  data.
- `summary_by_spot_orientation.csv` still uses the pilot's "full pass" column
  names; it summarises accepted segment windows.

## Next steps

1. Settle the ALS reference frame and epoch (Finding 1). This changes the bias
   and may change the year trend.
2. Fit the cached surfaces jointly per pass: one shift per pass plus per-spot
   terms, from `refit_cache`. This separates pass-level pointing from per-beam
   noise directly.
3. Investigate the per-beam excess scatter (Finding 2), e.g. relate the
   within-pass residual to Hessian conditioning, DTM fill fraction and terrain
   roughness. Use that to build an empirical σ model.
4. Model error against the spacecraft variables already in the table (roll,
   pitch, yaw, beta angle, orbit angle, `podppd`) at the pass level.
5. Make the acquisition step detect empty Harmony subsets automatically.
