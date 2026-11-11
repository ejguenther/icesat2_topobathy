# Poland 500-granule run: first findings

ICESat-2 R007 residual horizontal geolocation error against the 2025 Poland ALS
1 m DTM. Run 2026-11-10 16:16 → 2026-11-11 08:53 (about 16.5 h, 18 workers).
Joint per-pass refit 2026-11-11 (25 min, `refit_pass.py`). Code: branch
`geolocation-poland` (`c1a0e0c`). Outputs live under the run's `work_dir`
(`offsets/offsets.parquet`, `offsets/review/`, `offsets/pass_refit/`); see
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

The joint per-pass refit (Finding 3) confirms it: **E +0.54 m** (95% CI
0.39–0.71) and **N +0.26 m** (0.12–0.43). East is positive in all four
direction × orientation groups (+0.41 to +0.67 m); north ranges from +0.09 to
+0.63 m.

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

For the 1,985 passes with ≥ 4 accepted beams, the beams of the same pass
scatter by **1.71 m AT and 1.92 m XT** (median within-pass SD). The median
bootstrap σ for the same windows is 0.52 / 0.63 m. That is about 3× smaller,
which repeats the pilot result (1.8–2.1 m vs ~0.7 m) at scale.

These numbers group beams by segment-window index, which only approximately
lines up across beams (beams split the ALS coverage at different points; only
39% of such groups covered the same time span). The joint refit uses
time-aligned passes instead and gives the same answer: each beam's own fit
scatters about the pass's shared shift by a robust SD of **1.87 m AT and
1.95 m XT**.

So the block bootstrap captures photon-sampling noise but not the dominant
error. Possible causes include terrain and DTM structure the fit absorbs,
under-constrained directions on flat ground, and real spot-to-spot pointing
differences. Until this is understood, per-beam σ should be inflated (≈3×) or
taken from the empirical scatter, not from the bootstrap.

### 3. Most of the error is shared by the pass and persists along the orbit

**Joint refit** (`refit_pass.py`, from `refit_cache`, no ALS re-read). Each
granule's ALS stretches were cut into time chunks of at most 20 km (median
12 km), so every beam covers the same ground. That gives 5,055 passes; 2,370
have all six beams and 3,311 have at least four. Two models were fitted:
- **Model 1:** one along-/cross-track shift per pass shared by all beams,
  minimising the photon-pooled MAE. 2 m brute force over ±10 m, then bounded
  Nelder–Mead.
- **Model 2:** the same with a fixed per-spot × orientation offset applied to
  each beam (Finding 4).

Each beam was also fitted on its own over the same photons. Uncertainty comes
from leaving one beam out at a time. The ±10 m boundary was hit in 2.7% of
passes, which are excluded. Single-beam passes reproduce the solo fit exactly.

| Passes with ≥ 4 beams (3,311) | AT | XT |
|---|---|---|
| Spread of pass shifts (robust SD) | 1.51 m | 1.60 m |
| Pass-shift uncertainty (median, leave-one-beam-out) | 0.86–0.89 m | 0.25–0.65 m |
| Correlation between adjacent 20 km chunks of a granule (1,643 pairs) | 0.80 | 0.61 |
| Spread of granule means (robust SD) | 1.33 m | 1.34 m |
| Spread within a granule (robust SD) | 0.66 m | 1.02 m |

Pass shifts spread well beyond their uncertainty, leaving roughly **1.2–1.5 m
of real pass-level error** per axis. That error persists along the orbit:
neighbouring 20 km stretches of the same granule have strongly correlated
shifts, and along-track the granule means differ by twice as much as
stretches within a granule. This points to orbit or pointing error that holds
over hundreds of kilometres, rather than terrain noise. The per-granule shifts
are what to model against spacecraft attitude, beta angle and the other
ancillary variables.

The cross-track leave-one-beam-out uncertainty differs between the two models
(0.25 vs 0.65 m) and hasn't been investigated; treat it as rough.

### 4. Fixed per-spot offsets are small and don't explain the beam scatter

The median of (beam's own error − pass error) per spot × orientation gives a
fixed per-spot offset. These offsets are within ±1 m and mostly reverse
between orientations, as expected when the spot layout is mirrored. The
largest are for weak spot 4 (0.8–0.9 m AT, 0.6–1.0 m XT) and for spot 3
cross-track in backward orientation (0.9 m).

Model 2 applies these offsets. Its pass shifts are nearly identical to model
1's, and the beam scatter barely changes: robust SD 1.87 → 1.76 m AT and
1.95 → 1.98 m XT. Most of the scatter between beams is therefore **not** a
fixed per-spot bias. The large early-mission spot pattern in 2018 (spot 1 −
spot 4 = +2.6 m AT, forward) is real for those passes, but only 14 granules
come from 2018.

**Bias-removed spread by spot and orientation** (joint refit, time-aligned
passes with ≥ 4 beams). *Offset* is the spot's median error relative to its
pass. *SD about pass* is the scatter of the beam's own fit about the pass's
shared shift. *After offsets* also removes the fixed per-spot offset (model 2).
All values in metres.

| Spot | Beam | Orientation | Beams | Offset AT | Offset XT | SD about pass AT | SD about pass XT | SD after offsets AT | SD after offsets XT | Robust SD after offsets AT | Robust SD after offsets XT |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | strong | forward | 1,391 | −0.39 | +0.02 | 2.31 | 2.56 | 2.30 | 2.54 | 2.39 | 2.46 |
| 2 | weak | forward | 1,337 | +0.44 | +0.31 | 2.23 | 3.08 | 2.24 | 3.08 | 2.26 | 2.85 |
| 3 | strong | forward | 1,525 | +0.38 | −0.23 | 2.05 | 2.50 | 2.04 | 2.49 | 1.99 | 2.05 |
| 4 | weak | forward | 1,484 | +0.84 | +0.64 | 1.95 | 2.85 | 1.94 | 2.83 | 1.46 | 2.59 |
| 5 | strong | forward | 1,399 | −0.35 | +0.05 | 1.52 | 2.00 | 1.53 | 2.00 | 1.24 | 1.38 |
| 6 | weak | forward | 1,333 | +0.35 | −0.06 | 1.87 | 2.45 | 1.87 | 2.43 | 1.64 | 1.78 |
| 1 | strong | backward | 1,529 | +0.21 | −0.15 | 2.05 | 2.46 | 2.05 | 2.46 | 2.09 | 1.67 |
| 2 | weak | backward | 1,474 | −0.54 | −0.39 | 2.29 | 3.15 | 2.30 | 3.16 | 2.40 | 3.31 |
| 3 | strong | backward | 1,700 | −0.12 | +0.91 | 2.01 | 2.55 | 2.01 | 2.53 | 1.82 | 1.66 |
| 4 | weak | backward | 1,653 | −0.94 | −1.03 | 1.97 | 2.69 | 1.97 | 2.68 | 1.53 | 2.46 |
| 5 | strong | backward | 1,523 | +0.36 | +0.05 | 1.44 | 2.08 | 1.44 | 2.05 | 1.11 | 1.41 |
| 6 | weak | backward | 1,471 | −0.33 | −0.21 | 2.01 | 2.50 | 2.01 | 2.48 | 1.69 | 1.83 |

Spot 5 is the least noisy in both orientations (robust SD 1.1–1.2 m AT, 1.4 m
XT). Weak spots 2 and 4 are the noisiest cross-track (2.5–3.3 m). An earlier
version of this table, grouped by segment-window index and limited to accepted
beams, suggested spots 1–3 were noisier than spots 4–6; with time-aligned
passes that pattern does not hold.

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
2. Model the per-granule shifts from the joint refit (Finding 3) against the
   spacecraft variables already in the table (roll, pitch, yaw, beta angle,
   orbit angle, `podppd`).
3. Explain the per-beam excess scatter (Finding 2): relate each beam's
   residual about its pass shift to cost-surface conditioning, DTM fill
   fraction and terrain roughness, then build an empirical σ model. Spot 5
   being consistently cleanest is a lead.
4. Check the cross-track leave-one-beam-out uncertainty, which differs between
   the two refit models.
5. Make the acquisition step detect empty Harmony subsets automatically.
