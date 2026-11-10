
I have an existing codebase that reads ICESat-2 ATL03/ATL08 and compares tracks with airborne lidar (ALS; LAS/LAZ files) to estimate the optimal along-track and cross-track translation. Help me extend this workflow into a reproducible analysis of residual geolocation error, its dependence on spacecraft/instrument parameters, and its variation over time.

**Scientific objective**

Estimate the distribution of horizontal geolocation offsets across many independent passes and sites. Determine:

- Mean signed bias and variability along-track and cross-track.
- Differences among physical ATLAS spots and spacecraft orientations.
- Temporal trends and possible dependence on orbital geometry, solar-array mode, and viewing geometry.
- Whether additional empirical corrections improve performance on independent observations.

The primary target is **residual geolocation knowledge error in the delivered product**: the difference between reported and actual photon locations. This differs from pointing-control error relative to the intended ground track.

Measured ATL03–ALS offsets also contain ALS reference errors, surface changes, and registration uncertainty. Do not automatically attribute all observed variance to ICESat-2.

**Start by auditing the existing code**

Identify how the code:

- Reads and associates ATL03 photons with ATL08 classifications.
- Handles product releases, spacecraft orientation, and beam identifiers.
- Transforms horizontal and vertical coordinates.
- Selects photons and the corresponding ALS surface.
- Defines along-track/cross-track directions and offset signs.
- Chooses registration windows, search bounds, objective functions, and acceptance criteria.
- Handles vertical differences and stores results.

Summarize what already works and what needs extending. Preserve the existing estimator as a reproducible baseline before changing its behavior. Validate ATL03/ATL08 granule correspondence and photon indexing; do not silently mix incompatible inputs.

**Coordinate and correction conventions**

Explicitly distinguish:

- Correction vector: translation applied to ATL03 to align it with ALS.
- Geolocation error vector: reported location minus reference location, which has the opposite sign.

Define positive along-track and cross-track directions and retain the local heading. Derive heading from time-ordered track coordinates in a suitable metric frame. Earth rotation is already represented in the delivered coordinates; do not apply another Earth-rotation correction.

Use standard ATL03 Release 007 as the intended baseline and record actual release/revision information. Release 007 uses ITRF2020 with heights referenced to the WGS84 ellipsoid. Verify ALS datum realization, coordinate epoch where relevant, vertical datum, units, and transformations. Sharing a projection does not guarantee sharing a reference frame.

Do not reapply upstream pointing/ranging corrections or subtract historical beam biases without verifying applicability. Updated calibration does not guarantee zero residual bias.

**Metadata to preserve**

Store one analysis record per physical spot, pass, and registration window, with stable identifiers linking it to source data.

| Category | Variables |
|---|---|
| Provenance | ATL03/ATL08 granule IDs, releases/revisions, ALS survey/site, processing configuration |
| Acquisition | UTC time, orbit, RGT, cycle, window start/end, location |
| Beam identity | `gtx`, physical ATLAS spot, strong/weak designation, `sc_orient` |
| Spacecraft geometry | `beta_angle`, `orbit_angle`, `ref_elev`, `ref_azimuth`; optionally roll/pitch/yaw |
| Operating state | Airplane/sailboat solar-array mode and nearby maneuver/transition events, when available |
| Quality | `podppd_flag`, `quality_ph`, signal-selection/classification information |
| Uncertainty | `sigma_along`, `sigma_across`, `sigma_h`, `ph_uncorrelated_error` |
| Sampling | Window length, accepted photon count, illumination, ALS coverage/density, terrain characteristics |
| Results | Signed offsets, vertical nuisance offset if fitted, objective values, fit uncertainty, acceptance status/reason |

Physical spot identity must remain consistent across yaw flips; `gtx` alone is insufficient. Forward/backward orientation and airplane/sailboat solar-array mode are separate variables. Obtain solar-array mode from documented mission activity records where possible; mark unavailable values explicitly.

Verify field paths, dimensions, units, fill values, and photon/segment alignment against actual files. In particular, `ph_uncorrelated_error` is an ancillary beam-dependent array rather than an ordinary per-segment field. Summarize segment metadata over each registration window while retaining relevant ranges and flagged fractions.

**Quality and registration diagnostics**

Start with a conservative nominal-quality sample, including `podppd_flag == 0` and exclusion of orientation transitions. Use Release 007 photon-quality definitions and document every filter. Ground matching requires ground selection, not merely signal-photon selection.

Keep rejected records and rejection reasons. Do not reject a solution solely because its offset is large.

For every fit, retain:

- Best along-track/cross-track correction.
- Objective before and after shifting.
- Search-boundary hits and competing minima.
- An objective-surface representation or sufficient diagnostics to assess ambiguity.
- Estimated uncertainty and covariance of the fitted offsets, where feasible.
- Photon count, spatial coverage, and terrain variation.

Test sensitivity to a vertical nuisance offset: vertical mismatch can otherwise appear as horizontal displacement on a slope. However, a planar surface cannot uniquely constrain two horizontal shifts and a free vertical offset. Assess identifiability rather than assuming every optimizer minimum is meaningful.

Test a small set of registration-window lengths supported by the data. Window length changes both fitting precision and the scale of geolocation variation being measured.

**Analysis approach**

First characterize offsets without additional empirical bias correction. Begin with physical spot × forward/backward orientation, reporting:

- Mean and median signed offsets.
- Along-track and cross-track standard deviations and covariance.
- RMSE relative to zero, distinguished from spread around the mean.
- Empirical radial-error quantiles.
- Numbers of windows, independent passes, and ALS surveys/sites.

Plot signed offsets over time, along-track versus cross-track scatter, and distributions by spot/orientation. Examine orbital angles, solar-array mode, viewing geometry, and illumination as additional explanatory variables when sample coverage permits. Treat orbit angle as circular.

Check whether apparent spacecraft or temporal effects are confounded with site, ALS survey, terrain, season, or sampling quality. Nearby windows and simultaneous beams may share errors; use pass/site grouping for uncertainty estimation rather than treating every window as independent.

A useful conceptual decomposition is:

measured offset = conditional bias + residual pass variation + ALS survey effect + registration error.

Do not assume those components can all be uniquely separated with the available data.

**Empirical correction and validation**

Only develop an additional correction after establishing the baseline. Fit it on training passes/sites and evaluate it on held-out passes/sites; avoid random splits of neighboring windows.

Preserve both the original estimates and corrected residuals. Report whether correction reduces held-out bias and error, and whether improvements generalize across time and sites.

Do not treat ATL03 uncertainty fields as signed corrections. Check whether horizontal sigma fields contain defaults. Do not assume `sigma_h` components can be separated by quadrature subtraction without checking the product’s documented construction. Tide-system conversions are datum operations, not horizontal pointing corrections.

**Implementation and deliverables**

Implement incrementally: audit → metadata extraction → pilot batch → diagnostics → larger analysis → optional correction model.

Provide reproducible commands/configuration, resumable batch processing, structured result tables, diagnostic plots, and a concise findings report. Run a representative pilot before scaling. Report missing inputs and scientific limitations explicitly.

Primary references to consult when resolving product-specific questions:

- Luthcke et al. (2021): https://doi.org/10.1029/2020EA001494
- ATL03 Release 007 documentation, including ATBD, dictionary, known issues, and mission activities: https://nsidc.org/data/atl03/versions/7

USER NOTE:
gemini_geolocation_summary.pdf is from another engineer, it has some inaccuracies that would need to checked if used as a reference, document.

Historical calibration results motivate the analysis but should not be assumed to describe Release 007 residual errors.

Append:

Explicitly include illumination and laser incidence geometry in the metadata and analysis.
Variable	Purpose
solar_elevation	Continuous illumination variable; retain degrees rather than only day/night
solar_azimuth	Solar direction, useful alongside terrain aspect
Background photon rate, where available	Measure background conditions affecting photon selection and registration
ref_elev, ref_azimuth	Preserve native beam-pointing geometry and units
Beam angle from local vertical	Characterize viewing geometry independently of terrain slope
Local terrain incidence angle	Characterize the angle between the laser line of sight and the ALS-derived terrain normal
Terrain slope, aspect, and roughness	Interpret incidence geometry and registration performanceDefine the two laser angles explicitly.
1. Beam angle from local vertical: the angle between the surface-to-satellite line of sight and the local ellipsoid normal. Zero degrees means locally vertical viewing. This is often loosely called off-nadir angle; distinguish it from an angle defined at the spacecraft.
2. Local terrain incidence angle: the angle between the surface-to-satellite line of sight and the upward-facing terrain normal. Zero degrees means perpendicular incidence on the terrain. A nearly vertical beam can still have a large terrain incidence angle on a steep slope.
Construct a unit surface-to-satellite vector, l, from the documented pointing geometry, verifying its direction and coordinate convention. With local upward unit vector u and upward terrain normal n:
- Vertical-reference angle = arccos(l · u).
- Terrain incidence angle = arccos(l · n).
Clamp dot products to [-1, 1] before evaluating arccos. Verify radians versus degrees and test the geometry against simple flat and sloping surfaces. Do not assume ref_elev itself is the incidence angle.

Estimate terrain normals from the ALS ground surface using a documented neighborhood scale. Check sensitivity to that scale; roughness and small surface features can make incidence estimates unstable. Use the registered footprint locations for the final terrain-incidence calculation, retaining the unshifted values for comparison. Mark estimates invalid where ALS coverage or terrain-normal quality is inadequate.

Preserve segment-level geometry and summarize each registration window with its median, spread, and range. Use circular statistics for azimuths. Record missing values and their causes.

Analyze both signed offsets and offset dispersion against solar elevation, background rate, vertical-reference angle, and terrain incidence angle. Also examine registration uncertainty, photon count, and rejection rate against these variables.

Distinguish evidence of a spacecraft geolocation effect from changes in matching performance: for example, larger offset scatter under high background conditions could reflect less precise registration. Account for spot, orientation, site, terrain, and acquisition time when evaluating these relationships.
