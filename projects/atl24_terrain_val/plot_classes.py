"""Photon classifications along track: ATL08 (top) vs ATL24 (bottom).

Reads the per-beam photon parquet written by build_photons.py. Edit the
settings below and run the file, or run it cell by cell in VS Code
("Run Cell" above each `# %%`, or Shift+Enter in the interactive window).

Colours:
  gray    unclassified (ATL08 0 = noise / unclassified, ATL24 0)
  orange  ground       (ATL08 1, ATL24 2 = terrain)
  green   vegetation   (ATL08 2 canopy and 3 top of canopy, ATL24 1)
  blue    water        (ATL24 40 bathymetry and 41 sea surface)
  black   3DEP 1 m DTM along the track (needs stage_3dep.py), shown as
          ellipsoidal height so it lines up with the photons
"""
# %% settings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from build_segments import SECONDS_PER_YEAR, dem_paths
from phoreal.config import load_config
from phoreal.geo.dem import DemStack, itrf_to_dem_frame

BEAM = 'gt1l'
LAT_RANGE = (38.73, 39.77)      # degrees north; keep the window small (~5 km), beams hold millions of photons
X_AXIS = 'alongtrack'           # 'alongtrack' (km) or 'latitude'
POINT_SIZE = 2
DEM_BIN_M = 2                   # along-track spacing of the 3DEP line (m)

HERE = Path(__file__).resolve().parent if '__file__' in globals() else Path.cwd()
cfg = load_config(HERE / 'config.yaml')
work = Path(cfg['work_dir'])
work = work if work.is_absolute() else HERE / work

ATL08_CLASSES = {  # code: (label, colour); drawn in this order, so ground lands on top
    0: ('unclassified', '#9a9996'),
    2: ('vegetation', '#2ca02c'),
    3: ('vegetation', '#2ca02c'),
    1: ('ground', '#eb6834'),
}
ATL24_CLASSES = {
    0: ('unclassified', '#9a9996'),
    40: ('water', '#2a78d6'),
    41: ('water', '#2a78d6'),
    1: ('signal', '#2ca02c'),
    2: ('ground', '#eb6834'),
}

# %% load one window of one beam
ph = pd.read_parquet(
    work / 'photons' / f'photons_{BEAM}.parquet',
    columns=['latitude', 'longitude', 'alongtrack', 'h_ph', 'delta_time', 'atl08_class', 'atl24_class'],
    filters=[('latitude', '>=', min(LAT_RANGE)), ('latitude', '<=', max(LAT_RANGE))],
)
if ph.empty:
    raise SystemExit(f'no photons for {BEAM} in latitude {LAT_RANGE}')
x = ph['alongtrack'] / 1000 if X_AXIS == 'alongtrack' else ph['latitude']
xlabel = 'Along-track distance (km)' if X_AXIS == 'alongtrack' else 'Latitude (°N)'
print(f'{len(ph):,} photons')

# %% 3DEP along the track
# Sample the DTM under every photon, then express it as an ellipsoidal height in
# the photons' frame: dem_ell = dem + (h_ph - h_navd88), i.e. undo the same
# ITRF2014 -> NAD83(2011) + NAVD88 shift that was applied to the photon.
paths = dem_paths(Path(cfg['dem_dir']) if Path(cfg['dem_dir']).is_absolute() else HERE / cfg['dem_dir'])
if paths:
    stack = DemStack(paths)
    epoch = 2018.0 + ph['delta_time'].mean() / SECONDS_PER_YEAR
    dx, dy, h_navd = itrf_to_dem_frame(ph['longitude'], ph['latitude'], ph['h_ph'], epoch, stack.crs)
    dem_ell = stack.sample(dx, dy)['elev'] + (ph['h_ph'].to_numpy() - h_navd)
    b = np.floor(ph['alongtrack'].to_numpy() / DEM_BIN_M).astype(np.int64)
    line = pd.DataFrame({'b': b, 'x': x.to_numpy(), 'dem': dem_ell}).groupby('b').median()
    line = line.reindex(np.arange(b.min(), b.max() + 1))   # empty bins -> NaN -> gaps in the line
    if line['dem'].isna().all():
        print('no 3DEP coverage in this window')
else:
    line = None
    print('no staged 3DEP tiles: run stage_3dep.py to get the terrain line')

# %% plot
fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True, sharey=True, constrained_layout=True)
for ax, col, classes, title in ((axes[0], 'atl08_class', ATL08_CLASSES, 'ATL08'),
                                (axes[1], 'atl24_class', ATL24_CLASSES, 'ATL24')):
    seen = set()
    for code, (label, color) in classes.items():
        m = ph[col] == code
        if not m.any():
            continue
        ax.scatter(x[m], ph['h_ph'][m], s=POINT_SIZE, c=color, linewidths=0, rasterized=True,
                   label=None if label in seen else label)
        seen.add(label)
    if line is not None:
        ax.plot(line['x'], line['dem'], color='black', linewidth=1.2, label='3DEP DTM')
    other = ~ph[col].isin(list(classes))
    if other.any():
        print(f'{title}: {other.sum():,} photons with other codes not drawn: {sorted(ph[col][other].unique())}')
    ax.set_title(f'{title} photon classes, {BEAM}', loc='left')
    ax.set_ylabel('Ellipsoidal height (m)')
    ax.grid(color='0.9', linewidth=0.6)
    ax.set_axisbelow(True)
    ax.legend(loc='upper right', markerscale=4, frameon=False)
axes[1].set_xlabel(xlabel)

# zoom on the signal: drop most of the background noise from the y-range
sig = ph['h_ph'][(ph['atl08_class'] > 0) | ph['atl24_class'].isin([1, 2])]
if len(sig):
    lo, hi = sig.quantile([0.001, 0.999])
    pad = max(10.0, 0.1 * (hi - lo))
    axes[0].set_ylim(lo - pad, hi + pad)
plt.show()
