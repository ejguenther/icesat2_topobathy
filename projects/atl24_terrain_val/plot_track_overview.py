"""Overview of the granule: full ATL03 ground track over satellite imagery, with
the stretches that have 3DEP 1 m coverage and the "mountain" segments marked.

Left: map (lon / lat) over Esri World Imagery. Right: terrain height along the
track against latitude (shared y-axis), so the long north-south track can be
read section by section.
  white   ATL03 ground track (all six beams; the pairs are ~3 km apart)
  blue    20 m segments with 3DEP 1 m DTM under them
  orange  mountain segments (DTM >= analysis.mountain.min_elev_m and
          slope >= analysis.mountain.min_slope_deg, from config)

Needs build_segments.py output and internet access for the imagery.
Writes <work_dir>/results/fig_track_overview.png.
Run from this folder (or cell by cell in VS Code):  uv run python plot_track_overview.py
"""
# %% settings
import io
import urllib.parse
import urllib.request
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from phoreal.config import load_config

MAP_PAD_DEG = 4.0               # map margin around the track (degrees of longitude)
IMAGE_HEIGHT_PX = 3000
IMAGERY_URL = 'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/export'
ATTRIBUTION = 'Imagery: Esri, Maxar, Earthstar Geographics'
C_TRACK, C_DEM, C_MTN = 'white', '#2a9df4', '#ff7a1a'

HERE = Path(__file__).resolve().parent if '__file__' in globals() else Path.cwd()
cfg = load_config(HERE / 'config.yaml')


def resolve(p):
    p = Path(p)
    return p if p.is_absolute() else HERE / p


work = resolve(cfg['work_dir'])
mtn = cfg['analysis']['mountain']

# %% data
track, terrain = {}, []
with h5py.File(resolve(cfg['granule']['atl03']), 'r') as f03, h5py.File(resolve(cfg['granule']['atl08']), 'r') as f08:
    for gt in cfg['beams']:
        g = f03[gt + '/geolocation']
        ok = g['segment_ph_cnt'][:] > 0
        track[gt] = (g['reference_photon_lon'][:][ok], g['reference_photon_lat'][:][ok])
        ls = f08[gt + '/land_segments']
        h = ls['terrain/h_te_best_fit'][:].astype(float)
        h[np.abs(h) > 1e30] = np.nan
        terrain.append(pd.DataFrame({'lat': ls['latitude'][:], 'h': h}))
terrain = pd.concat(terrain)

seg = pd.read_parquet(work / 'segments.parquet', columns=['latitude', 'longitude', 'dem_elev', 'slope_deg'])
seg = seg[seg['dem_elev'].notna()]
is_mtn = (seg['dem_elev'] >= mtn['min_elev_m']) & (seg['slope_deg'] >= mtn['min_slope_deg'])
print(f'{len(seg):,} segments with 3DEP, {is_mtn.sum():,} mountain')

lon_all = np.concatenate([v[0] for v in track.values()])
lat_all = np.concatenate([v[1] for v in track.values()])
lat0, lat1 = lat_all.min() - 0.5, lat_all.max() + 0.5
lon0, lon1 = lon_all.min() - MAP_PAD_DEG, lon_all.max() + MAP_PAD_DEG
coslat = np.cos(np.radians((lat0 + lat1) / 2))

# %% imagery (plate carree, so it can sit on lon/lat axes directly)
width_px = round(IMAGE_HEIGHT_PX * (lon1 - lon0) / (lat1 - lat0))
params = {'bbox': f'{lon0},{lat0},{lon1},{lat1}', 'bboxSR': 4326, 'imageSR': 4326,
          'size': f'{width_px},{IMAGE_HEIGHT_PX}', 'format': 'jpg', 'f': 'image'}
try:
    with urllib.request.urlopen(f'{IMAGERY_URL}?{urllib.parse.urlencode(params)}', timeout=120) as r:
        img = plt.imread(io.BytesIO(r.read()), format='jpg')
except Exception as err:
    img = None
    print(f'imagery not available: {err}')

# %% plot
fig, (axm, axp) = plt.subplots(1, 2, figsize=(11, 12), sharey=True, constrained_layout=True,
                               gridspec_kw={'width_ratios': [(lon1 - lon0) * coslat / (lat1 - lat0) * 2.2, 1]})

if img is not None:
    axm.imshow(img, extent=(lon0, lon1, lat0, lat1), origin='upper', interpolation='bilinear')
    axm.text(0.01, 0.003, ATTRIBUTION, transform=axm.transAxes, fontsize=6, color='white', va='bottom')
for i, (lon, lat) in enumerate(track.values()):
    axm.plot(lon, lat, color=C_TRACK, linewidth=0.6, alpha=0.9, label='ATL03 ground track' if i == 0 else None)
axm.scatter(seg['longitude'], seg['latitude'], s=4, color=C_DEM, linewidths=0, rasterized=True, label='3DEP 1 m coverage')
axm.scatter(seg['longitude'][is_mtn], seg['latitude'][is_mtn], s=6, color=C_MTN, linewidths=0, rasterized=True,
            label=f'Mountains (≥ {mtn["min_elev_m"]} m, ≥ {mtn["min_slope_deg"]}° slope)')
axm.set_xlim(lon0, lon1)
axm.set_ylim(lat0, lat1)
axm.set_aspect(1 / coslat)
axm.set_xlabel('Longitude (°)')
axm.set_ylabel('Latitude (°N)')
leg = axm.legend(loc='upper left', fontsize=8, markerscale=3, facecolor='black', edgecolor='none', framealpha=0.6)
for t in leg.get_texts():
    t.set_color('white')

# profile: terrain height vs latitude
axp.scatter(terrain['h'], terrain['lat'], s=1, color='0.75', linewidths=0, rasterized=True, label='ATL08 terrain (all)')
axp.scatter(seg['dem_elev'], seg['latitude'], s=1, color=C_DEM, linewidths=0, rasterized=True, label='3DEP coverage')
axp.scatter(seg['dem_elev'][is_mtn], seg['latitude'][is_mtn], s=2, color=C_MTN, linewidths=0, rasterized=True,
            label='Mountains')
axp.axvline(mtn['min_elev_m'], color='0.4', linewidth=0.8, linestyle='--')
axp.set_xlabel('Terrain height (m)')
axp.set_xlim(left=min(-100, np.nanmin(terrain['h'])))
axp.grid(color='0.9', linewidth=0.6)
axp.set_axisbelow(True)
axp.legend(loc='upper right', fontsize=8, markerscale=6, frameon=False)
axp.tick_params(labelleft=False)

seg_lat = seg['latitude']
fig.suptitle(f'ATL03 track, 3DEP coverage {seg_lat.min():.1f}–{seg_lat.max():.1f}°N, '
             f'{is_mtn.sum():,} mountain segments', fontsize=11, x=0.02, ha='left')
out = work / 'results' / 'fig_track_overview.png'
fig.savefig(out, dpi=250)
print(f'-> {out}')
plt.show()
