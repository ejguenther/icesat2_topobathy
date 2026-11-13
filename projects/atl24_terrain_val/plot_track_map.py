"""Map of the ATL03 track over satellite imagery, zoomed to the 3DEP part, with
the 3DEP coverage and mountain sections drawn as thick bands (slide figure).

The track is cut into latitude bins (BIN_DEG). A bin is drawn
  blue    when >= MIN_SHARE of its 20 m segments have 3DEP 1 m under them,
  orange  when >= MIN_SHARE of its segments are "mountain" (3DEP >= analysis.
          mountain.min_elev_m and slope >= analysis.mountain.min_slope_deg).
All six beams are pooled; the bands are drawn on the centre beam's track.
Imagery: Esri World Imagery, with Esri's boundaries/places labels on top.

Needs build_segments.py output and internet access for the imagery.
Writes <work_dir>/results/fig_track_map.png.
Run from this folder (or cell by cell in VS Code):  uv run python plot_track_map.py
"""
# %% settings
import io
import urllib.parse
import urllib.request
from pathlib import Path

import h5py
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from phoreal.config import load_config

BIN_DEG = 0.02                  # latitude bin for the coverage bands (~2 km)
MIN_SHARE = 0.25                # share of a bin's segments needed to colour it
LAT_PAD = 1.0                   # degrees of margin north/south of the 3DEP part
IMAGE_PX = 2400
BASE_URL = 'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/export'
LABEL_URL = 'https://server.arcgisonline.com/ArcGIS/rest/services/Reference/World_Boundaries_and_Places/MapServer/export'
ATTRIBUTION = 'Imagery and labels: Esri, Maxar, Earthstar Geographics'
C_DEM, C_MTN = '#2a9df4', '#ff7a1a'

HERE = Path(__file__).resolve().parent if '__file__' in globals() else Path.cwd()
cfg = load_config(HERE / 'config.yaml')


def resolve(p):
    p = Path(p)
    return p if p.is_absolute() else HERE / p


work = resolve(cfg['work_dir'])
mtn = cfg['analysis']['mountain']


def fetch(url, bbox, size, fmt):
    params = {'bbox': ','.join(map(str, bbox)), 'bboxSR': 4326, 'imageSR': 4326,
              'size': f'{size[0]},{size[1]}', 'format': fmt, 'transparent': 'true', 'f': 'image'}
    try:
        with urllib.request.urlopen(f'{url}?{urllib.parse.urlencode(params)}', timeout=120) as r:
            return plt.imread(io.BytesIO(r.read()), format='png' if fmt.startswith('png') else 'jpg')
    except Exception as err:
        print(f'not available: {url.split("/services/")[1]}: {err}')
        return None


def bands(df, flag):
    """Track centre line (lon, lat) per bin, NaN where the bin is not flagged (breaks the line)."""
    lon = df['lon'].where(flag)
    return lon.to_numpy(), df['lat'].to_numpy()


# %% data
track = []
with h5py.File(resolve(cfg['granule']['atl03']), 'r') as f:
    for gt in cfg['beams']:
        g = f[gt + '/geolocation']
        ok = g['segment_ph_cnt'][:] > 0
        track.append((g['reference_photon_lon'][:][ok], g['reference_photon_lat'][:][ok]))

seg = pd.read_parquet(work / 'segments.parquet', columns=['latitude', 'longitude', 'dem_elev', 'slope_deg'])
seg['dem'] = seg['dem_elev'].notna()
seg['mtn'] = seg['dem'] & (seg['dem_elev'] >= mtn['min_elev_m']) & (seg['slope_deg'] >= mtn['min_slope_deg'])
b = np.floor(seg['latitude'] / BIN_DEG).astype(int)
bins = seg.groupby(b).agg(lat=('latitude', 'median'), lon=('longitude', 'median'),
                          dem=('dem', 'mean'), mtn=('mtn', 'mean'))
bins = bins.reindex(np.arange(b.min(), b.max() + 1))         # empty bins break the bands
# draw the bands on the centre beam's track, so pooling beams 3 km apart doesn't make the line wiggle
c_lon, c_lat = track[len(track) // 2]
o = np.argsort(c_lat)
bins['lat'] = (bins.index + 0.5) * BIN_DEG
bins['lon'] = np.interp(bins['lat'], c_lat[o], c_lon[o])
print(f'{(bins["dem"] >= MIN_SHARE).sum()} of {len(bins)} bins with 3DEP, '
      f'{(bins["mtn"] >= MIN_SHARE).sum()} mountain')

# square extent around the 3DEP part of the track
dem_lat = seg.loc[seg['dem'], 'latitude']
lat0, lat1 = dem_lat.min() - LAT_PAD, dem_lat.max() + LAT_PAD
coslat = np.cos(np.radians((lat0 + lat1) / 2))
half_lon = (lat1 - lat0) / coslat / 2
lon_c = seg.loc[seg['dem'], 'longitude'].median()
lon0, lon1 = lon_c - half_lon, lon_c + half_lon
bbox = (lon0, lat0, lon1, lat1)
size = (round(IMAGE_PX * (lon1 - lon0) / (lat1 - lat0)), IMAGE_PX)

# %% imagery
img = fetch(BASE_URL, bbox, size, 'jpg')
labels = fetch(LABEL_URL, bbox, size, 'png32')

# %% plot
fig, ax = plt.subplots(figsize=(9, 9 * (lat1 - lat0) / ((lon1 - lon0) * coslat) * 0.97), constrained_layout=True)
if img is not None:
    ax.imshow(img, extent=(lon0, lon1, lat0, lat1), origin='upper', interpolation='bilinear')
if labels is not None:
    ax.imshow(labels, extent=(lon0, lon1, lat0, lat1), origin='upper', interpolation='bilinear')

for i, (lon, lat) in enumerate(track):
    ax.plot(lon, lat, color='white', linewidth=0.8, alpha=0.9, label='ATL03 ground track' if i == 0 else None)
halo = [pe.Stroke(linewidth=10, foreground='black', capstyle='butt'), pe.Normal()]
ax.plot(*bands(bins, bins['dem'] >= MIN_SHARE), color=C_DEM, linewidth=7, solid_capstyle='butt',
        path_effects=halo, label='3DEP 1 m coverage')
ax.plot(*bands(bins, bins['mtn'] >= MIN_SHARE), color=C_MTN, linewidth=7, solid_capstyle='butt',
        label=f'Mountains (≥ {mtn["min_elev_m"]:,} m, slope ≥ {mtn["min_slope_deg"]}°)')

ax.set_xlim(lon0, lon1)
ax.set_ylim(lat0, lat1)
ax.set_aspect(1 / coslat)
ax.set_xlabel('Longitude (°)')
ax.set_ylabel('Latitude (°N)')
leg = ax.legend(loc='lower left', fontsize=10, facecolor='black', edgecolor='none', framealpha=0.65,
                handlelength=2.5)
for t in leg.get_texts():
    t.set_color('white')
ax.text(0.995, 0.004, ATTRIBUTION, transform=ax.transAxes, fontsize=6, color='white', ha='right', va='bottom')

out = work / 'results' / 'fig_track_map.png'
fig.savefig(out, dpi=250)
print(f'-> {out}')
plt.show()
